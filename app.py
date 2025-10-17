# app.py
# FastAPI service: receives tasks, generates a static app via Gemini, deploys to GitHub Pages,
# and notifies the evaluation URL. Hardened against GitHub 422 and Gemini tool-call edge cases.

import os
import re
import json
import base64
import shutil
import asyncio
import logging
import sys
import time
import uuid
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

import httpx
import git
import psutil
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse

# ---- Gemini SDK ----
from google import genai
from google.genai import types

# =========================
# Settings
# =========================
class Settings(BaseSettings):
    # Note: historically named OPENAI_API_KEY in your environment; keep for compatibility.
    OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY")  # Gemini key
    GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
    STUDENT_SECRET: str = Field("", env="STUDENT_SECRET")

    LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
    KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")

    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    GITHUB_PAGES_BASE: Optional[str] = None
    GEMINI_MODEL: str = Field("gemini-1.5-flash", env="GEMINI_MODEL")  # safer default

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
if not settings.GITHUB_PAGES_BASE:
    settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

# =========================
# Logging
# =========================
os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
logger = logging.getLogger("task_receiver")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False

def flush_logs():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass

# =========================
# Models
# =========================
class Attachment(BaseModel):
    name: str
    url: str

class TaskRequest(BaseModel):
    task: str
    email: str
    round: int
    brief: str
    evaluation_url: str
    nonce: str
    secret: str
    attachments: List[Attachment] = []

# =========================
# App & globals
# =========================
app = FastAPI(title="AI Web App Builder", description="Generate & deploy single-file web apps via Gemini")
background_tasks_list: List[asyncio.Task] = []
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
last_received_task: Optional[dict] = None

GEMINI_MODEL = settings.GEMINI_MODEL
GEMINI_API_KEY = settings.OPENAI_API_KEY

# =========================
# LLM tool/function schema
# =========================
class GeneratedFiles(BaseModel):
    """The complete content for all files to be deployed to GitHub."""
    index_html: str = Field(..., description="Single-file HTML (Tailwind CDN + inline JS).")
    readme_md: str = Field(..., description="Professional README.md content.")
    license: str   = Field(..., description="MIT License text.")

def generate_files_for_model(files: GeneratedFiles):
    """Function callable by Gemini (tool call)."""
    return {
        "index.html": files.index_html,
        "README.md": files.readme_md,
        "LICENSE": files.license,
    }

# =========================
# Helpers
# =========================
def verify_secret(secret_from_request: str) -> bool:
    return secret_from_request == settings.STUDENT_SECRET

async def process_attachment_for_llm(attachment_url: str) -> Optional[dict]:
    if not attachment_url or not attachment_url.startswith(("data:", "http")):
        logger.warning(f"Invalid attachment URL: {attachment_url}")
        return None
    try:
        if attachment_url.startswith("data:"):
            m = re.search(r"data:(?P<mime>[^;]+);base64,(?P<data>.*)", attachment_url, re.IGNORECASE)
            if not m:
                return None
            mime = m.group("mime")
            data_bytes = base64.b64decode(m.group("data"))
        else:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(attachment_url)
                resp.raise_for_status()
                mime = resp.headers.get("Content-Type", "application/octet-stream")
                data_bytes = resp.content

        if mime.startswith("image/"):
            return {"mimeType": mime, "data": data_bytes}
        elif mime in ("text/csv", "application/json", "text/plain") or attachment_url.lower().endswith((".csv", ".json", ".txt")):
            decoded = data_bytes.decode("utf-8", errors="ignore")
            if len(decoded) > 20000:
                decoded = decoded[:20000] + "\n\n...TRUNCATED..."
            return {"type": "text", "text": f"ATTACHMENT ({mime}):\n{decoded}"}
        return None
    except Exception as e:
        logger.exception(f"Error processing attachment {attachment_url}: {e}")
        return None

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

async def save_generated_files_locally(task_id: str, files: dict) -> str:
    base_dir = os.path.join(os.getcwd(), "generated_tasks")
    task_dir = os.path.join(base_dir, task_id)
    safe_makedirs(task_dir)
    logger.info(f"Saving generated files to {task_dir}")
    for fname, content in files.items():
        p = os.path.join(task_dir, fname)
        if isinstance(content, str):
            await asyncio.to_thread(lambda p_, c_: open(p_, "w", encoding="utf-8").write(c_), p, content)
            logger.info(f"  saved {fname}")
        else:
            logger.error(f"  Skipped saving {fname}: Content is not a string.")
    flush_logs()
    return task_dir

async def save_attachments_locally(task_dir: str, attachments: List[Attachment]) -> List[str]:
    saved = []
    async with httpx.AsyncClient(timeout=30) as client:
        for att in attachments:
            try:
                if att.url.startswith("data:"):
                    m = re.search(r"base64,(.*)", att.url, re.IGNORECASE)
                    if not m:
                        continue
                    data = base64.b64decode(m.group(1))
                else:
                    resp = await client.get(att.url)
                    resp.raise_for_status()
                    data = resp.content
                p = os.path.join(task_dir, att.name)
                await asyncio.to_thread(lambda p_, d_: open(p_, "wb").write(d_), p, data)
                saved.append(att.name)
                logger.info(f"Saved attachment {att.name}")
            except Exception as e:
                logger.exception(f"Failed to save attachment {att.name}: {e}")
    flush_logs()
    return saved

def remove_local_path(path: str):
    if not os.path.exists(path):
        return
    logger.info(f"Removing local path {path}")
    def _try_rmtree(p):
        try:
            shutil.rmtree(p)
            return True
        except Exception as e:
            logger.warning(f"rmtree attempt failed: {e}")
            return False
    for _ in range(6):
        if _try_rmtree(path):
            return True
        # Windows-only handles; harmless elsewhere
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for f in proc.open_files():
                        try:
                            if os.path.commonpath([os.path.abspath(path), os.path.abspath(f.path)]) == os.path.abspath(path):
                                logger.warning(f"Terminating process {proc.pid} ({proc.name()}) holding {f.path}")
                                try:
                                    proc.terminate()
                                except Exception:
                                    pass
                        except Exception:
                            continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        time.sleep(1.0)
    logger.error(f"Failed to remove {path}")
    return False

# ---------- GitHub helpers with 422 handling ----------
async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
    gh_user = settings.GITHUB_USERNAME
    gh_token = settings.GITHUB_TOKEN
    headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
    should_clone = (round_index > 1)
    creation_succeeded = False

    async with httpx.AsyncClient(timeout=60) as client:
        if round_index == 1:
            try:
                payload = {"name": repo_name, "private": False, "auto_init": True}
                r = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                if r.status_code == 201:
                    creation_succeeded = True
                    logger.info(f"GitHub: Created repository {repo_name}")
                elif r.status_code == 422:
                    text = (r.text or "").lower()
                    try:
                        body = r.json()
                    except Exception:
                        body = {"raw": r.text}
                    logger.warning(f"GitHub 422 for '{repo_name}': {body}")
                    exists_indicators = ["already exists", "name already exists", "repository creation failed"]
                    if any(ind in text for ind in exists_indicators) or any(ind in json.dumps(body).lower() for ind in exists_indicators):
                        logger.info(f"Repository {repo_name} likely exists — will clone.")
                        should_clone = True
                    else:
                        raise httpx.HTTPStatusError(f"GitHub 422: {r.text}", request=r.request, response=r)
                else:
                    r.raise_for_status()
            except Exception as e:
                logger.exception(f"Repo create error: {e}")
                raise

        if should_clone or (round_index > 1 and not creation_succeeded):
            try:
                logger.info(f"Cloning {repo_url_auth} -> {local_path}")
                repo = await asyncio.to_thread(git.Repo.clone_from, repo_url_auth, local_path)
                logger.info("Cloned repo successfully")
                return repo
            except Exception as e:
                logger.warning(f"Clone failed for {repo_name}: {e}")
                if round_index == 1:
                    # recover by creating a new repo with a suffix
                    fallback_suffix = uuid.uuid4().hex[:4]
                    fallback_repo_name = f"{repo_name}-{fallback_suffix}"
                    logger.info(f"Creating fallback repo: {fallback_repo_name}")
                    try:
                        payload = {"name": fallback_repo_name, "private": False, "auto_init": True}
                        r2 = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                        if r2.status_code == 201:
                            repo_url_auth = f"https://{gh_user}:{gh_token}@github.com/{gh_user}/{fallback_repo_name}.git"
                            repo = git.Repo.init(local_path)
                            repo.create_remote('origin', repo_url_auth)
                            logger.info(f"Initialized local repo for fallback {fallback_repo_name}")
                            return repo
                        else:
                            logger.error(f"Fallback repo create failed: {r2.status_code} {r2.text}")
                            r2.raise_for_status()
                    except Exception as e2:
                        logger.exception(f"Fallback repo creation also failed: {e2}")
                        raise
                raise

        # Fresh local repo path (should not happen often)
        repo = git.Repo.init(local_path)
        repo.create_remote('origin', repo_url_auth)
        logger.info("Initialized local repo (no remote clone)")
        return repo

async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
    gh_user = settings.GITHUB_USERNAME
    gh_token = settings.GITHUB_TOKEN
    repo_url_http = f"https://github.com/{gh_user}/{repo_name}"

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            await asyncio.to_thread(repo.git.add, A=True)
            commit_msg = f"Task {task_id} - Round {round_index}"
            commit = await asyncio.to_thread(lambda m: repo.index.commit(m), commit_msg)
            commit_sha = getattr(commit, "hexsha", "")

            # Push main (force only if needed)
            await asyncio.to_thread(lambda r: r.git.branch('-M', 'main'), repo)
            await asyncio.to_thread(lambda r: r.git.push('--set-upstream', 'origin', 'main', '--force'), repo)

            # Configure Pages
            await asyncio.sleep(2)
            pages_api = f"{settings.GITHUB_API_BASE}/repos/{gh_user}/{repo_name}/pages"
            payload = {"source": {"branch": "main", "path": "/"}}
            headers = {"Authorization": f"token {gh_token}"}

            for attempt in range(5):
                resp = await client.get(pages_api, headers=headers)
                if resp.status_code == 200:
                    # site exists -> update
                    await client.put(pages_api, json=payload, headers=headers)
                    break
                else:
                    # create
                    await client.post(pages_api, json=payload, headers=headers)
                    break

            await asyncio.sleep(5)  # wait for build
            pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
            return {"repo_url": repo_url_http, "commit_sha": commit_sha, "pages_url": pages_url}
        except Exception as e:
            logger.exception(f"Commit/publish failed: {e}")
            raise

# ---------- LLM wrapper with robust fallbacks ----------
def _extract_text_from_gemini_response(resp: Any) -> str:
    """Try to pull plain text from a Gemini response for JSON fallback."""
    try:
        if not hasattr(resp, "candidates") or not resp.candidates:
            return ""
        parts = getattr(resp.candidates[0], "content", None)
        if not parts or not hasattr(parts, "parts"):
            return ""
        out = []
        for p in parts.parts:
            t = getattr(p, "text", None)
            if t:
                out.append(t)
        return "\n".join(out).strip()
    except Exception:
        return ""

async def call_llm_for_code(prompt: str, task_id: str, attachment_parts: List[dict]) -> dict:
    logger.info(f"[LLM] Generating code for {task_id} using model: {GEMINI_MODEL}")
    system_prompt = (
        "You are an expert full-stack web engineer.\n"
        "Return files via the function call 'generate_files_for_model' with keys index_html, readme_md, license.\n"
        "If you cannot call the function, return a single JSON object with exactly these keys.\n\n"
        "If the brief asks for ?url= handling:\n"
        " - read ?url, try to load it as an image, run Tesseract.js via CDN; if fails, fallback to attached sample.\n"
        " - UI: image preview, OCR status, input field, submit, result, and show parsed ?url.\n"
        " - Client-only. Use Tailwind CDN + vanilla JS.\n"
    )

    # Build content list
    content_parts: List[Union[types.Part, str]] = [system_prompt, prompt]
    for part in attachment_parts:
        if "mimeType" in part and "data" in part:
            try:
                content_parts.append(types.Part.from_bytes(data=part["data"], mime_type=part["mimeType"]))
            except Exception as e:
                logger.exception(f"Failed to add binary attachment part: {e}")
        elif part.get("type") == "text" and part.get("text"):
            content_parts.append(part["text"])

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Gemini client init failed: {e}")
        raise

    max_retries = 3
    last_text_snapshot = ""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    tools=[generate_files_for_model],
                    temperature=0.0
                )
            )

            if response is None:
                logger.warning("[LLM] Response is None. Retrying...")
                raise ConnectionError("Gemini returned None")

            # Log a truncated raw snapshot for diagnostics
            try:
                raw_str = str(response)
                logger.info(f"[LLM raw] {raw_str[:1500]}")
            except Exception:
                pass

            # Preferred path: tool (function) call
            function_calls = getattr(response, "function_calls", None)
            if function_calls:
                fc = function_calls[0]
                if getattr(fc, "name", "") == "generate_files_for_model":
                    args = getattr(fc, "args", None)
                    if not args:
                        raise ValueError("Function call has no args")
                    generated = dict(args)
                    mapping = [
                        ("index_html", "index.html"),
                        ("readme_md", "README.md"),
                        ("license", "LICENSE"),
                    ]
                    final_files = {}
                    for src, dst in mapping:
                        if src not in generated:
                            raise ValueError(f"Missing required key: {src}")
                        final_files[dst] = str(generated[src])
                    logger.info(f"[LLM] Tool-call success on attempt {attempt+1}")
                    return final_files

            # Fallback: try to parse JSON text from candidates
            txt = _extract_text_from_gemini_response(response)
            if txt:
                last_text_snapshot = txt[:1000]
                try:
                    parsed = json.loads(txt)
                    if all(k in parsed for k in ["index_html", "readme_md", "license"]):
                        logger.info(f"[LLM] JSON fallback success on attempt {attempt+1}")
                        return {
                            "index.html": str(parsed["index_html"]),
                            "README.md": str(parsed["readme_md"]),
                            "LICENSE": str(parsed["license"]),
                        }
                except Exception:
                    # not JSON; continue
                    pass

            raise ValueError("LLM did not return function_calls nor valid JSON content.")

        except Exception as e:
            logger.warning(f"[LLM] Attempt {attempt+1} error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                if last_text_snapshot:
                    logger.error(f"[LLM] Last text snapshot for debug:\n{last_text_snapshot}")
                logger.error("[LLM] Exhausted retries; giving up.", exc_info=True)
                raise

# ---------- Notify evaluator ----------
async def notify_evaluation_server(evaluation_url: str, email: str, task_id: str, round_index: int,
                                   nonce: str, repo_url: str, commit_sha: str, pages_url: str) -> bool:
    if not evaluation_url or "example.com" in evaluation_url or evaluation_url.strip() == "":
        logger.warning("Skipping notify due to invalid URL")
        return False
    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url,
    }
    max_retries = 5
    backoff = 1
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(evaluation_url, json=payload, headers={"Content-Type": "application/json"})
                r.raise_for_status()
                logger.info("Notify succeeded")
                return True
        except Exception as e:
            logger.warning(f"Notify attempt {attempt+1} failed: {e}")
            await asyncio.sleep(backoff)
            backoff *= 2
    logger.error("Notify failed after retries")
    return False

# ---------- Orchestration ----------
async def generate_files_and_deploy(task_data: TaskRequest):
    acquired = False
    try:
        await task_semaphore.acquire()
        acquired = True

        logger.info(f"Start task {task_data.task} round {task_data.round}")
        task_id = task_data.task
        round_index = task_data.round
        attachments = task_data.attachments or []

        repo_name = task_id.replace(" ", "-").lower()
        gh_user = settings.GITHUB_USERNAME
        gh_token = settings.GITHUB_TOKEN
        repo_url_auth = f"https://{gh_user}:{gh_token}@github.com/{gh_user}/{repo_name}.git"
        repo_url_http = f"https://github.com/{gh_user}/{repo_name}"

        base_dir = os.path.join(os.getcwd(), "generated_tasks")
        local_path = os.path.join(base_dir, task_id)

        if os.path.exists(local_path):
            try:
                await asyncio.to_thread(remove_local_path, local_path)
            except Exception as e:
                logger.exception(f"Cleanup failed: {e}")
                raise

        safe_makedirs(local_path)

        # Repo: create or clone with 422 handling
        repo = await setup_local_repo(local_path, repo_name, repo_url_auth, repo_url_http, round_index)

        # Process attachments for LLM
        attachment_parts: List[dict] = []
        attachment_meta_lines = []
        for att in attachments:
            part = await process_attachment_for_llm(att.url)
            if part:
                attachment_parts.append(part)
            k = "image" if att.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")) else "file"
            attachment_meta_lines.append(f"{att.name} ({k}) - url: {att.url}")
        if attachment_meta_lines:
            meta_text = "ATTACHMENTS METADATA:\n" + "\n".join(attachment_meta_lines)
            attachment_parts.append({"type": "text", "text": meta_text})

        # Prompt
        brief = task_data.brief
        if round_index > 1:
            llm_prompt = (
                f"UPDATE (ROUND {round_index}): Make minimal edits to implement: {brief}. "
                "Preserve existing layout/style. Provide full replacement content for index.html, README.md, LICENSE."
            )
        else:
            llm_prompt = (
                f"CREATE (ROUND {round_index}): Build a complete single-file responsive Tailwind web app for: {brief}. "
                "Provide index.html, README.md, and MIT LICENSE. If attachments exist, incorporate them and base64-fallback."
            )
        if attachment_meta_lines:
            llm_prompt += "\n\nProvided attachments:\n" + "\n".join(attachment_meta_lines)

        # Generate files
        generated_files = await call_llm_for_code(llm_prompt, task_id, attachment_parts)

        # Save locally + add attachments
        task_dir = await save_generated_files_locally(task_id, generated_files)
        _ = await save_attachments_locally(task_dir, attachments)

        # Stage → commit → push → enable Pages
        deployment = await commit_and_publish(repo, task_id, round_index, os.path.basename(repo.working_tree_dir))
        repo_url = deployment["repo_url"]
        commit_sha = deployment["commit_sha"]
        pages_url = deployment["pages_url"]
        logger.info(f"Deployed: {pages_url}")

        # Notify evaluator
        await notify_evaluation_server(task_data.evaluation_url, task_data.email, task_id,
                                       round_index, task_data.nonce, repo_url, commit_sha, pages_url)

    except Exception as e:
        logger.exception(f"Task failed: {e}")
    finally:
        if acquired:
            task_semaphore.release()
        flush_logs()

# ---------- Background & endpoints ----------
def _task_done_callback(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc:
            logger.error(f"Background task exception: {exc}")
        else:
            logger.info("Background task finished successfully")
    except asyncio.CancelledError:
        logger.warning("Background task cancelled")
    finally:
        flush_logs()

@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest, background_tasks: BackgroundTasks, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    last_received_task = {
        "task": task_data.task,
        "round": task_data.round,
        "brief": (task_data.brief[:250] + "...") if len(task_data.brief) > 250 else task_data.brief,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    bg = asyncio.create_task(generate_files_and_deploy(task_data))
    bg.add_done_callback(_task_done_callback)
    background_tasks_list.append(bg)
    background_tasks.add_task(lambda: None)  # noop to keep connection alive
    logger.info(f"Received task {task_data.task}")
    flush_logs()
    return JSONResponse(status_code=200, content={"status": "processing_scheduled", "task": task_data.task})

@app.get("/")
async def root():
    return {"message": "Service running. POST /ready to submit tasks."}

@app.get("/status")
async def status():
    if last_received_task:
        active = len([t for t in background_tasks_list if not t.done()])
        return {"last_received_task": last_received_task, "running_background_tasks": active}
    return {"message": "No tasks yet."}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path):
        return PlainTextResponse("Log file not found.", status_code=404)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            buf = bytearray()
            block = 1024
            while size > 0 and len(buf) < lines * 2000:
                read_size = min(block, size)
                f.seek(size - read_size)
                buf.extend(f.read(read_size))
                size -= read_size
        text = buf.decode(errors="ignore").splitlines()
        last_lines = "\n".join(text[-lines:])
        return PlainTextResponse(last_lines)
    except Exception as e:
        logger.exception(f"Logs read failed: {e}")
        return PlainTextResponse(f"Error: {e}", status_code=500)

@app.on_event("startup")
async def startup_event():
    async def keepalive():
        while True:
            try:
                logger.info("[KEEPALIVE] heartbeat")
                flush_logs()
            except Exception:
                pass
            await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
    asyncio.create_task(keepalive())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutdown: cancel background tasks")
    for t in background_tasks_list:
        if not t.done():
            try:
                t.cancel()
            except Exception:
                pass
    await asyncio.sleep(0.5)
    flush_logs()

