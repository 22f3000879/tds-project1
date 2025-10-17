# api_app.py
# Full server: receives tasks, generates code using Gemini, and handles deployment.

import os
import re
import json
import base64
import shutil
import asyncio
import logging
import sys
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

import httpx
import git
import psutil
# --- LLM IMPORTS ---
from google import genai
from google.genai import types
# --- Pydantic Imports ---
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse

# -------------------------
# Tool Schema Definition (Gemini Native)
# -------------------------

# 1. Define the desired output structure using Pydantic BaseModel
class GeneratedFiles(BaseModel):
    """The complete content for all files to be deployed to GitHub."""
    # Ensure field names are snake_case for Python/Pydantic
    index_html: str = Field(..., description="The full, single-file HTML, including Tailwind CDN and JavaScript.")
    readme_md: str = Field(..., description="The professional markdown content for README.md.")
    license: str = Field(..., description="The full content for the MIT LICENSE file.")

# 2. Define the Python function the model is intended to call
def generate_files_for_model(files: GeneratedFiles):
    """Returns the generated file contents as a dictionary with standard file keys."""
    # Convert Pydantic object keys (snake_case) back to deployment keys (dot/hyphen)
    return {
        "index.html": files.index_html,
        "README.md": files.readme_md,
        "LICENSE": files.license,
    }

# -------------------------
# Settings
# -------------------------
class Settings(BaseSettings):
    # OPENAI_API_KEY holds the Gemini API Key value in Render environment
    OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY") 
    GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
    STUDENT_SECRET: str = Field("", env="STUDENT_SECRET")
    LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
    KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")
    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    GITHUB_PAGES_BASE: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
if not settings.GITHUB_PAGES_BASE:
    settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

# -------------------------
# Logging, Models, App & globals (standard)
# -------------------------
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
            try: h.flush()
            except Exception: pass
    except Exception: pass

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

app = FastAPI(title="AI Web App Builder", description="Generate & deploy single-file web apps via Gemini")
background_tasks_list: List[asyncio.Task] = []
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
last_received_task: Optional[dict] = None

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = settings.OPENAI_API_KEY 

# -------------------------
# Attachment helpers (unchanged from final version)
# -------------------------
def verify_secret(secret_from_request: str) -> bool:
    return secret_from_request == settings.STUDENT_SECRET

async def process_attachment_for_llm(attachment_url: str) -> Optional[dict]:
    if not attachment_url or not attachment_url.startswith(("data:", "http")):
        logger.warning(f"Invalid attachment URL: {attachment_url}")
        return None
    try:
        if attachment_url.startswith("data:"):
            match = re.search(r"data:(?P<mime>[^;]+);base64,(?P<data>.*)", attachment_url, re.IGNORECASE)
            if not match: return None
            mime = match.group("mime")
            b64 = match.group("data")
            data_bytes = base64.b64decode(b64)
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
            if len(decoded) > 20000: decoded = decoded[:20000] + "\n\n...TRUNCATED..."
            return {"type":"text", "text": f"ATTACHMENT ({mime}):\n{decoded}"}
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
             await asyncio.to_thread(lambda p, c: open(p, "w", encoding="utf-8").write(c), p, content)
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
                    if not m: continue
                    data = base64.b64decode(m.group(1))
                else:
                    resp = await client.get(att.url)
                    resp.raise_for_status()
                    data = resp.content
                p = os.path.join(task_dir, att.name)
                await asyncio.to_thread(lambda p, d: open(p, "wb").write(d), p, data)
                saved.append(att.name)
                logger.info(f"Saved attachment {att.name}")
            except Exception as e:
                logger.exception(f"Failed to save attachment {att.name}: {e}")
    flush_logs(); return saved

def remove_local_path(path: str):
    if not os.path.exists(path): return
    logger.info(f"Removing local path {path}")
    def _try_rmtree(p):
        try: shutil.rmtree(p); return True
        except Exception as e: logger.warning(f"rmtree attempt failed: {e}"); return False
    for i in range(6):
        if _try_rmtree(path): return True
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for f in proc.open_files():
                        try:
                            if os.path.commonpath([os.path.abspath(path), os.path.abspath(f.path)]) == os.path.abspath(path):
                                logger.warning(f"Terminating process {proc.pid} ({proc.name()}) holding {f.path}")
                                try: proc.terminate()
                                except Exception: pass
                        except Exception: continue
                except (psutil.NoSuchProcess, psutil.AccessDenied): continue
        except Exception: pass
        time.sleep(1.0)
    logger.error(f"Failed to remove {path}"); return False

# Git helpers (unchanged)
async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
    gh_user = settings.GITHUB_USERNAME; gh_token = settings.GITHUB_TOKEN
    headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
    should_clone = (round_index > 1); creation_succeeded = False
    if round_index == 1:
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                payload = {"name": repo_name, "private": False, "auto_init": True}
                r = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                if r.status_code == 201: creation_succeeded = True
                elif r.status_code == 422:
                    msg = r.json().get("message", "");
                    if "already exists" in msg: should_clone = True
                    else: r.raise_for_status()
                else: r.raise_for_status()
            except Exception as e: logger.exception(f"Repo create error: {e}"); raise
    if should_clone or (round_index > 1 and not creation_succeeded):
        try:
            repo = await asyncio.to_thread(git.Repo.clone_from, repo_url_auth, local_path)
            logger.info("Cloned repo"); return repo
        except Exception as e: logger.exception(f"Clone failed: {e}"); raise
    else:
        repo = git.Repo.init(local_path); repo.create_remote('origin', repo_url_auth)
        logger.info("Initialized local repo"); return repo

async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
    gh_user = settings.GITHUB_USERNAME; gh_token = settings.GITHUB_TOKEN
    repo_url_http = f"https://github.com/{gh_user}/{repo_name}"
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            await asyncio.to_thread(repo.git.add, A=True)
            msg = f"Task {task_id} - Round {round_index}"; commit = await asyncio.to_thread(lambda m: repo.index.commit(m), msg)
            commit_sha = getattr(commit, "hexsha", ""); await asyncio.to_thread(lambda *args: repo.git.branch(*args), '-M', 'main')
            await asyncio.to_thread(lambda r: r.git.push('--set-upstream', 'origin', 'main', force=True), repo); await asyncio.sleep(2)
            pages_api = f"{settings.GITHUB_API_BASE}/repos/{gh_user}/{repo_name}/pages"; payload = {"source": {"branch": "main", "path": "/"}}
            for attempt in range(5):
                try:
                    resp = await client.get(pages_api, headers={"Authorization": f"token {gh_token}"})
                    if resp.status_code == 200: await client.put(pages_api, json=payload, headers={"Authorization": f"token {gh_token}"})
                    else: await client.post(pages_api, json=payload, headers={"Authorization": f"token {gh_token}"})
                    break
                except httpx.HTTPStatusError as e:
                    text = e.response.text.lower() if e.response and e.response.text else ""
                    if e.response.status_code == 422 and "main branch must exist" in text and attempt < 4:
                        await asyncio.sleep(2 ** attempt); continue
                    raise
            await asyncio.sleep(5); pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
            return {"repo_url": repo_url_http, "commit_sha": commit_sha, "pages_url": pages_url}
        except Exception as e: logger.exception(f"Commit/publish failed: {e}"); raise

# LLM wrapper is now Gemini native (using the key stored in OPENAI_API_KEY)
async def call_llm_for_code(prompt: str, task_id: str, attachment_parts: List[dict]) -> dict:
    logger.info(f"[LLM] Generating code for {task_id} using model: {GEMINI_MODEL}")
    system_prompt = (
        "You are an expert full-stack web engineer. You must use the `generate_files_for_model` tool to return "
        "the required files. Your primary goal is to create a single-file, clean, professional web application.\n\n"
        # --- CRITICAL PROMPT FIX ---
        "You MUST return the content using the generate_files_for_model tool. You must not use any other function or return any natural language text. Respond ONLY with the requested tool call."
        # --- END CRITICAL PROMPT FIX ---
        "IMPORTANT: If the brief or user asks for handling a ?url=... parameter or a remote image URL, implement the following behavior in the generated index.html:\n"
        " - Detect a URL parameter named 'url' (e.g., ?url=https://.../image.png). If present, attempt to load that image into an <img> element with crossOrigin='anonymous'.\n"
        " - Use a robust client-side OCR fallback using Tesseract.js (via CDN). Attempt OCR on the loaded image and store the OCRed text.\n"
        " - Provide an input box for the user to type the captcha text and a Submit button. On submit, compare the user's input to the OCRed text (case-insensitive) and show success/failure.\n"
        " - If loading the remote image fails (CORS, network, 404) or OCR fails, default to an attached sample image that is stored in the project root (e.g., './sample.png').\n"
        " - Embed the attached sample image in the generated files by referencing the attachment filename AND include a base64 inline fallback so the page always works.\n"
        " - Avoid server-side calls: app must work client-side in the browser. Handle CORS gracefully: if the remote image is tainted, display an informative message and fall back to sample.\n"
        " - Include minimal accessible UI: image preview, OCR status, input box, submit button, result message, and a 'Try URL' area showing the parsed ?url value.\n\n"
        "ROUND RULES:\n"
        " - Round 1: Create full single-file Tailwind index.html implementing above behavior if requested; include README.md and MIT LICENSE.\n"
        " - Round 2+: Make minimal precise edits to the previously generated files; preserve layout/style.\n\n"
        "FILES:\n"
        " index.html: single-file HTML using Tailwind CDN + vanilla JS + Tesseract.js via CDN. Must reference any attached image file by filename in the root (./<name>) and include base64 fallback.\n"
        " README.md: describe the app, mention attachment usage, and Live Demo link.\n"
        " LICENSE: MIT with [year] and [author].\n"
    )

    content_parts: List[types.Part] = [system_prompt]
    content_parts.append(prompt)
    
    # Add attachments (images and text blocks)
    for part in attachment_parts:
        if "mimeType" in part and "data" in part:
            content_parts.append(
                types.Part.from_bytes(data=part["data"], mime_type=part["mimeType"])
            )
        elif part.get("type") == "text" and part.get("text"):
            content_parts.append(part["text"])

    try:
        client = genai.Client(api_key=settings.OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Gemini client init failed: {e}"); raise

    max_retries = 3
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

            if response.function_calls:
                function_call = response.function_calls[0]
                if function_call.name == "generate_files_for_model":
                    generated = dict(function_call.args)
                    
                    final_files = {}
                    for model_key, deploy_key in [("index_html", "index.html"), ("readme_md", "README.md"), ("license", "LICENSE")]:
                        if model_key not in generated:
                            raise ValueError(f"Missing required argument in model call: {model_key}")
                        final_files[deploy_key] = str(generated[model_key])
                    
                    logger.info(f"[LLM] Successfully generated files on attempt {attempt+1}")
                    return final_files
                else:
                    raise ValueError(f"Model called unexpected function: {function_call.name}")
            else:
                 if response.candidates and response.candidates[0].content and response.candidates[0].content.parts[0].text:
                     error_text = response.candidates[0].content.parts[0].text
                     raise ValueError(f"LLM did not call tool. Response: {error_text[:100]}...")
                 else:
                     raise ValueError("LLM did not call tool and returned no content.")

        except Exception as e:
            logger.warning(f"[LLM] Attempt {attempt+1} error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    raise Exception("LLM generation failed after retries")

# Notify (unchanged)
async def notify_evaluation_server(evaluation_url: str, email: str, task_id: str, round_index: int, nonce: str, repo_url: str, commit_sha: str, pages_url: str) -> bool:
    if not evaluation_url or "example.com" in evaluation_url or evaluation_url.strip()=="":
        logger.warning("Skipping notify due to invalid URL"); return False
    payload = {"email":email,"task":task_id,"round":round_index,"nonce":nonce,"repo_url":repo_url,"commit_sha":commit_sha,"pages_url":pages_url}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(evaluation_url, json=payload); r.raise_for_status()
                logger.info("Notify succeeded"); return True
        except Exception as e:
            logger.warning(f"Notify attempt {attempt+1} failed: {e}")
            if attempt < max_retries-1: await asyncio.sleep(2 ** attempt)
    logger.error("Notify failed after retries"); return False

# Main orchestration (unchanged)
async def generate_files_and_deploy(task_data: TaskRequest):
    acquired = False
    try:
        await task_semaphore.acquire(); acquired = True
        logger.info(f"Start task {task_data.task} round {task_data.round}")
        task_id = task_data.task; round_index = task_data.round; brief = task_data.brief; attachments = task_data.attachments or []
        repo_name = task_id.replace(" ","-").lower(); gh_user = settings.GITHUB_USERNAME; gh_token = settings.GITHUB_TOKEN
        repo_url_auth = f"https://{gh_user}:{gh_token}@github.com/{gh_user}/{repo_name}.git"
        repo_url_http = f"https://github.com/{gh_user}/{repo_name}"
        base_dir = os.path.join(os.getcwd(),"generated_tasks"); local_path = os.path.join(base_dir, task_id)

        if os.path.exists(local_path):
            try: await asyncio.to_thread(remove_local_path, local_path)
            except Exception as e: logger.exception(f"Cleanup failed: {e}"); raise

        safe_makedirs(local_path)
        repo = await setup_local_repo(local_path, repo_name, repo_url_auth, repo_url_http, round_index)
        attachment_parts: List[dict] = []; attachment_meta_lines = []
        for att in attachments:
            part = await process_attachment_for_llm(att.url)
            if part: attachment_parts.append(part)
            lower = att.name.lower(); kind = "image" if lower.endswith((".png",".jpg",".jpeg",".gif")) else "file"
            attachment_meta_lines.append(f"{att.name} ({kind}) - url: {att.url}")
        if attachment_meta_lines:
            meta_text = "ATTACHMENTS METADATA:\n" + "\n".join(attachment_meta_lines)
            attachment_parts.append({"type": "text", "text": meta_text})

        if round_index > 1:
            llm_prompt = (f"UPDATE (ROUND {round_index}): Make minimal edits to existing project to implement: {brief}. " "Preserve structure and style. Provide full replacement content for index.html, README.md, LICENSE.")
        else:
            llm_prompt = (f"CREATE (ROUND {round_index}): Build a complete single-file responsive Tailwind web app for: {brief}. " "Provide index.html, README.md, and MIT LICENSE. If attachments are included, incorporate them.")
        if attachment_meta_lines:
            llm_prompt += "\n\n" + "Provided attachments:\n" + "\n".join(attachment_meta_lines)

        generated_files = await call_llm_for_code(llm_prompt, task_id, attachment_parts)
        task_dir = await save_generated_files_locally(task_id, generated_files)
        saved_names = await save_attachments_locally(task_dir, attachments)
        deployment = await commit_and_publish(repo, task_id, round_index, repo_name)
        repo_url = deployment["repo_url"]; commit_sha = deployment["commit_sha"]; pages_url = deployment["pages_url"]
        logger.info(f"Deployed: {pages_url}")

        await notify_evaluation_server(task_data.evaluation_url, task_data.email, task_id, round_index, task_data.nonce, repo_url, commit_sha, pages_url)

    except Exception as e: logger.exception(f"Task failed: {e}")
    finally:
        if acquired: task_semaphore.release()
        flush_logs()

# Background callback, Endpoints, Keepalive, Shutdown (all unchanged)
def _task_done_callback(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc: logger.error(f"Background task exception: {exc}")
        else: logger.info("Background task finished successfully")
    except asyncio.CancelledError: logger.warning("Background task cancelled")
    finally: flush_logs()

@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest, background_tasks: BackgroundTasks, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret): raise HTTPException(status_code=401, detail="Unauthorized")
    last_received_task = {"task": task_data.task, "round": task_data.round, "brief": (task_data.brief[:250]+"...") if len(task_data.brief)>250 else task_data.brief, "time": datetime.utcnow().isoformat()+"Z"}
    bg = asyncio.create_task(generate_files_and_deploy(task_data))
    bg.add_done_callback(_task_done_callback); background_tasks_list.append(bg); background_tasks.add_task(lambda: None)
    logger.info(f"Received task {task_data.task}"); flush_logs()
    return JSONResponse(status_code=200, content={"status":"processing_scheduled","task":task_data.task})

@app.get("/")
async def root():
    return {"message":"Service running. POST /ready to submit tasks."}

@app.get("/status")
async def status():
    if last_received_task: return {"last_received_task": last_received_task, "running_background_tasks": len([t for t in background_tasks_list if not t.done()])}
    return {"message":"No tasks yet."}

@app.get("/health")
async def health():
    return {"status":"ok", "timestamp": datetime.utcnow().isoformat()+"Z"}

@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path): return PlainTextResponse("Log file not found.", status_code=404)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell(); buf = bytearray(); block = 1024
            while size > 0 and len(buf) < lines * 2000:
                read_size = min(block, size); f.seek(size - read_size); buf.extend(f.read(read_size)); size -= read_size
            text = buf.decode(errors="ignore").splitlines(); last_lines = "\n".join(text[-lines:])
            return PlainTextResponse(last_lines)
    except Exception as e:
        logger.exception(f"Logs read failed: {e}"); return PlainTextResponse(f"Error: {e}", status_code=500)

@app.on_event("startup")
async def startup_event():
    async def keepalive():
        while True:
            try: logger.info("[KEEPALIVE] heartbeat"); flush_logs()
            except Exception: pass
            await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
    asyncio.create_task(keepalive())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutdown: cancel background tasks")
    for t in background_tasks_list:
        if not t.done():
            try: t.cancel()
            except Exception: pass
    await asyncio.sleep(0.5); flush_logs()
