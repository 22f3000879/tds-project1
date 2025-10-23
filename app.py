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
from typing import List, Optional
from datetime import datetime

import httpx
import git
import psutil
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI

# -------------------------
# Settings
# -------------------------
class Settings(BaseSettings):
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
# Logging
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

# -------------------------
# Models
# -------------------------
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

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="AI Web App Builder", description="Generate & deploy single-file web apps via OpenAI")
background_tasks_list: List[asyncio.Task] = []
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
last_received_task: Optional[dict] = None

# -------------------------
# Helpers
# -------------------------
def verify_secret(secret_from_request: str) -> bool:
    return secret_from_request == settings.STUDENT_SECRET

async def process_attachment_for_llm(attachment_url: str) -> Optional[dict]:
    if not attachment_url or not attachment_url.startswith(("data:", "http")):
        return None
    try:
        if attachment_url.startswith("data:"):
            match = re.search(r"data:(?P<mime>[^;]+);base64,(?P<data>.*)", attachment_url, re.IGNORECASE)
            if not match: return None
            mime = match.group("mime")
            data_bytes = base64.b64decode(match.group("data"))
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
            return {"type": "text", "text": decoded[:20000]}
        return None
    except Exception as e:
        logger.exception(f"Error processing attachment {attachment_url}: {e}")
        return None

def safe_makedirs(path: str): os.makedirs(path, exist_ok=True)

async def save_generated_files_locally(task_id: str, files: dict) -> str:
    base_dir = os.path.join(os.getcwd(), "generated_tasks")
    task_dir = os.path.join(base_dir, task_id)
    safe_makedirs(task_dir)
    logger.info(f"Saving generated files to {task_dir}")
    for fname, content in files.items():
        p = os.path.join(task_dir, fname)
        await asyncio.to_thread(lambda p, c: open(p, "w", encoding="utf-8").write(c), p, content)
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
            except Exception as e:
                logger.exception(f"Failed to save attachment {att.name}: {e}")
    flush_logs()
    return saved

def remove_local_path(path: str):
    if not os.path.exists(path): return
    shutil.rmtree(path, ignore_errors=True)

async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
    gh_user = settings.GITHUB_USERNAME
    gh_token = settings.GITHUB_TOKEN
    headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
    should_clone = (round_index > 1)
    creation_succeeded = False

    async with httpx.AsyncClient(timeout=60) as client:
        if round_index == 1:
            payload = {"name": repo_name, "private": False, "auto_init": True}
            r = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
            if r.status_code == 201:
                creation_succeeded = True
                logger.info(f"GitHub: Created repository {repo_name}")
            elif r.status_code == 422:
                msg = r.json().get("message", "")
                logger.warning(f"Repo create 422: {msg}")
                should_clone = True
            else:
                r.raise_for_status()

        if should_clone:
            repo = await asyncio.to_thread(git.Repo.clone_from, repo_url_auth, local_path)
            return repo
        else:
            repo = git.Repo.init(local_path)
            repo.create_remote('origin', repo_url_auth)
            return repo

async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
    gh_user = settings.GITHUB_USERNAME
    gh_token = settings.GITHUB_TOKEN
    repo_url_http = f"https://github.com/{gh_user}/{repo_name}"

    await asyncio.to_thread(repo.git.add, A=True)
    commit = await asyncio.to_thread(lambda m: repo.index.commit(m), f"Task {task_id} - Round {round_index}")
    commit_sha = getattr(commit, "hexsha", "")

    await asyncio.to_thread(lambda *args: repo.git.branch(*args), '-M', 'main')
    await asyncio.to_thread(lambda r: r.git.push('--set-upstream', 'origin', 'main', force=True), repo)

    async with httpx.AsyncClient(timeout=60) as client:
        pages_api = f"{settings.GITHUB_API_BASE}/repos/{gh_user}/{repo_name}/pages"
        payload = {"source": {"branch": "main", "path": "/"}}
        await client.post(pages_api, json=payload, headers={"Authorization": f"token {gh_token}"})

    pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
    return {"repo_url": repo_url_http, "commit_sha": commit_sha, "pages_url": pages_url}

# -------------------------
# LLM (OpenAI)
# -------------------------
async def call_llm_for_code(prompt: str, task_id: str, attachment_parts: List[dict]) -> dict:
    logger.info(f"[LLM] Generating code for {task_id} using OpenAI GPT")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    system_instructions = (
        "You are an expert full-stack web engineer. "
        "Return THREE files: index.html, README.md, LICENSE. "
        "index.html must use Tailwind via CDN and JS only. "
        "README.md should be professional. LICENSE must be MIT."
        "Output strictly in JSON with keys: index.html, README.md, LICENSE."
    )

    # Add attachment info if any
    attachment_info = ""
    if attachment_parts:
        attachment_info = "\n\nAttachments:\n" + "\n".join(
            [f"- {p.get('type','file')}: {p.get('text','[binary]')[:200]}..." for p in attachment_parts]
        )

    full_prompt = f"{prompt}\n\n{attachment_info}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    files = json.loads(raw)
    required = ["index.html", "README.md", "LICENSE"]
    for key in required:
        if key not in files:
            raise ValueError(f"Missing {key} in LLM response")
    return files

# -------------------------
# Notify
# -------------------------
async def notify_evaluation_server(evaluation_url: str, email: str, task_id: str, round_index: int, nonce: str, repo_url: str, commit_sha: str, pages_url: str) -> bool:
    if not evaluation_url or "example.com" in evaluation_url:
        return False
    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(evaluation_url, json=payload)
        r.raise_for_status()
        return True

# -------------------------
# Orchestration
# -------------------------
async def generate_files_and_deploy(task_data: TaskRequest):
    acquired = False
    try:
        await task_semaphore.acquire(); acquired = True
        task_id = task_data.task; round_index = task_data.round; brief = task_data.brief
        repo_name = task_id.replace(" ","-").lower()
        gh_user = settings.GITHUB_USERNAME; gh_token = settings.GITHUB_TOKEN
        repo_url_auth = f"https://{gh_user}:{gh_token}@github.com/{gh_user}/{repo_name}.git"
        repo_url_http = f"https://github.com/{gh_user}/{repo_name}"
        base_dir = os.path.join(os.getcwd(),"generated_tasks"); local_path = os.path.join(base_dir, task_id)

        if os.path.exists(local_path):
            remove_local_path(local_path)
        safe_makedirs(local_path)

        repo = await setup_local_repo(local_path, repo_name, repo_url_auth, repo_url_http, round_index)

        attachment_parts: List[dict] = []
        for att in task_data.attachments:
            part = await process_attachment_for_llm(att.url)
            if part: attachment_parts.append(part)

        if round_index > 1:
            llm_prompt = f"UPDATE (ROUND {round_index}): Make minimal edits to existing project to implement: {brief}."
        else:
            llm_prompt = f"CREATE (ROUND {round_index}): Build a complete app for: {brief}."

        generated_files = await call_llm_for_code(llm_prompt, task_id, attachment_parts)
        task_dir = await save_generated_files_locally(task_id, generated_files)
        await save_attachments_locally(task_dir, task_data.attachments)
        deployment = await commit_and_publish(repo, task_id, round_index, repo_name)

        await notify_evaluation_server(task_data.evaluation_url, task_data.email, task_id, round_index, task_data.nonce,
                                       deployment["repo_url"], deployment["commit_sha"], deployment["pages_url"])

    except Exception as e:
        logger.exception(f"Task failed: {e}")
    finally:
        if acquired: task_semaphore.release()
        flush_logs()

# -------------------------
# Endpoints
# -------------------------
def _task_done_callback(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc: logger.error(f"Background task exception: {exc}")
    except asyncio.CancelledError:
        logger.warning("Background task cancelled")
    finally:
        flush_logs()

@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest, background_tasks: BackgroundTasks, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret): raise HTTPException(status_code=401, detail="Unauthorized")
    last_received_task = {"task": task_data.task, "round": task_data.round, "brief": task_data.brief[:250], "time": datetime.utcnow().isoformat()+"Z"}
    bg = asyncio.create_task(generate_files_and_deploy(task_data))
    bg.add_done_callback(_task_done_callback); background_tasks_list.append(bg); background_tasks.add_task(lambda: None)
    logger.info(f"Received task {task_data.task}")
    flush_logs()
    return JSONResponse(status_code=200, content={"status":"processing_scheduled","task":task_data.task})
    # New endpoint: allow POST to "/" as well
@app.post("/", status_code=200)
async def root_post(task_data: TaskRequest, background_tasks: BackgroundTasks, request: Request):
    return await receive_task(task_data, background_tasks, request)


@app.get("/")
async def root():
    return {"message":"Service running. POST /ready to submit tasks."}

@app.get("/status")
async def status():
    if last_received_task:
        return {"last_received_task": last_received_task, "running_background_tasks": len([t for t in background_tasks_list if not t.done()])}
    return {"message":"No tasks yet."}

@app.get("/health")
async def health():
    return {"status":"ok", "timestamp": datetime.utcnow().isoformat()+"Z"}

@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path): return PlainTextResponse("Log file not found.", status_code=404)
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END); size = f.tell(); buf = bytearray(); block = 1024
        while size > 0 and len(buf) < lines * 2000:
            read_size = min(block, size); f.seek(size - read_size); buf.extend(f.read(read_size)); size -= read_size
        text = buf.decode(errors="ignore").splitlines(); last_lines = "\n".join(text[-lines:])
        return PlainTextResponse(last_lines)

@app.on_event("startup")
async def startup_event():
    async def keepalive():
        while True:
            logger.info("[KEEPALIVE] heartbeat"); flush_logs()
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
