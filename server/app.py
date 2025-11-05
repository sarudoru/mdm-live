from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


from types import SimpleNamespace

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed


BASE_DIR = Path(__file__).resolve().parent.parent
THREE_VRM_PUBLIC_MOTIONS = BASE_DIR / "three-vrm" / "public" / "motions"
DEFAULT_MODEL_PATH = os.environ.get("MDM_MODEL_PATH", "./save/model000200000.pt")
DEFAULT_MODEL_GUIDANCE = float(os.environ.get("MDM_GUIDANCE_PARAM", "2.5"))
LLM_MODEL = os.environ.get("OPENAI_MOTION_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

app = FastAPI(title="MDM Motion Conversation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for motion JSON files
app.mount("/motions", StaticFiles(directory=str(THREE_VRM_PUBLIC_MOTIONS)), name="motions")


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Conversation prompt describing what the character should say and do")
    model_path: Optional[str] = Field(None, description="Override default motion model checkpoint path")


class MovementPlanItem(BaseModel):
    description: str
    duration_seconds: float


class SegmentResponse(BaseModel):
    description: str
    duration_seconds: float
    motion_url: str


class ConversationResponse(BaseModel):
    speech_text: str
    session_id: str
    segments: List[SegmentResponse]


# Lightweight HTTP client for OpenAI Responses API (no openai package)
import urllib.request
import urllib.error


def _to_obj(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_obj(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_obj(v) for v in x]
    return x


class _ResponsesHTTPClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def create(self, **kwargs):
        data = json.dumps(kwargs).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/responses",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
            return _to_obj(json.loads(body))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"OpenAI API error: {err}") from e


class _HTTPClient:
    def __init__(self, api_key: str, base_url: str):
        self.responses = _ResponsesHTTPClient(api_key, base_url)


# Override client to use HTTP-based OpenAI API
client: Optional[_HTTPClient] = None
if OPENAI_API_KEY:
    client = _HTTPClient(OPENAI_API_KEY, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))


PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "speech_text": {
            "type": "string",
            "description": "What the character should say aloud."
        },
        "movements": {
            "type": "array",
            "description": "Ordered list of motion segments describing the body movement during the speech.",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short instruction for the body movement"
                    },
                    "duration_seconds": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 9.5,
                        "description": "Approximate duration of the movement in seconds"
                    }
                },
                "required": ["description", "duration_seconds"],
                "additionalProperties": False
            },
            "minItems": 1,
            "maxItems": 6
        }
    },
    "required": ["speech_text", "movements"],
    "additionalProperties": False
}


SYSTEM_PROMPT = (
    "You are a creative motion director for a humanoid character. "
    "Given a conversation prompt, you must write what the character should say and break the body performance into "
    "natural mini-segments that cover the full duration of the speech. Each segment should be 2-6 seconds long and include "
    "clear physical actions that can be described succinctly."
)


def call_motion_planner(prompt: str) -> tuple[str, List[MovementPlanItem]]:
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is not set.")

    response = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "motion_plan",
                "schema": PLAN_JSON_SCHEMA
                },
            },
        max_output_tokens=800,
    )

    try:
        output_blocks = getattr(response, "output", [])
        text_payload: Optional[str] = None
        for block in output_blocks:
            for item in getattr(block, "content", []):
                candidate = getattr(item, "text", None)
                if candidate:
                    text_payload = candidate
                    break
            if text_payload:
                break

        if not text_payload:
            raise ValueError("No text content returned from LLM response")

        parsed = json.loads(text_payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {exc}")

    speech_text = parsed.get("speech_text", "").strip()
    movements = parsed.get("movements", [])
    plan_items: List[MovementPlanItem] = []
    for item in movements:
        try:
            plan_items.append(MovementPlanItem(**item))
        except Exception as exc:  # pragma: no cover - validation path
            raise HTTPException(status_code=500, detail=f"Invalid movement plan item: {exc}")

    return speech_text, plan_items


def run_generate_segment(description: str, duration: float, segment_idx: int, session_dir: Path, model_path: str) -> Path:
    duration = max(1.5, min(duration, 9.5))
    segment_filename = f"segment_{segment_idx:02d}.json"
    vrm_motion_path = session_dir / segment_filename

    output_dir = session_dir / f"gen_{segment_idx:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BASE_DIR / "sample" / "generate.py"),
        "--model_path", model_path,
        "--output_dir", str(output_dir),
        "--text_prompt", description,
        "--motion_length", str(duration),
        "--num_samples", "1",
        "--num_repetitions", "1",
        "--guidance_param", str(DEFAULT_MODEL_GUIDANCE),
        "--vrm_motion_path", str(vrm_motion_path),
    ]

    env = os.environ.copy()
    # Ensure PYTHONPATH includes project root so utils package imports resolve
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{BASE_DIR}:{pythonpath}" if pythonpath else str(BASE_DIR)

    
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR), env=env)
    if process.returncode != 0:
        print(f"[DEBUG] Command failed. stderr:\n{process.stderr}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {process.stderr}")

    if not vrm_motion_path.exists():
        raise HTTPException(status_code=500, detail="Motion JSON was not produced by generator.")

    return vrm_motion_path


@app.post("/api/conversation", response_model=ConversationResponse)
async def conversation_endpoint(request: PromptRequest):
    model_path = request.model_path or DEFAULT_MODEL_PATH
    if not model_path:
        raise HTTPException(status_code=400, detail="Model path is not configured. Set MDM_MODEL_PATH env variable or include in request.")

    speech_text, movements = call_motion_planner(request.prompt)

    session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    session_dir = THREE_VRM_PUBLIC_MOTIONS / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    segments: List[SegmentResponse] = []
    for idx, movement in enumerate(movements):
        motion_path = run_generate_segment(movement.description, movement.duration_seconds, idx, session_dir, model_path)
        motion_url = "/motions/" + session_id + "/" + motion_path.name
        segments.append(SegmentResponse(description=movement.description,
                                         duration_seconds=movement.duration_seconds,
                                         motion_url=motion_url))

    return ConversationResponse(speech_text=speech_text, session_id=session_id, segments=segments)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/upload-motion")
async def upload_motion(file: UploadFile = File(...)):
    """Upload a motion JSON file for testing."""
    if not file.filename or not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    
    # Create a test uploads directory
    uploads_dir = THREE_VRM_PUBLIC_MOTIONS / "test-uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the file with a unique name
    unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = uploads_dir / unique_filename
    
    content = await file.read()
    
    # Validate it's valid JSON
    try:
        json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    motion_url = f"/motions/test-uploads/{unique_filename}"
    return {"motion_url": motion_url, "filename": file.filename}


@app.get("/motion-test", response_class=HTMLResponse)
async def motion_test_page():
    """Serve the motion test page."""
    test_page_path = BASE_DIR / "three-vrm" / "public" / "motion-test.html"
    if not test_page_path.exists():
        raise HTTPException(status_code=404, detail="Test page not found")
    return FileResponse(test_page_path)


@app.get("/modelA.vrm")
async def serve_vrm_model():
    """Serve the VRM model file for the motion test page."""
    vrm_path = BASE_DIR / "three-vrm" / "public" / "modelA.vrm"
    if not vrm_path.exists():
        raise HTTPException(status_code=404, detail="VRM model not found")
    return FileResponse(vrm_path, media_type="application/octet-stream")
