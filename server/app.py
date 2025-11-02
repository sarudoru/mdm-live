import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - guidance message if package missing
    raise RuntimeError("openai package is required for the motion server. Install with `pip install openai`." ) from exc


BASE_DIR = Path(__file__).resolve().parent.parent
THREE_VRM_PUBLIC_MOTIONS = BASE_DIR / "three-vrm" / "public" / "motions"
DEFAULT_MODEL_PATH = os.environ.get("MDM_MODEL_PATH", "")
DEFAULT_MODEL_GUIDANCE = float(os.environ.get("MDM_GUIDANCE_PARAM", "2.5"))
LLM_MODEL = os.environ.get("OPENAI_MOTION_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="MDM Motion Conversation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Conversation prompt describing what the character should say and do")
    model_path: str | None = Field(None, description="Override default motion model checkpoint path")


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


PLAN_SCHEMA = {
    "name": "motion_plan",
    "schema": {
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
                    "required": ["description", "duration_seconds"]
                },
                "minItems": 1,
                "maxItems": 6
            }
        },
        "required": ["speech_text", "movements"]
    }
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
                    {"type": "text", "text": SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            },
        ],
        response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA},
        max_output_tokens=800,
    )

    try:
        content = response.output[0].content[0].text  # type: ignore[attr-defined]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
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
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
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
