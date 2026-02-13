"""
Baseline MedGemma Server for MedHELM Evaluation

HELM-compatible server that runs a single MedGemma model WITHOUT the
multi-agent pipeline. This provides the "pure MedGemma" baseline for
comparison against MARC on MedHELM benchmarks.

Usage:
    python baseline_server.py

    Runs on port 8081 by default (MARC pipeline server runs on 8080).

Environment variables:
    BASELINE_HOST       Server host (default: 0.0.0.0)
    BASELINE_PORT       Server port (default: 8081)
    BASELINE_MODEL      HuggingFace model ID (default: google/medgemma-4b-it)
"""

import os
import re
import base64
import logging
import tempfile
import time
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.local_model import LocalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("marc.baseline")


# ---------------------------------------------------------------------------
# Request / Response models matching HELM HTTPModelClient contract
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    num_return_sequences: int = 1
    max_new_tokens: int = 100
    top_p: float = 1.0
    echo_prompt: bool = False
    top_k_per_token: int = 1
    stop_sequences: List[str] = []


class TokenResponse(BaseModel):
    text: str
    logprob: float


class ProcessResponse(BaseModel):
    text: str
    logprob: float
    tokens: List[TokenResponse]
    request_time: float


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

model: Optional[LocalModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_name = os.environ.get("BASELINE_MODEL", "google/medgemma-4b-it")
    logger.info("Loading baseline model: %s", model_name)
    model = LocalModel.get(model_name)
    logger.info("Baseline model ready")
    yield
    logger.info("Baseline server shutting down")


app = FastAPI(
    title="MedGemma Baseline Server",
    description="HELM-compatible single-model baseline for MedHELM comparison",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest) -> ProcessResponse:
    """
    Process a prompt through a single MedGemma call (no pipeline).

    This is the baseline: one model, one call, one response.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_path, text_prompt = _extract_image_from_prompt(request.prompt)

    start_time = time.time()
    try:
        response_text = model.generate(
            text_prompt,
            image_path=image_path,
            max_new_tokens=request.max_new_tokens,
        )
    finally:
        if image_path and image_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(image_path)
            except OSError:
                pass

    elapsed = time.time() - start_time

    # Apply stop sequences
    for stop_seq in request.stop_sequences:
        if stop_seq in response_text:
            response_text = response_text[: response_text.index(stop_seq)]

    if request.echo_prompt:
        response_text = request.prompt + response_text

    return ProcessResponse(
        text=response_text,
        logprob=0.0,
        tokens=[TokenResponse(text=response_text, logprob=0.0)],
        request_time=elapsed,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_image_from_prompt(prompt: str) -> tuple:
    """Extract image from prompt. Same logic as server.py."""
    match = re.match(r"^\[IMAGE:(.*?)\]\s*(.*)", prompt, re.DOTALL)
    if match:
        image_path = match.group(1).strip()
        text_prompt = match.group(2).strip()
        if os.path.exists(image_path):
            return image_path, text_prompt
        return None, prompt

    match = re.match(r"^\[IMAGE_BASE64:([\w/]+):(.*?)\]\s*(.*)", prompt, re.DOTALL)
    if match:
        mime_type = match.group(1)
        b64_data = match.group(2)
        text_prompt = match.group(3).strip()
        try:
            image_bytes = base64.b64decode(b64_data)
            ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}.get(mime_type, ".jpg")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(image_bytes)
            tmp.close()
            return tmp.name, text_prompt
        except Exception:
            return None, prompt

    return None, prompt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("BASELINE_HOST", "0.0.0.0")
    port = int(os.environ.get("BASELINE_PORT", "8081"))
    logger.info("Starting baseline server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
