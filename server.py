"""
MARC Pipeline Server for MedHELM Evaluation

FastAPI server that exposes the MARC multi-agent pipeline via an HTTP endpoint
compatible with HELM's HTTPModelClient. This allows MedHELM to evaluate the
MARC pipeline as if it were a single language model.

Usage:
    python server.py

    Or with uvicorn directly:
    uvicorn server:app --host 0.0.0.0 --port 8080

Environment variables:
    MARC_SERVER_HOST    Server host (default: 0.0.0.0)
    MARC_SERVER_PORT    Server port (default: 8080)
    MARC_BASE_DIR       MARC project root (default: directory of this file)
    MARC_CONFIG         Config file path relative to base dir (default: config/agents.yaml)
    MARC_PROMPTS_DIR    Prompts directory relative to base dir (default: prompts)
    MARC_OUTPUT_MODE    Output mode: "final" or "combined" (default: final)
"""

import os
import re
import base64
import logging
import tempfile
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import MARCPipeline, PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("marc.server")


# ---------------------------------------------------------------------------
# Request / Response models matching HELM HTTPModelClient contract
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    """Request body matching HELM's HTTPModelClient payload."""
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
    """Response body matching HELM's HTTPModelClient expected format."""
    text: str
    logprob: float
    tokens: List[TokenResponse]
    request_time: float


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

pipeline: Optional[MARCPipeline] = None
output_mode: str = "final"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the MARC pipeline once at server startup."""
    global pipeline, output_mode

    base_dir = os.environ.get(
        "MARC_BASE_DIR", os.path.dirname(os.path.abspath(__file__))
    )
    config_path = os.environ.get("MARC_CONFIG", "config/agents.yaml")
    prompts_dir = os.environ.get("MARC_PROMPTS_DIR", "prompts")
    output_mode = os.environ.get("MARC_OUTPUT_MODE", "final")

    logger.info("Initializing MARC pipeline (base_dir=%s, config=%s)", base_dir, config_path)
    pipeline = MARCPipeline(
        config_path=config_path,
        prompts_dir=prompts_dir,
        base_dir=base_dir,
    )
    logger.info("MARC pipeline ready")
    yield
    logger.info("MARC server shutting down")


app = FastAPI(
    title="MARC Pipeline Server",
    description="MedHELM-compatible API for the MARC multi-agent pipeline",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest) -> ProcessResponse:
    """
    Process a prompt through the MARC multi-agent pipeline.

    This endpoint implements the contract expected by HELM's HTTPModelClient:
      - Input:  prompt, temperature, num_return_sequences, max_new_tokens, ...
      - Output: text, logprob, tokens[], request_time

    Notes:
      - temperature / top_p / top_k_per_token are accepted but NOT forwarded to
        the underlying Gemini models. The pipeline uses its own configured params.
      - num_return_sequences > 1 is not supported (always returns 1 completion).
      - logprob is always 0.0 because a multi-agent pipeline cannot produce
        meaningful token-level log-probabilities.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Extract image from prompt if present
    image_path, text_prompt = _extract_image_from_prompt(request.prompt)

    try:
        result: PipelineResult = pipeline.run(
            input_text=text_prompt,
            image_path=image_path,
        )
    finally:
        # Clean up temp image file if one was created
        if image_path and image_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(image_path)
            except OSError:
                pass

    # Select output based on mode
    if output_mode == "combined":
        response_text = _format_combined(result)
    else:
        response_text = result.final_output

    # Apply stop sequences
    for stop_seq in request.stop_sequences:
        if stop_seq in response_text:
            response_text = response_text[: response_text.index(stop_seq)]

    # Handle echo_prompt
    if request.echo_prompt:
        response_text = request.prompt + response_text

    return ProcessResponse(
        text=response_text,
        logprob=0.0,
        tokens=[TokenResponse(text=response_text, logprob=0.0)],
        request_time=result.request_time,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_image_from_prompt(prompt: str) -> tuple:
    """
    Extract image data from the prompt string if present.

    Supports two formats:
      1. [IMAGE:/path/to/file.jpg] prefix  (MARC native format)
      2. [IMAGE_BASE64:mime_type:data] prefix  (inline base64-encoded image)

    Returns:
        (image_path_or_None, clean_text_prompt)
    """
    # Format 1: file path
    match = re.match(r"^\[IMAGE:(.*?)\]\s*(.*)", prompt, re.DOTALL)
    if match:
        image_path = match.group(1).strip()
        text_prompt = match.group(2).strip()
        if os.path.exists(image_path):
            return image_path, text_prompt
        logger.warning("Image path not found: %s", image_path)
        return None, prompt

    # Format 2: base64 inline
    match = re.match(
        r"^\[IMAGE_BASE64:([\w/]+):(.*?)\]\s*(.*)", prompt, re.DOTALL
    )
    if match:
        mime_type = match.group(1)
        b64_data = match.group(2)
        text_prompt = match.group(3).strip()
        try:
            image_bytes = base64.b64decode(b64_data)
            ext = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp",
            }.get(mime_type, ".jpg")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(image_bytes)
            tmp.close()
            return tmp.name, text_prompt
        except Exception as e:
            logger.warning("Failed to decode base64 image: %s", e)
            return None, prompt

    return None, prompt


def _format_combined(result: PipelineResult) -> str:
    """Format all agent outputs into a single combined string."""
    parts = []
    for key in sorted(result.agent_outputs.keys()):
        parts.append(f"[{key}]\n{result.agent_outputs[key]}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MARC_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("MARC_SERVER_PORT", "8080"))
    logger.info("Starting MARC server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
