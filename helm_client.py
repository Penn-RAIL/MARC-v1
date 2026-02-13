"""
Custom HELM Client for the MARC Multi-Agent Pipeline

This module provides an in-process HELM Client that runs the full MARC pipeline
directly, without requiring a separate HTTP server. Use this when you want to
run MedHELM evaluation with MARC in the same Python process.

Usage in model_deployments.yaml:
    client_spec:
      class_name: "helm_client.MARCClient"
      args:
        marc_base_dir: "/path/to/MARC-v1"
        config_path: "config/agents.yaml"
        prompts_dir: "prompts"

Requires: pip install "crfm-helm[medhelm]"
"""

import os
import sys
import time
import logging
from typing import Optional

logger = logging.getLogger("marc.helm_client")

# Guard HELM imports — this module should be importable even without crfm-helm
# installed, so that the rest of MARC doesn't break.
try:
    from helm.clients.client import CachingClient
    from helm.common.cache_config import CacheConfig
    from helm.common.request import (
        Request,
        RequestResult,
        GeneratedOutput,
        Token,
    )
    HELM_AVAILABLE = True
except ImportError:
    HELM_AVAILABLE = False
    # Define stub base class so the module can be imported without helm
    class CachingClient:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            pass
    CacheConfig = None  # type: ignore[assignment,misc]

from pipeline import MARCPipeline


class MARCClient(CachingClient):
    """
    HELM Client that wraps the MARC multi-agent pipeline.

    Each HELM Request triggers a full pipeline execution:
    Agent 1 (Tagger) -> Agent 2 (Classifier) -> Agent 3 (Recommender),
    with an optional evaluator feedback loop.
    """

    def __init__(
        self,
        cache_config: "CacheConfig" = None,
        marc_base_dir: Optional[str] = None,
        config_path: str = "config/agents.yaml",
        prompts_dir: str = "prompts",
    ):
        if not HELM_AVAILABLE:
            raise ImportError(
                "crfm-helm is required for MARCClient. "
                "Install it with: pip install 'crfm-helm[medhelm]'"
            )

        super().__init__(cache_config=cache_config)

        base_dir = marc_base_dir or os.path.dirname(os.path.abspath(__file__))

        # Ensure MARC project is on sys.path so imports work
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)

        self.pipeline = MARCPipeline(
            config_path=config_path,
            prompts_dir=prompts_dir,
            base_dir=base_dir,
        )
        logger.info("MARCClient initialized (base_dir=%s)", base_dir)

    def make_request(self, request: "Request") -> "RequestResult":
        """
        Run the MARC pipeline for a single HELM request.

        Extracts the prompt text (from request.prompt or request.messages),
        optional image data (from request.multimodal_prompt), runs the
        full pipeline, and returns a HELM RequestResult.
        """
        # Extract prompt text
        if request.prompt:
            prompt_text = request.prompt
        elif request.messages:
            # Concatenate chat messages into a single prompt
            prompt_text = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in request.messages
            )
        else:
            prompt_text = ""

        # Extract image from multimodal_prompt if present
        image_path = self._extract_image(request)

        # Run the full pipeline
        result = self.pipeline.run(
            input_text=prompt_text,
            image_path=image_path,
        )

        response_text = result.final_output

        # Apply stop sequences
        if request.stop_sequences:
            for stop_seq in request.stop_sequences:
                if stop_seq in response_text:
                    response_text = response_text[: response_text.index(stop_seq)]

        # Build HELM response
        completion = GeneratedOutput(
            text=response_text,
            logprob=0.0,
            tokens=[Token(text=response_text, logprob=0.0)],
        )

        return RequestResult(
            success=True,
            completions=[completion],
            embedding=[],
            cached=False,
            request_time=result.request_time,
        )

    def _extract_image(self, request: "Request") -> Optional[str]:
        """Extract image file path from a HELM multimodal request."""
        if not hasattr(request, "multimodal_prompt") or request.multimodal_prompt is None:
            return None

        try:
            for media_obj in request.multimodal_prompt.media_objects:
                if hasattr(media_obj, "is_type") and media_obj.is_type("image"):
                    if hasattr(media_obj, "location") and media_obj.location:
                        return media_obj.location
        except (AttributeError, TypeError):
            logger.debug("Could not extract image from multimodal_prompt")

        return None
