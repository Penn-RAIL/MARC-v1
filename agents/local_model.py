"""
Local Model Singleton for GPU Inference

Loads a HuggingFace model (e.g. MedGemma 4B-IT) once into GPU memory and
provides a simple generate() interface shared across all agents in the pipeline.
"""

import logging
import threading
from typing import Optional, Dict

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

logger = logging.getLogger("marc.local_model")


class LocalModel:
    """
    Singleton wrapper around a local HuggingFace VLM.

    All agents in the MARC pipeline share one model instance to avoid
    loading the same weights multiple times.

    Usage:
        model = LocalModel.get("google/medgemma-4b-it")
        text = model.generate("Analyze this report...", image_path="scan.jpg")
    """

    _instances: Dict[str, "LocalModel"] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str, device: str = "auto"):
        """
        Load model and processor. Use LocalModel.get() instead of calling
        this directly to ensure singleton behavior.
        """
        logger.info("Loading model %s (device=%s)...", model_name, device)
        self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(model_name)

        device_map = device if device == "auto" else {"": device}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).eval()

        logger.info("Model %s loaded successfully", model_name)

    @classmethod
    def get(
        cls,
        model_name: str = "google/medgemma-4b-it",
        device: str = "auto",
    ) -> "LocalModel":
        """Return the singleton instance for a given model name."""
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = cls(model_name, device=device)
            return cls._instances[model_name]

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Generate text from a prompt, optionally with an image.

        Args:
            prompt: The text prompt.
            image_path: Optional path to an image file.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text (assistant response only, prompt excluded).
        """
        # Build message content
        content = []
        image = None

        if image_path:
            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image"})

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if image is not None:
            # Process the image and merge with text inputs
            image_inputs = self.processor.image_processor(
                images=[image], return_tensors="pt"
            )
            inputs.update(image_inputs)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the new tokens (exclude the prompt)
        new_tokens = output_ids[0, input_len:]
        response = self.processor.decode(new_tokens, skip_special_tokens=True)

        return response
