import os

from .base import Provider
from .google import GoogleProvider
from .ollama import OllamaProvider

PROVIDERS = {
    "gemini": GoogleProvider,
    "ollama": OllamaProvider,
}


def resolve_backend() -> str:
    """Reads MARC_BACKEND lazily (not at import time), so it works regardless
    of whether the caller loads .env before or after importing this module."""
    return os.getenv("MARC_BACKEND", "gemini").lower()


def get_provider(name: str) -> Provider:
    provider_cls = PROVIDERS.get(name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown MARC_BACKEND: {name!r}. Expected one of: {', '.join(PROVIDERS)}."
        )
    return provider_cls()


__all__ = ["Provider", "GoogleProvider", "OllamaProvider", "PROVIDERS", "get_provider", "resolve_backend"]
