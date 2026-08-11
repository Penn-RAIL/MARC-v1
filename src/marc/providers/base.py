from typing import Any, Protocol


class Provider(Protocol):
    """Constructs the LLM and embeddings clients for a specific model backend."""

    default_model: str

    def build_llm(self, model_name: str, temperature: float, **kwargs: Any) -> Any:
        """Returns a LangChain chat model (a Runnable with .invoke())."""
        ...

    def build_embeddings(self) -> Any:
        """Returns a LangChain embeddings client, for RAG."""
        ...
