from typing import Any


class OllamaProvider:
    default_model = "MedAIBase/MedGemma1.5:4b"

    def build_llm(self, model_name: str, temperature: float, **kwargs: Any) -> Any:
        from langchain_ollama import ChatOllama

        llm_kwargs = dict(
            model=model_name,
            temperature=temperature,
            num_predict=kwargs.get("num_predict", 768),
            num_ctx=kwargs.get("num_ctx", 3072),
        )
        base_url = kwargs.get("base_url")
        if base_url:
            llm_kwargs["base_url"] = base_url

        return ChatOllama(**llm_kwargs)

    def build_embeddings(self) -> Any:
        from langchain_ollama import OllamaEmbeddings

        # Requires an embedding model pulled, e.g.: ollama pull nomic-embed-text
        return OllamaEmbeddings(model="nomic-embed-text")
