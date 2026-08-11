import os
from typing import Any

# The embedding model previously used here (models/embedding-001) has been
# retired by Google (returns 404 NOT_FOUND). gemini-embedding-001 is the
# current stable replacement as of this writing.
EMBEDDING_MODEL = "models/gemini-embedding-001"


class GoogleProvider:
    default_model = "gemini-1.5-flash"

    def build_llm(self, model_name: str, temperature: float, **kwargs: Any) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            from dotenv import find_dotenv, load_dotenv
            load_dotenv(find_dotenv("keys.env", usecwd=True))
            api_key = os.getenv("GOOGLE_API_KEY")

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

    def build_embeddings(self) -> Any:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
