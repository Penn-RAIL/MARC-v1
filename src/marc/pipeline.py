from typing import List, Optional, Any
import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .providers import get_provider, resolve_backend

# RAG embedding backends are imported lazily inside _initialize_rag so that
# users only need the dependencies for the backend they actually run.
try:
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import CharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


def clean_model_output(text: str) -> str:
    """
    Normalize raw LLM text before downstream JSON extraction.

    Strips leading/trailing whitespace and any zero-width or non-breaking
    characters, and normalizes line endings. Does NOT parse JSON — callers
    handle that separately.
    """
    if text is None:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-breaking spaces and zero-width characters that break parsers
    for bad in (" ", "​", "﻿"):
        text = text.replace(bad, " " if bad == " " else "")

    text = re.sub(r"<unused\d+>", " ", text)
    return text.strip()


class GenericAgent:
    def __init__(
        self,
        name: str,
        prompt_template: str,
        model_name: Optional[str] = None,
        context_files: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_predict: int = 768,
        num_ctx: int = 3072,
        ollama_base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.backend = resolve_backend()
        self.provider = get_provider(self.backend)
        self.model_name = model_name or self.provider.default_model
        self.context_files = context_files or []
        self.retriever = None

        # NOTE: the Ollama num_predict/num_ctx values reproduce the exact
        # call the pipeline has been benchmarked with — do not change them
        # without re-running the benchmarks. GoogleProvider ignores these,
        # since Gemini doesn't use them.
        self.llm = self.provider.build_llm(
            model_name=self.model_name,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            base_url=ollama_base_url,
        )

        # Initialize RAG if context files exist
        if self.context_files:
            self._initialize_rag()

    def _initialize_rag(self):
        """Ingests context files into a vector store."""
        if not RAG_AVAILABLE:
            print(f"[{self.name}] Warning: RAG dependencies not installed. Skipping RAG.")
            return

        documents = []
        for file_path in self.context_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    text = f.read()
                    documents.append(
                        Document(page_content=text, metadata={"source": file_path})
                    )
            else:
                print(f"Warning: Context file {file_path} not found.")

        if not documents:
            return

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        embeddings = self.provider.build_embeddings()

        # Ephemeral in-memory vector store, one per agent (V1 simplicity).
        self.vectorstore = Chroma.from_documents(docs, embeddings)
        self.retriever = self.vectorstore.as_retriever()
        print(
            f"[{self.name}] RAG initialized with {len(docs)} chunks "
            f"from {len(self.context_files)} files."
        )

    def run(self, input_text: str, previous_agent_output: Optional[str] = None) -> str:
        """Runs the agent on the input query."""

        context_str = ""
        if self.retriever:
            # Simple retrieval based on input
            relevant_docs = self.retriever.invoke(input_text)
            context_str = "\n\nRelevant Context from Knowledge Base:\n" + "\n".join(
                [d.page_content for d in relevant_docs]
            )

        # Decomposer-generated prompts reference {input} and
        # {previous_agent_output} directly in the system template, so both
        # are supplied as template variables. But static prompt files (e.g.
        # prompts/agent_2_prompt.txt) only reference {input} - if
        # previous_agent_output were *only* passed as a template variable,
        # it would silently never reach the LLM for those prompts, since
        # str.format() ignores unused kwargs. To guarantee prior-step output
        # and RAG context are never dropped, they're always also folded into
        # the human message content explicitly, regardless of what the
        # system prompt text references.
        prev = previous_agent_output or ""
        if context_str:
            prev = f"{prev}\n{context_str}"

        human_content = f"Original Input: {input_text}"
        if prev:
            human_content += f"\n\nInput from previous step:\n{prev}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_template),
                ("human", "{human_content}"),
            ]
        )

        chain = prompt | self.llm

        print(f"--- Running {self.name} ---")
        response = chain.invoke({
            "input": input_text,
            "previous_agent_output": prev,
            "human_content": human_content,
        })

        return clean_model_output(response.content)


def run_pipeline(pipeline: List["GenericAgent"], user_input: str) -> Optional[str]:
    """Runs every agent in sequence, threading each agent's output into the next."""
    previous_output = None
    for agent in pipeline:
        output = agent.run(input_text=user_input, previous_agent_output=previous_output)
        previous_output = output
    return previous_output
