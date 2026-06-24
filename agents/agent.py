from typing import List, Optional, Any, Dict
import os
import json
import re
import base64
import logging
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger("marc.agent")

# --- API backend imports (Google Gemini) ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False

# --- RAG imports ---
try:
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import CharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# --- Local backend imports ---
try:
    from agents.local_model import LocalModel
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False


class GenericAgent:
    def __init__(
        self,
        name: str,
        prompt_template: str,
        model_name: str = "gemini-1.5-flash",
        context_files: Optional[List[str]] = None,
        backend: str = "api",
    ):
        """
        Initialize an agent.

        Args:
            name: Agent display name.
            prompt_template: The prompt template string.
            model_name: Model identifier. For backend="api", a Gemini model name.
                        For backend="local", a HuggingFace model ID.
            context_files: Optional list of file paths for RAG context.
            backend: "api" for Google Gemini API, "local" for local GPU inference.
        """
        self.name = name
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.context_files = context_files or []
        self.retriever = None
        self.backend = backend

        if backend == "local":
            if not LOCAL_MODEL_AVAILABLE:
                raise ImportError(
                    "Local backend requires torch and transformers. "
                    "Install with: pip install torch transformers accelerate"
                )
            self.local_model = LocalModel.get(model_name)
            self.llm = None
            logger.info("[%s] Using local model: %s", name, model_name)
        else:
            if not GEMINI_API_AVAILABLE:
                raise ImportError(
                    "API backend requires langchain-google-genai. "
                    "Install with: pip install langchain-google-genai"
                )
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                from dotenv import load_dotenv
                load_dotenv("keys.env")
                api_key = os.getenv("GOOGLE_API_KEY")
            self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
            self.local_model = None
            logger.info("[%s] Using Gemini API: %s", name, model_name)

        # Initialize RAG if context files exist
        if self.context_files:
            self._initialize_rag()

    def _initialize_rag(self):
        """Ingests context files into a vector store."""
        if not RAG_AVAILABLE:
            logger.warning("[%s] RAG dependencies not installed. Skipping RAG.", self.name)
            return

        documents = []
        for file_path in self.context_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    text = f.read()
                    documents.append(Document(page_content=text, metadata={"source": file_path}))
            else:
                logger.warning("Context file %s not found.", file_path)

        if documents:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)

            # Choose embedding model based on backend
            if self.backend == "local":
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                except ImportError:
                    logger.warning(
                        "[%s] sentence-transformers not installed. Skipping RAG.",
                        self.name,
                    )
                    return
            else:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            self.vectorstore = Chroma.from_documents(docs, embeddings)
            self.retriever = self.vectorstore.as_retriever()
            logger.info(
                "[%s] RAG initialized with %d chunks from %d files.",
                self.name, len(docs), len(self.context_files),
            )

    def _encode_image(self, image_path: str) -> dict:
        """Encode image to base64 for multimodal LLM."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Determine MIME type
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(suffix, 'image/jpeg')

        return {
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{encoded_string}"
        }

    def run(self, input_text: str, previous_agent_output: Optional[str] = None, image_path: Optional[str] = None, feedback: Optional[str] = None) -> str:
        """Runs the agent on the input query with optional image input and feedback."""

        context_str = ""
        if self.retriever:
            relevant_docs = self.retriever.invoke(input_text)
            context_str = "\n\nRelevant Context from Knowledge Base:\n" + "\n".join([d.page_content for d in relevant_docs])

        # Prepare the text input
        instruction_prompt = self.prompt_template.replace("{input}", "{{input}}")

        input_data = f"Original Input: {input_text}"
        if previous_agent_output:
             input_data += f"\n\nInput from previous step:\n{previous_agent_output}"

        if context_str:
            input_data += f"\n{context_str}"

        # Inject feedback from evaluator if provided
        if feedback:
            input_data += f"\n\n=== FEEDBACK FOR IMPROVEMENT ===\n{feedback}\n=== Please address the above feedback in your response ==="

        # --- Local backend ---
        if self.backend == "local":
            full_prompt = f"System Instructions: {instruction_prompt}\n\n{input_data}"
            logger.debug("Running %s (local)...", self.name)
            response_text = self.local_model.generate(
                full_prompt,
                image_path=image_path if image_path and os.path.exists(image_path) else None,
            )
            return response_text

        # --- API backend (original behavior) ---
        if image_path and os.path.exists(image_path):
            logger.debug("[%s] Processing with image: %s", self.name, image_path)

            message_content = [
                {"type": "text", "text": f"System Instructions: {instruction_prompt}\n\n{input_data}"}
            ]
            image_data = self._encode_image(image_path)
            message_content.append(image_data)
            response = self.llm.invoke([HumanMessage(content=message_content)])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", instruction_prompt),
                ("human", "{input_data}")
            ])
            chain = prompt | self.llm
            response = chain.invoke({"input_data": input_data})

        logger.debug("--- Running %s ---", self.name)
        return response.content

    @staticmethod
    def parse_evaluation(evaluation_output: str) -> Dict[str, Any]:
        """Parse JSON evaluation output from evaluator agent."""
        try:
            # Try to extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', evaluation_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON in the output
                json_match = re.search(r'{.*}', evaluation_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = evaluation_output

            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse evaluation JSON: %s", e)
            return {
                "overall_score": 0,
                "agent_scores": {},
                "feedback": {},
                "pass": False
            }
