from typing import List, Optional, Any
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
try:
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_text_splitters import CharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Fallback keys (to be replaced by env vars)
HARDCODED_GOOGLE_API_KEY = "TODO_CHANGE_ME"

class GenericAgent:
    def __init__(
        self, 
        name: str, 
        prompt_template: str, 
        model_name: str = "gemini-1.5-flash",
        context_files: Optional[List[str]] = None
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.context_files = context_files or []
        self.retriever = None
        
        # Initialize LLM
        api_key = os.getenv("GOOGLE_API_KEY") 
        if not api_key:
             # Try to load from .env if not in env
             from dotenv import load_dotenv
             load_dotenv("keys.env")
             api_key = os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        
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
                with open(file_path, 'r') as f:
                    text = f.read()
                    documents.append(Document(page_content=text, metadata={"source": file_path}))
            else:
                print(f"Warning: Context file {file_path} not found.")

        if documents:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            
            # Use Google Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # specialized vector store for this agent
            # We use an ephemeral client (in-memory) for simplicity in this V1
            self.vectorstore = Chroma.from_documents(docs, embeddings)
            self.retriever = self.vectorstore.as_retriever()
            print(f"[{self.name}] RAG initialized with {len(docs)} chunks from {len(self.context_files)} files.")

    def run(self, input_text: str, previous_agent_output: Optional[str] = None) -> str:
        """Runs the agent on the input query."""
        
        context_str = ""
        if self.retriever:
            # Simple retrieval based on input
            relevant_docs = self.retriever.invoke(input_text)
            context_str = "\n\nRelevant Context from Knowledge Base:\n" + "\n".join([d.page_content for d in relevant_docs])
        
        # Prepare the full prompt
        # We use a human message to contain the input and previous output
        # to avoid curly brace parsing issues in the previous output.
        
        instruction_prompt = self.prompt_template.replace("{input}", "{{input}}")
        
        input_data = f"Original Input: {input_text}"
        if previous_agent_output:
             input_data += f"\n\nInput from previous step:\n{previous_agent_output}"
             
        if context_str:
            input_data += f"\n{context_str}"
            
        # Using a list of messages is safer than a single large template string
        prompt = ChatPromptTemplate.from_messages([
            ("system", instruction_prompt),
            ("human", "{input_data}")
        ])
        
        chain = prompt | self.llm
        
        print(f"--- Running {self.name} ---")
        response = chain.invoke({"input_data": input_data})
        
        return response.content
