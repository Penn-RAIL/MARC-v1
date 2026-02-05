from typing import List, Optional, Any, Dict
import os
import json
import re
import base64
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
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
            # Simple retrieval based on input
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
        
        # If image is provided, use multimodal message format
        if image_path and os.path.exists(image_path):
            print(f"[{self.name}] Processing with image: {image_path}")
            
            # Create multimodal message with both text and image
            message_content = [
                {"type": "text", "text": f"System Instructions: {instruction_prompt}\n\n{input_data}"}
            ]
            
            # Add image
            image_data = self._encode_image(image_path)
            message_content.append(image_data)
            
            # Use direct message invocation for multimodal
            response = self.llm.invoke([HumanMessage(content=message_content)])
            
        else:
            # Text-only processing (original behavior)
            prompt = ChatPromptTemplate.from_messages([
                ("system", instruction_prompt),
                ("human", "{input_data}")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({"input_data": input_data})
        
        print(f"--- Running {self.name} ---")
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
            print(f"Warning: Failed to parse evaluation JSON: {e}")
            # Return a default structure if parsing fails
            return {
                "overall_score": 0,
                "agent_scores": {},
                "feedback": {},
                "pass": False
            }
