from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# classifier takes in report text and tags, outputs disease type, impression, and reasoning

class Classification(BaseModel):
    disease_type: str = Field(..., description="The specific disease identified in the report.")
    impression: str = Field(..., description="A concise summary of the key findings.")
    reasoning: str = Field(..., description="The rationale behind the classification decisions.")

def classify_report(report_text: str, tags: dict, llm: ChatGoogleGenerativeAI, prompt_template: str) -> Classification:

    try:
        # Create structured LLM with Pydantic schema
        structured_llm = llm.with_structured_output(Classification)
        
        # Format prompt with report text and tags as variables
        prompt = ChatPromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(report_text=report_text, tags=tags)
        
        # Invoke and return Classification instance directly
        result = structured_llm.invoke(formatted_prompt)
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to classify report: {str(e)}") from e
    