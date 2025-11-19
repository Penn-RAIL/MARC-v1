from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Tags(BaseModel):
    exam_modality: str = Field(..., description="High-level imaging type.")
    contrast_status: str = Field(..., description="How contrast is used, if mentioned.")
    body_region: str = Field(..., description="Primary anatomic region of the study.")
    study_purpose: Optional[str] = Field(None, description="Coarse clinical context.")
    normality: str = Field(..., description="Is this basically normal vs not?")
    exam_type_string: str = Field(..., description="Free-text normalized label Agent 1 composes.")
    needs_downstream_disease_analysis: bool = Field(..., description="Boolean indicating if downstream analysis is needed.")

def tag_report(report_text: str, llm: ChatGoogleGenerativeAI, prompt_template: str) -> Tags:

    try:
        # Create structured LLM with Pydantic schema
        structured_llm = llm.with_structured_output(Tags)
        
        # Format prompt with report text as a variable
        prompt = ChatPromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(report_text=report_text)
        
        # Invoke and return Tags instance directly
        result = structured_llm.invoke(formatted_prompt)
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to tag report: {str(e)}") from e