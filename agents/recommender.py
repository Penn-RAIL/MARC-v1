from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Recommendation(BaseModel):
    recommendations: List[str] = Field(..., description="A list of follow-up recommendations.")

def recommend_follow_up(
    disease_type: str, 
    impression: str, 
    reason: str, 
    llm: ChatGoogleGenerativeAI, 
    prompt_template: str
) -> Recommendation:
    """
    Generates follow-up recommendations based on disease type, impression, and reason.
    """
    try:
        # Create a structured LLM with the Pydantic schema
        structured_llm = llm.with_structured_output(Recommendation)
        
        # Create the prompt from the template
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Format the prompt with the input variables
        formatted_prompt = prompt.format(
            disease_type=disease_type,
            impression=impression,
            reason=reason
        )
        
        # Invoke the model and return the structured output
        result = structured_llm.invoke(formatted_prompt)
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to generate recommendations: {str(e)}") from e
