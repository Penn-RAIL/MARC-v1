from typing import Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RadiologistEvaluation(BaseModel):
    classification_score: float = Field(
        ..., description="Score (0-1) for disease classification accuracy and alignment with report findings."
    )
    impression_score: float = Field(
        ..., description="Score (0-1) reflecting the clinical coherence and clarity of the impression."
    )
    impression_reasoning: str = Field(..., description="Radiologist rationale supporting the impression score.")
    follow_up_recommended: bool = Field(
        ..., description="Whether the radiologist agrees a follow-up is needed based on the report."
    )
    follow_up_reasoning: str = Field(..., description="Reasoning behind the yes/no follow-up decision.")
    overall_report_score: float = Field(
        ..., description="Overall radiologist quality score (0-1) considering safety and clinical actionability."
    )
    additional_notes: Optional[str] = Field(
        None, description="Optional concise notes on safety flags or alignment issues."
    )


def evaluate_report(
    report_text: str,
    tags: Dict,
    classification: Optional[Dict],
    recommendations: Optional[Dict],
    llm: ChatGoogleGenerativeAI,
    prompt_template: str,
    original_report_text: str,  
) -> RadiologistEvaluation:
    """
    Final radiologist role evaluator that synthesizes upstream agent outputs and the source report,
    producing four grades: classification quality, impression quality (with reasoning), a yes/no
    follow-up decision (with reasoning), and an overall report quality score.
    
    Args:
        report_text: The masked/processed report text used by agents
        original_report_text: The unmasked original report for comparison/ground truth
    """
    try:
        structured_llm = llm.with_structured_output(RadiologistEvaluation)

        prompt = ChatPromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(
            report_text=report_text,
            original_report_text=original_report_text,
            tags=tags,
            classification=classification or {},
            recommendations=recommendations or {},
        )

        result = structured_llm.invoke(formatted_prompt)
        return result
    except Exception as e:
        raise ValueError(f"Failed to evaluate report: {str(e)}") from e