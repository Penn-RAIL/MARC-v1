import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from agents.tagger import tag_report
from agents.classifier import classify_report
from agents.recommender import recommend_follow_up
from agents.orchestrator import evaluate_report
load_dotenv()

def create_llm():
    """Create and return the LLM instance (called once)."""
    load_dotenv("keys.env")
    api_key = os.getenv("GOOGLE_API_KEY") or HARDCODED_GOOGLE_API_KEY
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Provide it via .env, shell export, or HARDCODED_GOOGLE_API_KEY."
        )

    # Ensure the client can read the key even if it came from the hardcoded fallback.
    os.environ["GOOGLE_API_KEY"] = api_key

    model_name = os.getenv("GEMINI_MODEL") or HARDCODED_MEDGEMMA_MODEL or "medgemma"
    llm = ChatGoogleGenerativeAI(model=model_name)
    return llm

# get csv data - both masked (for processing) and unmasked (for evaluation)
masked_report = pd.read_csv("data/postive_chest_ct_synthetic_radiology_reports_masked.csv")
unmasked_report = pd.read_csv("data/postive_chest_ct_synthetic_radiology_reports.csv")

# get first report (masked for agent processing)
report_text = masked_report.iloc[0]["Report"]
# get first report (unmasked original for evaluation/comparison)
original_report_text = unmasked_report.iloc[0]["Report"]

llm = create_llm()

# Agent 1 processing
print("--- Running Agent 1: Tagger ---")
agent_1_template = open("prompts/tagger_prompt.txt").read()
tags = tag_report(report_text, llm, prompt_template=agent_1_template)
print(tags)
print("--- Agent 1: Tagger Complete ---\n")

if (tags.needs_downstream_disease_analysis == False):
    print("Skipping Agent 2 and 3 as no downstream disease analysis is needed.")
    exit(0)

# Agent 2 processing

print("--- Running Agent 2: Classifier ---")
agent_2_template = open("prompts/classifier_prompt.txt").read()
classification = classify_report(report_text, tags.model_dump(), llm, prompt_template=agent_2_template)
print(classification)
print("--- Agent 2: Classifier Complete ---\n")

# Agent 3 processing
print("--- Running Agent 3: Recommender ---")
agent_3_template = open("prompts/recommender_prompt.txt").read()
recommendations = recommend_follow_up(
    disease_type=classification.disease_type,
    impression=classification.impression,
    reason=classification.reasoning,
    llm=llm,
    prompt_template=agent_3_template
)
print(recommendations)
print("--- Agent 3: Recommender Complete ---\n")

# Agent 4 processing
print("--- Running Agent 4: Final Evaluator ---")
agent_4_template = open("prompts/final_evaluator_prompt.txt").read()
final_evaluation = evaluate_report(
    report_text=report_text,
    original_report_text=original_report_text,
    tags=tags.model_dump(),
    classification=classification.model_dump(),
    recommendations=recommendations.model_dump(),
    llm=llm,
    prompt_template=agent_4_template,
)
print(final_evaluation)
print("--- Agent 4: Final Evaluator Complete ---")