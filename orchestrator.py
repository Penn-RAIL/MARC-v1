import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")

import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from agents.tagger import tag_report
from agents.classifier import classify_report
from agents.recommender import recommend_follow_up

def create_llm():
    """Create and return the LLM instance (called once)."""
    dotenv.load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return llm

# get csv data
report = pd.read_csv("data/healthy_chest_ct_synthetic_radiology_reports.csv")

# get first report
report_text = report.iloc[0]["Report"]

llm = create_llm()

# Agent 1 processing
print("--- Running Agent 1: Tagger ---")
agent_1_template = open("prompts/tagger_prompt.txt").read()
tags = tag_report(report_text, llm, prompt_template=agent_1_template)
print(tags)
print("--- Agent 1: Tagger Complete ---\n")

# Agent 2 processing
print("--- Running Agent 2: Classifier ---")
agent_2_template = open("prompts/classifier_prompt.txt").read()
classification = classify_report(report_text, tags.dict(), llm, prompt_template=agent_2_template)
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
print("--- Agent 3: Recommender Complete ---")