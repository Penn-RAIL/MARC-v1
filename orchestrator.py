import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")

import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from agents.tagger import tag_report

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
agent_1_template = open("prompts/tagger_prompt.txt").read()
tags = tag_report(report_text, llm, prompt_template=agent_1_template)
print(tags)