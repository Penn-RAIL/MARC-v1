# Radiology Report Classifier

This project is a multi-agent system designed to classify and extract structured information from radiology reports.

## Project Structure

-   `orchestrator.py`: The main script that loads data, initializes the language model, and runs the agent pipeline.
-   `agents/`: Contains the different agents responsible for specific tasks.
    -   `tagger.py`: Agent 1, which tags a report with high-level information such as imaging modality, body region, and contrast status.
-   `prompts/`: Stores the prompt templates used by the agents.
-   `data/`: Contains the raw data files (e.g., CSVs of radiology reports).

## How It Works

1.  The `orchestrator.py` script loads a radiology report from a CSV file.
2.  It initializes a Google Generative AI model using LangChain.
3.  It invokes the `tag_report` function from the `tagger.py` agent, passing the report text and the model.
4.  The `tagger` agent uses a Pydantic model to define the desired JSON output structure and a prompt template to guide the language model in extracting the information.
5.  The result is a structured `Tags` object containing the classified information, which can then be used for downstream tasks.

## Next Steps

1. Implement Agent 2,3
2. Implement Orchestrator