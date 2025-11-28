# Radiology Report Classifier

This project is a multi-agent system designed to classify and extract structured information from radiology reports.

## Project Structure

-   `main.py`: The main script that loads data, initializes the language model, and runs the agent pipeline.
-   `agents/`: Contains the different agents responsible for specific tasks.
    -   `tagger.py`: Agent 1, which tags a report with high-level information such as imaging modality, body region, and contrast status.
-   `prompts/`: Stores the prompt templates used by the agents.
-   `data/`: Contains the raw data files (e.g., CSVs of radiology reports).
-   `orchestrator.py`: Scores the output/recommendations of Agents 1-3 on output accuracy, chain-of-thought reasoning.
-   `requirements.txt`: Contains required libraries to run this system
-   `keys.env`: File shell for holding personal Google AI Studio API key.

## How It Works

1.  The `main.py` script loads a radiology report from a CSV file.
2.  It initializes a Google Generative AI model using LangChain.
3.  It invokes the `tag_report` function from the `tagger.py` agent, passing the report text and the model.
4.  The `tagger` agent uses a Pydantic model to define the desired JSON output structure and a prompt template to guide the language model in extracting the information.
5.  The result is a structured `Tags` object containing the classified information, which can then be used for downstream tasks.

