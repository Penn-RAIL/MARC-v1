# MARC v1 – Multi-Agent Reasoning & Coordination (Report Classification)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-agent AI system for classifying and extracting structured clinical information from radiology reports with **iterative feedback refinement** for improved accuracy.

## Features

- **Multi-Agent Architecture**: Four specialized AI agents work together to analyze reports.
- **Iterative Feedback Loop**: Automatic quality evaluation and refinement (up to 3 iterations).
- **Structured Output**: Extracts disease classification, clinical impressions, and follow-up recommendations.
- **Quality Scoring**: Built-in radiologist-level evaluation with classification, impression, and overall scores.
- **Synthetic Data Support**: Includes masked and unmasked radiology report datasets.
- **Comprehensive Logging**: Detailed evaluation results saved to CSV for analysis.

## Architecture

### Four Specialized Agents

1. **Agent 1 - Tagger**
    - Extracts high-level metadata: imaging modality, body region, contrast status.
    - Determines normality and need for downstream analysis.
    - Output: `Tags` object with structured metadata.

2. **Agent 2 - Classifier**
    - Identifies specific disease type from findings.
    - Generates clinical impression and reasoning.
    - Output: `Classification` with disease type, impression, and rationale.

3. **Agent 3 - Recommender**
    - Generates evidence-based follow-up recommendations.
    - Considers disease type, impression, and clinical context.
    - Output: `Recommendation` list of actionable next steps.

4. **Agent 4 - Evaluator (Orchestrator)**
    - Scores outputs against actual reports (quality threshold: 0.7).
    - Generates detailed feedback for agents that underperform.
    - Output: `RadiologistEvaluation` with scores and feedback.

### Feedback Loop Mechanism

When quality scores fall below the threshold:
1. Evaluator generates specific feedback.
2. Feedback is injected into agent prompts for the next iteration.
3. Agents refine their outputs based on feedback.
4. Process repeats until the quality threshold is met (max 3 iterations).

**Result**: Average score improvement of 0.1+ per iteration ([see test results](FEEDBACK_LOOP_TEST_RESULTS.md)).

## Quick Start

### Prerequisites

- Python 3.8 or higher.
- Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey)).

### Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/report-classifier.git
    cd report-classifier
    ```

2. **Create virtual environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API key**
    ```bash
    cp .env.example .env
    # Edit .env and add your GOOGLE_API_KEY
    ```

### Project Walkthrough

The MARC framework successfully automates complex classification tasks through a multi-agent pipeline. Here is a demonstration of the current system in action:

### Sample Execution Flow
When a user provides a query (e.g., "Hello?"), the pipeline processes it through sequential stages:

1.  **Tagging Agent**: Extracts key entities and creates a structured summary in JSON format.
2.  **Classification Agent**: Analyzes the summary and tags to determine the specific category (e.g., "Greeting") and explains its reasoning.
3.  **Action Agent**: Recommends human-in-the-loop or automated next steps based on the classification.

```text
--- Agent 1 is working... ---
Output: { "entities": ["Greeting"], "summary": "User initiated contact." }

--- Agent 2 is working... ---
Output: Category: Greeting. Reason: The user is initiating a conversation.

--- Agent 3 is working... ---
Output: Recommendation: Acknowledge the greeting and offer assistance.
```

### RAG Capabilities
Agents can be equipped with external knowledge via the `context_files` configuration. For example, `Agent 2` is currently configured to use `data/knowledge/sample_knowledge.txt`, allowing it to reference framework documentation and medical guidelines during its decision-making process.

---

## Customizing Your Agents

You can easily add new agents, change models, or give agents access to external knowledge (RAG) by editing the `config/agents.yaml` file.

### Adding or Modifying Agents

Each agent in the `agents` list in `config/agents.yaml` has the following properties:

-   `name`: The display name of the agent.
-   `model`: The model ID to use (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`).
-   `prompt_file`: The filename of the prompt template located in the `prompts/` directory.
-   `context_files`: (Optional) A list of paths to text files that the agent should use for RAG.

Example configuration:

```yaml
agents:
  - name: "Medical Analyst"
    model: "gemini-1.5-pro"
    prompt_file: "analyst_prompt.txt"
    context_files: ["data/knowledge/medical_guidelines.txt"]
```

### Enabling RAG

To enable RAG for an agent, simply provide a list of file paths in the `context_files` field. The agent will automatically ingest these files and search for relevant information to include in its prompt when processing requests.

### Using Different Models

You can specify different models for different agents. This allows you to use more powerful models for complex tasks and faster, cheaper models for simpler ones. Available models depend on your Google AI Studio account, but common ones include:

-   `gemini-1.5-flash` (Fast and efficient)
-   `gemini-1.5-pro` (Highly capable for complex reasoning)

### Usage

**Run the full pipeline** (processes 100 reports):
```bash
python main.py
```

**Test the feedback loop**:
```bash
python test_feedback_loop.py          # Standard threshold (0.7)
python test_feedback_loop_strict.py   # Strict threshold (0.95)
```

**As a package** (after installation):
```bash
pip install -e .
report-classifier
```

### Output

Results are saved to `logs/evaluation_results.csv` with:
- Classification, impression, and overall quality scores.
- Number of iterations required per report.
- Disease type and clinical recommendations.
- Feedback and reasoning for each evaluation.

## Project Structure

```
report-classifier/
├── agents/                 # Multi-agent system
│   ├── __init__.py        # Package initialization
│   ├── tagger.py          # Agent 1: Metadata extraction
│   ├── classifier.py      # Agent 2: Disease classification
│   ├── recommender.py     # Agent 3: Follow-up recommendations
│   └── orchestrator.py    # Agent 4: Quality evaluation
├── prompts/               # LLM prompt templates
│   ├── tagger_prompt.txt
│   ├── classifier_prompt.txt
│   ├── recommender_prompt.txt
│   └── final_evaluator_prompt.txt
├── data/                  # Radiology report datasets
│   ├── postive_chest_ct_synthetic_radiology_reports.csv
│   └── postive_chest_ct_synthetic_radiology_reports_masked.csv
├── logs/                  # Evaluation results
│   └── evaluation_results.csv
├── main.py               # Main pipeline script
│   ├── visualizer.py         # Results visualization
│   ├── test_feedback_loop.py # Feedback loop tests
│   ├── requirements.txt      # Python dependencies
│   └── setup.py             # Package configuration
```

## Configuration

Edit `.env` or `keys.env`:

```bash
# Required
GOOGLE_API_KEY=your_api_key_here

# Optional (defaults to 'medgemma')
GEMINI_MODEL=gemini-1.5-flash
```

Adjust quality threshold and max iterations in `main.py`:
```python
quality_threshold = 0.7  # Minimum acceptable score (0-1)
max_iterations = 3       # Maximum refinement attempts
```

## Performance

- **Quality Threshold Success Rate**: 85%+ on first iteration (threshold 0.7).
- **Average Iterations**: 1.2 per report.
- **Feedback Loop Improvement**: +0.1 overall score per iteration.
- **Perfect Scores (1.0)**: 40%+ of reports.

See [FEEDBACK_LOOP_TEST_RESULTS.md](FEEDBACK_LOOP_TEST_RESULTS.md) for detailed test results.

## Testing

Run feedback loop tests to verify the system:
```bash
python test_feedback_loop.py          # Quick test with 1 report
python test_feedback_loop_strict.py   # Force multiple iterations
```

## Development

**Install in development mode**:
```bash
pip install -e .
```

**Package for distribution**:
```bash
pip install build
python -m build
# Creates dist/report_classifier-0.1.0-py3-none-any.whl
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [Google Gemini](https://ai.google.dev/).
- Uses synthetic radiology reports for demonstration.
- Inspired by multi-agent AI architectures for healthcare.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Disclaimer**: This system is for research and educational purposes only. Always validate AI-generated medical information with qualified healthcare professionals.
