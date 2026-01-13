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

4. **Configure API key**
    ```bash
    cp .env.example .env
    # Edit .env and add your GOOGLE_API_KEY
    ```

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
