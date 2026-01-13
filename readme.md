# MARC v1 – Multi-Agent Reasoning & Coordination

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-agent AI system for processing, analyzing, and extracting structured information from complex documents with **iterative feedback refinement** for maximum accuracy.

## Features

- **Multi-Agent Architecture**: Specialized AI agents work together to sequence complex analysis tasks.
- **Iterative Feedback Loop**: Automatic quality evaluation and refinement (up to 3 iterations).
- **Structured Output**: Extracts core entities, context summaries, and actionable recommendations.
- **Quality Scoring**: Built-in evaluation system with custom scoring rubrics for high-precision results.
- **Flexible Configuration**: Easily add, remove, or modify agents and their models via YAML.
- **Retrieval-Augmented Generation (RAG)**: Enhance agent knowledge with external data sources.

## Architecture

### Specialized Agents

The framework uses a pipeline of specialized agents that can be configured for any workflow:

1. **Information Extraction (Tagger)**
    - Extracts high-level metadata and core entities.
    - Determines initial context and necessity for downstream analysis.
    - Output: Structured metadata and tags.

2. **Categorization (Classifier)**
    - Performs in-depth analysis based on extracted tags.
    - Generates detailed categorizations and reasoning.
    - Output: Specific classifications and rationale.

3. **Action Generation (Recommender)**
    - Generates evidence-based next steps or recommendations.
    - Considers all previous context and analysis.
    - Output: List of actionable items.

4. **Quality Control (Evaluator/Orchestrator)**
    - Scores outputs against target benchmarks or quality thresholds.
    - Generates detailed feedback for agents that underperform.
    - Output: Evaluation scores and refinement instructions.

### Feedback Loop Mechanism

When quality scores fall below the configurable threshold:
1. **Evaluator** generates specific, constructive feedback.
2. Feedback is injected into agent prompts for the next iteration.
3. Agents refine their outputs based on the direct feedback.
4. Process repeats until the quality threshold is met or maximum iterations are reached.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher.
- Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey)).

### Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/Penn-RAIL/MARC-v1.git
    cd MARC-v1
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

---

## Project Walkthrough

The MARC framework automates complex analysis through a multi-agent pipeline. Below is a demonstration of the system processing a simple query:

### Sample Execution Flow
When a user provides a query (e.g., "Analyze this technical request"), the pipeline processes it through sequential stages:

1.  **Extract Agent**: Identifies the core intent and key technical entities.
2.  **Analyze Agent**: Categorizes the request and provides deep reasoning for the classification.
3.  **Action Agent**: Recommends the specific next steps for fulfillment.

```text
--- Agent 1 is working... ---
Output: { "entities": ["Technical Support", "Priority High"], "summary": "User requesting system access." }

--- Agent 2 is working... ---
Output: Category: Access Management. Reason: Request requires infrastructure permissions.

--- Agent 3 is working... ---
Output: Recommendation: Verify user identity and initiate access workflow.
```

### RAG Capabilities
Agents can be equipped with external knowledge via the `context_files` configuration. This allows the system to reference technical documentation, standard operating procedures, or historical data during its decision-making process.

---

## Customizing Your Agents

You can easily customize the pipeline by editing `config/agents.yaml`.

### Adding or Modifying Agents

Each agent in the `config/agents.yaml` file has the following properties:

- `name`: The display name of the agent.
- `model`: The model ID to use (e.g., `gemini-2.0-flash`, `gemini-1.5-pro`).
- `prompt_file`: The filename of the prompt template in the `prompts/` directory.
- `context_files`: (Optional) A list of paths to text files for RAG.

Example:
```yaml
agents:
  - name: "Data Analyst"
    model: "gemini-2.0-flash"
    prompt_file: "analyst_prompt.txt"
    context_files: ["data/knowledge/ops_manual.txt"]
```

### Using Different Models

You can assign more powerful models to complex reasoning tasks and faster models to simpler extraction tasks to optimize for cost and speed.

---

## Usage

**Run the interactive pipeline**:
```bash
python main.py
```

**Run specialized tests**:
```bash
python list_models.py  # Verify available Google models
```

## Project Structure

```
MARC v1/
├── agents/             # Core agent logic
│   └── agent.py        # GenericAgent implementation
├── config/             # Pipeline configuration
│   └── agents.yaml     # YAML-based agent definitions
├── prompts/            # Agent prompt templates
├── data/               # Input data and knowledge bases
│   └── knowledge/      # Text files for RAG
├── main.py             # Main entry point and interactive CLI
├── requirements.txt    # Project dependencies
└── readme.md           # Documentation
```

## Performance & Optimization

- **Quality Thresholds**: Configurable scoring ensures results meet your precision requirements.
- **Multi-Iteration**: The framework can automatically refine its own work based on internal evaluation.
- **Modular Design**: Swapping models or prompts is a no-code operation once agents are defined.

---

**Disclaimer**: This system is a general-purpose framework. Always validate AI-generated content for your specific use case.
