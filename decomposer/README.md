# MARC Decomposer

The decomposer is an Ollama-based MARC pipeline with an optional task decomposition trigger.

## Usage

### Run MARC with Ollama (manual prompts)
```bash
python3 decomposer/main.py
```
Loads the existing pipeline from `config/decomposed_agents.yaml` and runs it.
You are expected to have written the agent prompts manually:
- `prompts/agent_1_prompt.txt`
- `prompts/agent_2_prompt.txt`
- `prompts/agent_3_prompt.txt`

### Run MARC with Ollama + Decomposer
```bash
python3 decomposer/main.py --decompose
```
1. You describe your task in one message
2. The decomposer generates all 3 agent prompts automatically
3. Writes them to `prompts/agent_1/2/3_prompt.txt` and `config/decomposed_agents.yaml`
4. Pipeline starts immediately

## Files

| File | Description |
|---|---|
| `decomposer.py` | DecomposerAgent class — talks to Ollama, parses JSON pipeline spec |
| `decomp.txt` | System prompt for the decomposer LLM |
| `main.py` | Entry point — runs MARC with or without the decomposer trigger |

## Models

- **Decomposer**: `gemma4:e4b` (via Ollama)
- **Pipeline agents**: `MedAIBase/MedGemma1.5:4b` (via Ollama)

Both can be changed in `decomposer/main.py` and `decomposer/decomposer.py`.
