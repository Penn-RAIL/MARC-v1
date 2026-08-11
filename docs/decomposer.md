# MARC Decomposer

The decomposer is an Ollama-based MARC pipeline with an optional task decomposition trigger.

## Usage

### Run MARC with Ollama (manual prompts)
```bash
export MARC_BACKEND=ollama   # or set MARC_BACKEND=ollama in .env
marc run --config config/decomposed_agents.yaml
```
Loads the existing pipeline from `config/decomposed_agents.yaml` and runs it.
You are expected to have written the agent prompts manually:
- `prompts/decomposed_agent_1_prompt.txt`
- `prompts/decomposed_agent_2_prompt.txt`
- `prompts/decomposed_agent_3_prompt.txt`

### Run MARC with Ollama + Decomposer
```bash
export MARC_BACKEND=ollama
marc decompose
```
1. You describe your task in one message
2. The decomposer generates all 3 agent prompts automatically
3. Writes them to `prompts/decomposed_agent_1/2/3_prompt.txt` and `config/decomposed_agents.yaml`
4. Run `marc run --config config/decomposed_agents.yaml` to use the generated pipeline

These filenames are deliberately distinct from `prompts/agent_1/2/3_prompt.txt`, which the
default (non-decomposed) pipeline uses — running the decomposer no longer overwrites the
default pipeline's prompts.

## Files

| File | Description |
|---|---|
| `src/marc/decomposer.py` | `DecomposerAgent` class — talks to Ollama, parses JSON pipeline spec |
| `prompts/decomp.txt` | System prompt for the decomposer LLM |
| `src/marc/cli.py` | `marc decompose` / `marc run` entry points |

## Models

- **Decomposer**: `gemma4:e4b` (via Ollama)
- **Pipeline agents**: `MedAIBase/MedGemma1.5:4b` (via Ollama)

The decomposer's model is set in `DecomposerAgent.__init__` (`src/marc/decomposer.py`); the
generated pipeline's default model is `DECOMPOSED_DEFAULT_MODEL` in `src/marc/config.py`.
