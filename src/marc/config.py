import os

import yaml

from .pipeline import GenericAgent

PROMPTS_DIR = "prompts"
DEFAULT_CONFIG_PATH = "config/agents.yaml"
DECOMPOSED_CONFIG_PATH = "config/decomposed_agents.yaml"
DEFAULT_MODEL = "gemini-1.5-flash"
DECOMPOSED_DEFAULT_MODEL = "MedAIBase/MedGemma1.5:4b"

# Distinct filename prefix from the static agent_1/2/3_prompt.txt files, so
# running the decomposer never overwrites the default pipeline's prompts.
DECOMPOSED_PROMPT_PREFIX = "decomposed_agent"

_FALLBACK_PROMPT = "You are a helpful assistant. Input: {input}"


def read_prompt(filename: str, prompts_dir: str = PROMPTS_DIR) -> str:
    path = os.path.join(prompts_dir, filename)
    if os.path.exists(path):
        return open(path).read()
    print(f"Warning: Prompt file {path} not found. Using default prompt.")
    return _FALLBACK_PROMPT


def load_pipeline(
    config_path: str = DEFAULT_CONFIG_PATH,
    prompts_dir: str = PROMPTS_DIR,
    default_model: str = DEFAULT_MODEL,
) -> list[GenericAgent]:
    """Loads a pipeline of GenericAgents from a YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        agents_config = config_data.get("agents", [])

    pipeline = []
    for cfg in agents_config:
        agent = GenericAgent(
            name=cfg["name"],
            model_name=cfg.get("model", default_model),
            prompt_template=read_prompt(cfg["prompt_file"], prompts_dir),
            context_files=cfg.get("context_files", []),
            temperature=cfg.get("temperature", 0.0),
            num_predict=cfg.get("num_predict", 768),
            num_ctx=cfg.get("num_ctx", 3072),
        )
        pipeline.append(agent)
    return pipeline


def write_decomposed_pipeline(decomp: dict, prompts_dir: str = PROMPTS_DIR) -> None:
    """Writes a decomposer-generated 3-agent spec to prompt files + a config YAML."""
    agents_config = []
    for i, spec in enumerate(decomp["agents"]):
        filename = f"{DECOMPOSED_PROMPT_PREFIX}_{i + 1}_prompt.txt"
        with open(os.path.join(prompts_dir, filename), "w") as f:
            f.write(spec["prompt_template"])
        print(f"  Agent {i + 1}: {spec['name']} — {spec['role']}")
        agents_config.append({
            "name": spec["name"],
            "model": DECOMPOSED_DEFAULT_MODEL,
            "prompt_file": filename,
            "context_files": [],
            "temperature": 0.0,
            "num_predict": 1024,
            "num_ctx": 3072,
        })

    with open(DECOMPOSED_CONFIG_PATH, "w") as f:
        yaml.dump({"agents": agents_config}, f, default_flow_style=False)
