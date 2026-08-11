import os

import yaml
from pydantic import ValidationError

from .exceptions import ConfigError
from .pipeline import GenericAgent, resolve_backend
from .schemas import PipelineConfig

PROMPTS_DIR = "prompts"
DEFAULT_CONFIG_PATH = "config/agents.yaml"
DECOMPOSED_CONFIG_PATH = "config/decomposed_agents.yaml"
DEFAULT_MODEL = "gemini-1.5-flash"
DECOMPOSED_DEFAULT_MODEL = "MedAIBase/MedGemma1.5:4b"

# Distinct filename prefix from the static agent_1/2/3_prompt.txt files, so
# running the decomposer never overwrites the default pipeline's prompts.
DECOMPOSED_PROMPT_PREFIX = "decomposed_agent"

REQUIRED_ENV_VARS = {
    "gemini": "GOOGLE_API_KEY",
}


def check_environment() -> None:
    """Fails fast with a clear message if the selected backend's credentials are missing."""
    backend = resolve_backend()
    required_var = REQUIRED_ENV_VARS.get(backend)
    if required_var and not os.getenv(required_var):
        raise ConfigError(
            f"{required_var} is not configured.\n"
            "Copy .env.example to .env and add your API key."
        )


def _parse_pipeline_config(config_path: str) -> PipelineConfig:
    if not os.path.exists(config_path):
        raise ConfigError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    try:
        return PipelineConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration in {config_path}:\n{e}") from e


def read_prompt(filename: str, prompts_dir: str = PROMPTS_DIR) -> str:
    path = os.path.join(prompts_dir, filename)
    if not os.path.exists(path):
        raise ConfigError(f"Prompt file not found: {path}")
    return open(path).read()


def load_pipeline(
    config_path: str = DEFAULT_CONFIG_PATH,
    prompts_dir: str = PROMPTS_DIR,
    default_model: str = DEFAULT_MODEL,
) -> list[GenericAgent]:
    """Loads and validates a pipeline of GenericAgents from a YAML config file."""
    pipeline_config = _parse_pipeline_config(config_path)

    pipeline = []
    for agent_cfg in pipeline_config.agents:
        agent = GenericAgent(
            name=agent_cfg.name,
            model_name=agent_cfg.model or default_model,
            prompt_template=read_prompt(agent_cfg.prompt_file, prompts_dir),
            context_files=agent_cfg.context_files,
            temperature=agent_cfg.temperature,
            num_predict=agent_cfg.num_predict,
            num_ctx=agent_cfg.num_ctx,
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
