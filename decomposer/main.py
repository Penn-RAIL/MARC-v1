"""
MARC pipeline runner (Ollama).

Run without decomposer (set prompts manually in prompts/decomposed_a/b/c.txt):
    python3 decomposer/main.py

Run with decomposer (generates prompts from task description):
    python3 decomposer/main.py --decompose
"""

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv("keys.env")
load_dotenv()

from agents.agent import GenericAgent

PROMPTS_DIR = "prompts"
DECOMPOSED_CONFIG = "config/decomposed_agents.yaml"
DEFAULT_MODEL = "MedAIBase/MedGemma1.5:4b"


def run_decomposer():
    from decomposer.decomposer import DecomposerAgent

    print("\n" + "=" * 60)
    print("  MARC Decomposer")
    print("=" * 60)
    print("Describe your task and the decomposer will generate a")
    print("3-agent Ollama pipeline automatically.\n")

    task_desc = input("Your task: ").strip()
    if task_desc.lower() == "skip" or not task_desc:
        print("Cancelled.")
        return False

    decomposer = DecomposerAgent()
    print("\n[Decomposer] Generating pipeline...\n")

    prompt = (
        f"{task_desc}\n\n"
        "Now generate the JSON decomposition for this task. "
        "Output the ```json ... ``` block immediately."
    )
    response = decomposer.chat(prompt)

    if not decomposer.has_decomposition():
        print(f"Decomposer: {response}\n")
        response = decomposer.chat(
            "Please output the JSON decomposition block now based on the task I described."
        )

    if not decomposer.has_decomposition():
        print("Decomposer could not generate a decomposition. Try again.")
        return False

    _write_decomposition(decomposer.get_decomposition())
    return True


def _write_decomposition(decomp: dict):
    agents_config = []

    print("\n" + "=" * 60)
    print(f"  Task: {decomp.get('task_summary', '')}")
    print("=" * 60)
    print("Updating agent prompts...")

    for i, spec in enumerate(decomp["agents"]):
        filename = f"agent_{i + 1}_prompt.txt"
        path = os.path.join(PROMPTS_DIR, filename)
        with open(path, "w") as f:
            f.write(spec["prompt_template"])
        print(f"  Agent {i + 1}: {spec['name']} — {spec['role']}")
        agents_config.append({
            "name": spec["name"],
            "model": DEFAULT_MODEL,
            "prompt_file": filename,
            "context_files": [],
            "temperature": 0.0,
            "num_predict": 1024,
            "num_ctx": 3072,
        })

    with open(DECOMPOSED_CONFIG, "w") as f:
        yaml.dump({"agents": agents_config}, f, default_flow_style=False)

    print("\nAgent prompts updated.")


def load_pipeline() -> list:
    if not os.path.exists(DECOMPOSED_CONFIG):
        raise FileNotFoundError(
            f"No config found at {DECOMPOSED_CONFIG}. "
            "Run with --decompose first, or set prompts manually."
        )

    with open(DECOMPOSED_CONFIG) as f:
        config_data = yaml.safe_load(f)

    pipeline = []
    print(f"\nLoading pipeline from {DECOMPOSED_CONFIG}...")
    for cfg in config_data.get("agents", []):
        prompt_path = os.path.join(PROMPTS_DIR, cfg["prompt_file"])
        prompt = open(prompt_path).read() if os.path.exists(prompt_path) else "{input}"
        agent = GenericAgent(
            name=cfg["name"],
            model_name=cfg.get("model", DEFAULT_MODEL),
            prompt_template=prompt,
            context_files=cfg.get("context_files", []),
            temperature=cfg.get("temperature", 0.0),
            num_predict=cfg.get("num_predict", 1024),
            num_ctx=cfg.get("num_ctx", 3072),
        )
        pipeline.append(agent)
    print(f"Pipeline ready: {[a.name for a in pipeline]}\n")
    return pipeline


def run_pipeline(pipeline: list, user_input: str) -> str:
    previous_output = None
    for agent in pipeline:
        output = agent.run(input_text=user_input, previous_agent_output=previous_output)
        previous_output = output

    print("\n" + "=" * 40)
    print(f"  FINAL ANSWER: {(previous_output or '').strip()}")
    print("=" * 40)
    return previous_output


def main():
    parser = argparse.ArgumentParser(description="MARC Ollama Pipeline")
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Run the decomposer to generate agent prompts from a task description",
    )
    args = parser.parse_args()

    print("Welcome to MARC (Ollama)")

    if args.decompose:
        if not run_decomposer():
            return

    pipeline = load_pipeline()
    print("Enter input (or 'exit' to quit):")

    while True:
        user_input = input("\n>>> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        run_pipeline(pipeline, user_input)


if __name__ == "__main__":
    main()
