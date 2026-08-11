import argparse
import sys

from dotenv import find_dotenv, load_dotenv

from .config import (
    DEFAULT_CONFIG_PATH,
    DECOMPOSED_CONFIG_PATH,
    check_environment,
    load_pipeline,
    write_decomposed_pipeline,
)
from .exceptions import ConfigError
from .pipeline import run_pipeline


def _run_decomposer() -> bool:
    from .decomposer import DecomposerAgent

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

    decomp = decomposer.get_decomposition()
    print("\n" + "=" * 60)
    print(f"  Task: {decomp.get('task_summary', '')}")
    print("=" * 60)
    print("Writing agent prompts...")
    write_decomposed_pipeline(decomp)
    print("\nAgent prompts written.")
    return True


def _interactive_loop(pipeline) -> None:
    print("Enter input (or 'exit' to quit):")
    while True:
        user_input = input("\n>>> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        output = run_pipeline(pipeline, user_input)
        print("\n" + "=" * 40)
        print(f"  FINAL ANSWER: {(output or '').strip()}")
        print("=" * 40)


def cmd_run(args: argparse.Namespace) -> None:
    check_environment()
    config_path = args.config or DEFAULT_CONFIG_PATH
    print(f"Welcome to MARC — loading pipeline from {config_path}")
    pipeline = load_pipeline(config_path)
    print(f"Pipeline ready: {[a.name for a in pipeline]}")
    _interactive_loop(pipeline)


def cmd_decompose(args: argparse.Namespace) -> None:
    if _run_decomposer():
        print(
            f"\nRun `marc run --config {DECOMPOSED_CONFIG_PATH}` "
            "to use the generated pipeline."
        )


def main() -> None:
    # usecwd=True: search from the directory the command is actually run
    # from, not from this installed package's source location (the default
    # search walks up from the calling file, which for an installed/editable
    # package can find an unrelated .env in some parent directory).
    load_dotenv(find_dotenv("keys.env", usecwd=True))
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(prog="marc", description="MARC multi-agent pipeline")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the agent pipeline interactively")
    run_parser.add_argument(
        "--config",
        default=None,
        help=f"Path to pipeline config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    run_parser.set_defaults(func=cmd_run)

    decompose_parser = subparsers.add_parser(
        "decompose", help="Generate a new 3-agent pipeline from a task description"
    )
    decompose_parser.set_defaults(func=cmd_decompose)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return

    try:
        args.func(args)
    except ConfigError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
