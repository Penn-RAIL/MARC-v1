import os
import warnings
from dotenv import load_dotenv
from pipeline import MARCPipeline

# Suppress warnings
warnings.filterwarnings("ignore")

load_dotenv("keys.env")
load_dotenv() # Load from .env as well

def main():
    print("Welcome to the MARC (Medical Agentic Report Classification) Framework")
    print("Now with Iterative Feedback Loop for Quality Refinement!")

    # 1. Initialize Pipeline
    print("\nInitializing Pipeline...")
    try:
        pipeline = MARCPipeline(
            config_path="config/agents.yaml",
            prompts_dir="prompts",
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Pipeline initialized with {len(pipeline.pipeline)} agents.")
    if pipeline.evaluator:
        print(f"Evaluator configured: threshold={pipeline.quality_threshold}, max_iterations={pipeline.max_iterations}")
    else:
        print("Feedback loop disabled. Running in single-pass mode.")

    # 2. Input Handling
    print("\nEnter input text for the agent pipeline (or 'exit' to quit):")
    print("To include an image, use format: [IMAGE:/path/to/image.jpg] Your text here")

    while True:
        user_input = input("\n>>> ")
        if user_input.lower() in ['exit', 'quit']:
            break

        if not user_input.strip():
            continue

        # Parse image path if provided
        image_path = None
        text_input = user_input

        # Check for image marker: [IMAGE:path]
        if user_input.startswith("[IMAGE:"):
            end_bracket = user_input.find("]")
            if end_bracket != -1:
                image_path = user_input[7:end_bracket].strip()
                text_input = user_input[end_bracket+1:].strip()

                if not os.path.exists(image_path):
                    print(f"Error: Image file not found: {image_path}")
                    continue
                print(f"Image loaded: {image_path}")

        # 3. Run Pipeline
        result = pipeline.run(input_text=text_input, image_path=image_path)

        # 4. Display Results
        for agent_key, output in result.agent_outputs.items():
            agent_idx = int(agent_key.split("_")[1]) - 1
            agent_name = pipeline.pipeline[agent_idx].name if agent_idx < len(pipeline.pipeline) else agent_key
            print(f"\n--- {agent_name} ---")
            print(f"Output:\n{output}")

        if result.final_score is not None:
            print(f"\nEvaluation Score: {result.final_score}/100")
            if result.quality_passed:
                print(f"Quality threshold met! ({result.final_score} >= {pipeline.quality_threshold})")
            else:
                print(f"Quality threshold not met ({result.final_score} < {pipeline.quality_threshold})")
            print(f"Iterations: {result.iterations}")

        print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()
