import os
import yaml
import warnings
from dotenv import load_dotenv
from agents.agent import GenericAgent

# Suppress warnings
warnings.filterwarnings("ignore")

load_dotenv("keys.env")
load_dotenv() # Load from .env as well

def main():
    print("Welcome to the MARC (Medical Agentic Report Classification) Framework")
    
    # 1. Configuration Loading
    config_path = "config/agents.yaml"
    prompts_dir = "prompts"
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        agents_config = config_data.get('agents', [])

    # helper to safely read prompt
    def read_prompt(name):
        path = os.path.join(prompts_dir, name)
        if os.path.exists(path):
            return open(path).read()
        print(f"Warning: Prompt file {path} not found. Using default prompt.")
        return "You are a helpful assistant. Input: {input}"

    # Instantiate Agents
    pipeline = []
    print("\nInitializing Pipeline...")
    for config in agents_config:
        agent = GenericAgent(
            name=config["name"],
            model_name=config.get("model", "gemini-1.5-flash"),
            prompt_template=read_prompt(config["prompt_file"]),
            context_files=config.get("context_files", [])
        )
        pipeline.append(agent)
    print("Pipeline initialized successfully.")
        
    # 2. Input Handling
    print("\nEnter input text for the agent pipeline (or 'exit' to quit):")
    
    while True:
        user_input = input("\n>>> ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input.strip():
            continue

        current_input = user_input
        previous_output = None
        
        # 3. Pipeline Execution
        for i, agent in enumerate(pipeline):
            print(f"\n--- {agent.name} is working... ---")
            
            # Logic:
            # Agent 1: receives user_input, prev_output=None
            # Agent chain: subsequent agents receive the previous output as primary input 
            # and may also have the original input available in the prompt template.
            
            output = agent.run(input_text=current_input, previous_agent_output=previous_output)
            
            print(f"Output from {agent.name}:\n{output}")
            
            # Update for next step
            previous_output = output
            
        print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()