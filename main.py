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
    print("Now with Iterative Feedback Loop for Quality Refinement!")
    
    # 1. Configuration Loading
    config_path = "config/agents.yaml"
    prompts_dir = "prompts"
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        agents_config = config_data.get('agents', [])
        evaluator_config = config_data.get('evaluator', {})
        feedback_config = config_data.get('feedback_loop', {})

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
    
    # Initialize Evaluator if enabled
    evaluator = None
    feedback_enabled = feedback_config.get('enabled', False)
    if evaluator_config.get('enabled', False) and feedback_enabled:
        print("Initializing Quality Evaluator...")
        evaluator = GenericAgent(
            name="Quality Evaluator",
            model_name=evaluator_config.get("model", "gemini-2.0-flash"),
            prompt_template=read_prompt(evaluator_config["prompt_file"]),
            context_files=[]
        )
        quality_threshold = evaluator_config.get('quality_threshold', 70)
        max_iterations = evaluator_config.get('max_iterations', 3)
        verbose = feedback_config.get('verbose', True)
        print(f"Evaluator configured: threshold={quality_threshold}, max_iterations={max_iterations}")
    else:
        print("Feedback loop disabled. Running in single-pass mode.")
        quality_threshold = 0
        max_iterations = 1
        verbose = False
        
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

        current_input = text_input
        previous_output = None
        
        # 3. Pipeline Execution with Feedback Loop
        iteration = 0
        quality_passed = False
        agent_outputs = {}
        agent_feedback = {}
        
        while iteration < max_iterations and not quality_passed:
            iteration += 1
            
            if iteration > 1:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration}: Refining outputs based on feedback...")
                print(f"{'='*60}")
            
            # Run all agents in the pipeline
            for i, agent in enumerate(pipeline):
                print(f"\n--- {agent.name} is working... ---")
                
                # Get feedback for this specific agent if available
                agent_specific_feedback = agent_feedback.get(f"agent_{i+1}", None)
                
                # Logic:
                # Agent 1: receives user_input, prev_output=None
                # Agent chain: subsequent agents receive the previous output as primary input 
                # and may also have the original input available in the prompt template.
                # All agents receive the same image if provided
                
                output = agent.run(
                    input_text=current_input, 
                    previous_agent_output=previous_output,
                    image_path=image_path,
                    feedback=agent_specific_feedback
                )
                
                print(f"Output from {agent.name}:\n{output}")
                
                # Store output for evaluation
                agent_outputs[f"agent_{i+1}"] = output
                
                # Update for next step
                previous_output = output
            
            # 4. Quality Evaluation (if evaluator is enabled)
            if evaluator:
                print(f"\n{'='*60}")
                print(f"Quality Evaluator is assessing outputs (Iteration {iteration})...")
                print(f"{'='*60}")
                
                # Prepare evaluation input
                eval_input = f"Iteration: {iteration}"
                
                # Build the evaluation prompt with all outputs
                eval_prompt_data = {
                    "original_input": text_input,
                    "agent_1_output": agent_outputs.get("agent_1", "N/A"),
                    "agent_2_output": agent_outputs.get("agent_2", "N/A"),
                    "agent_3_output": agent_outputs.get("agent_3", "N/A")
                }
                
                # Format the evaluator's prompt template
                formatted_eval_input = evaluator.prompt_template
                for key, value in eval_prompt_data.items():
                    formatted_eval_input = formatted_eval_input.replace(f"{{{key}}}", value)
                
                # Get evaluation
                evaluation_output = evaluator.llm.invoke(formatted_eval_input).content
                
                # Parse evaluation results
                evaluation = GenericAgent.parse_evaluation(evaluation_output)
                
                overall_score = evaluation.get("overall_score", 0)
                quality_passed = overall_score >= quality_threshold
                
                print(f"\nEvaluation Results:")
                print(f"Overall Score: {overall_score}/100")
                
                if verbose:
                    print(f"\nDetailed Scores:")
                    for agent_key, scores in evaluation.get("agent_scores", {}).items():
                        print(f"  {agent_key}: {scores}")
                
                if quality_passed:
                    print(f"✓ Quality threshold met! ({overall_score} >= {quality_threshold})")
                    if verbose:
                        print("\nFinal Outputs:")
                        for agent_key, output in agent_outputs.items():
                            print(f"\n{agent_key}:\n{output}")
                else:
                    print(f"✗ Quality threshold not met ({overall_score} < {quality_threshold})")
                    
                    if iteration < max_iterations:
                        print(f"\nFeedback for refinement:")
                        agent_feedback = evaluation.get("feedback", {})
                        for agent_key, feedback_text in agent_feedback.items():
                            if feedback_text and feedback_text != "No issues detected":
                                print(f"  {agent_key}: {feedback_text}")
                    else:
                        print(f"\nMaximum iterations ({max_iterations}) reached.")
                        print("Proceeding with current outputs despite not meeting threshold.")
            else:
                # No evaluator - single pass complete
                quality_passed = True
            
        print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()