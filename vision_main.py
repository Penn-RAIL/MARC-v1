import os
import yaml
import asyncio
import warnings
from dotenv import load_dotenv
from agents.generic_vision_agent import GenericVisionAgent
from utils.image_handler import ImageHandler

warnings.filterwarnings("ignore")
load_dotenv()


async def main():
    print("*"*80)
    print("Multi-API Vision Analysis Framework")
    print("*"*80)
    
    # Load configuration
    config_path = "prompts/vision_apis.yaml"
    prompts_dir = "prompts"
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    apis_config = config_data.get('vision_apis', [])
    prompt_file = config_data.get('prompt_file', 'vision_prompt.txt')
    settings = config_data.get('settings', {})
    
    # Load shared prompt
    prompt_path = os.path.join(prompts_dir, prompt_file)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
    else:
        print(f"Warning: {prompt_path} not found. Using default prompt.")
        prompt_template = "Analyze this image in detail."
    
    # Initialize agents from config
    agents = []
    print("\nInitializing APIs from config...")
    
    for config in apis_config:
        if not config.get("enabled", True):
            print(f"  ⊗ {config['name']} - Disabled")
            continue
        
        try:
            agent = GenericVisionAgent(
                name=config["name"],
                api_key_env=config["api_key_env"],
                api_type=config["api_type"],
                endpoint=config["endpoint"],
                model=config["model"],
                prompt_template=prompt_template,
                max_tokens=settings.get('max_tokens', 1500),
                timeout=settings.get('timeout', 30)
            )
            agents.append(agent)
            print(f"  ✓ {config['name']} ({config['model']})")
        except ValueError as e:
            print(f"  ✗ {config['name']} - {e}")
        except Exception as e:
            print(f"  ✗ {config['name']} - Error: {e}")
    
    if not agents:
        print("\nError: No APIs successfully initialized.")
        print("Check that API keys are set in your .env file")
        return
    
    print(f"\nInitialized {len(agents)} API(s).")
    
    # Interactive input loop
    print("\n" + "="*80)
    print("Enter image file path to analyze (or 'exit' to quit)")
    print("="*80)
    
    while True:
        image_path = input("\n>>> ")
        
        if image_path.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        if not image_path.strip():
            continue

        try:
            # Load image
            print(f"\nLoading: {image_path}")
            image_handler = ImageHandler()
            image_data = image_handler.load_image(image_path)
            
            # Show image info
            info = image_handler.get_image_info(image_data)
            print(f"Image: {info['width']}x{info['height']} {info['format']}, "
                  f"{info['file_size_bytes']:,} bytes")
            
            # Resize if needed
            image_data = image_handler.resize_if_needed(image_data)
            
            print(f"\n{'='*80}")
            print(f"Analyzing with {len(agents)} API(s)...")
            print(f"{'='*80}\n")
            
            # Run all APIs in parallel
            tasks = [agent.run(image_data) for agent in agents]
            results = await asyncio.gather(*tasks)
            
            # Display results
            for result in results:
                print(f"\n{'─'*80}")
                
                if result.get('error'):
                    print(f"❌ {result['provider']} ({result['model']})")
                    print(f"Error: {result['error']}")
                else:
                    print(f"✓ {result['provider']} ({result['model']})")
                    print(f"Tokens: {result['tokens']:,} | Latency: {result['latency']:.2f}s")
                    print(f"\nResponse:")
                    print(result['response'])
            
            print(f"\n{'='*80}")
            print("Analysis complete.")
            print(f"{'='*80}")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
        except ValueError as e:
            print(f"\n❌ Error: {e}")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())