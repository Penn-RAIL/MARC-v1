"""
Baseline Evaluation Script for Gemini Flash 2.0
Evaluates the base model on all image questions from the CSV test set.
"""

import os
import csv
import base64
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv("../keys.env")
load_dotenv("../.env")
load_dotenv()  # Try current directory as well

class BaselineEvaluator:
    def __init__(self, model_name="gemini-2.0-flash", api_key=None):
        """Initialize the evaluator with a specific model."""
        self.model_name = model_name
        
        # Try to get API key from parameter, then environment
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("\nError: GOOGLE_API_KEY not found!")
            print("Please set it in one of these ways:")
            print("1. Create a .env file in the project root with: GOOGLE_API_KEY=your_key_here")
            print("2. Set it as an environment variable: export GOOGLE_API_KEY=your_key_here")
            print("3. Pass it directly when running the script\n")
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key
        )
        print(f"Initialized evaluator with model: {model_name}")
    
    def encode_image(self, image_path):
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine MIME type from file extension
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(suffix, 'image/jpeg')
        
        return {
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{encoded_string}"
        }
    
    def find_image_file(self, image_id, images_dir):
        """Find the image file for a given image ID."""
        # Try different extensions
        extensions = ['.jpg', '.jpeg', '.png', '.gif']
        for ext in extensions:
            image_path = os.path.join(images_dir, f"{image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        return None
    
    def query_model(self, question, image_path, max_retries=3):
        """Query the model with a question and image, with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                # Create multimodal message
                message_content = [
                    {"type": "text", "text": question}
                ]
                
                # Add image
                image_data = self.encode_image(image_path)
                message_content.append(image_data)
                
                # Get response
                response = self.llm.invoke([HumanMessage(content=message_content)])
                return response.content
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit or quota error
                if "quota" in error_msg or "rate" in error_msg or "resource" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10  # 10, 20, 30 seconds
                        print(f"  ⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f"ERROR: Rate limit exceeded - {str(e)}"
                else:
                    return f"ERROR: {str(e)}"
        
        return "ERROR: Max retries exceeded"
    
    def _save_results(self, results, output_path):
        """Save results to CSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Image Number', 'Modality', 'Organ System', 'Question', 
                         'Ground Truth', 'Model Response', 'Model']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    def evaluate(self, csv_path, images_dir, output_path=None, delay_between_requests=2.0, save_interval=10):
        """
        Evaluate all questions in the CSV file.
        
        Args:
            csv_path: Path to the CSV file with test questions
            images_dir: Directory containing the test images
            output_path: Path to save results (optional)
            delay_between_requests: Delay in seconds between API calls (default 2.0)
            save_interval: Save results every N questions (default 10)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"baseline_results_{timestamp}.csv"
        
        results = []
        
        print(f"\nReading test cases from: {csv_path}")
        print(f"Images directory: {images_dir}")
        print(f"Delay between requests: {delay_between_requests}s")
        print(f"Auto-saving every {save_interval} questions")
        
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_cases = list(reader)
        
        total_cases = len(test_cases)
        print(f"Found {total_cases} test cases\n")
        
        # Process each test case
        for idx, row in enumerate(test_cases, 1):
            image_id = row['Image Number']
            question = row['Question']
            ground_truth = row['Ground Truth']
            modality = row['Modality']
            organ_system = row['Organ System ']  # Note the space in column name
            
            print(f"[{idx}/{total_cases}] Processing {image_id}: {question[:50]}...")
            
            # Find image file
            image_path = self.find_image_file(image_id, images_dir)
            
            if not image_path:
                print(f"  ⚠️  Image not found for {image_id}")
                model_response = "ERROR: Image file not found"
            else:
                # Query the model
                model_response = self.query_model(question, image_path)
                print(f"  ✓ Response: {model_response[:100]}...")
                
                # Add delay to avoid rate limits
                if idx < total_cases:  # Don't delay after the last one
                    time.sleep(delay_between_requests)
            
            # Store result
            results.append({
                'Image Number': image_id,
                'Modality': modality,
                'Organ System': organ_system,
                'Question': question,
                'Ground Truth': ground_truth,
                'Model Response': model_response,
                'Model': self.model_name
            })
            
            # Auto-save progress every N questions
            if idx % save_interval == 0 or idx == total_cases:
                self._save_results(results, output_path)
                print(f"  💾 Progress saved ({idx}/{total_cases} completed)")
        
        # Final save
        print(f"\n{'='*60}")
        print(f"✓ Evaluation complete! Final results saved.")
        print(f"Total test cases: {total_cases}")
        print(f"{'='*60}\n")
        
        return results


def main():
    """Main execution function."""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Image + Prompt Tracker - Sheet1.csv")
    images_dir = os.path.join(script_dir, "Images")
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(script_dir, f"baseline_results_{timestamp}.csv")
    
    print("="*60)
    print("Baseline Model Evaluation - Gemini Flash 2.0")
    print("="*60)
    
    # Initialize evaluator
    evaluator = BaselineEvaluator(model_name="gemini-2.0-flash")
    
    # Run evaluation with 2 second delay between requests
    results = evaluator.evaluate(
        csv_path, 
        images_dir, 
        output_path,
        delay_between_requests=2.0,  # 2 seconds between API calls
        save_interval=10  # Save every 10 questions
    )
    
    print("\nEvaluation Summary:")
    print(f"- Test cases processed: {len(results)}")
    print(f"- Results file: {output_path}")
    
    # Count errors
    errors = sum(1 for r in results if r['Model Response'].startswith('ERROR'))
    if errors > 0:
        print(f"- Errors encountered: {errors}")
    else:
        print(f"- All queries completed successfully ✓")


if __name__ == "__main__":
    main()
