import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from agents.tagger import tag_report
from agents.classifier import classify_report
from agents.recommender import recommend_follow_up
from agents.orchestrator import evaluate_report
load_dotenv()

def create_llm():
    """Create and return the LLM instance (called once)."""
    load_dotenv("keys.env")
    api_key = os.getenv("GOOGLE_API_KEY") or HARDCODED_GOOGLE_API_KEY
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Provide it via .env, shell export, or HARDCODED_GOOGLE_API_KEY."
        )

    # Ensure the client can read the key even if it came from the hardcoded fallback.
    os.environ["GOOGLE_API_KEY"] = api_key

    model_name = os.getenv("GEMINI_MODEL") or HARDCODED_MEDGEMMA_MODEL or "medgemma"
    llm = ChatGoogleGenerativeAI(model=model_name)
    return llm

def process_report(masked_text: str, original_text: str, llm, prompts: dict, index: int, max_iterations: int = 3) -> dict:
    """Process a single report through all 4 agents with iterative refinement based on feedback."""
    print(f"\n{'='*60}")
    print(f"Processing Report {index + 1}")
    print(f"{'='*60}")
    
    iteration = 0
    feedback = None
    
    while iteration < max_iterations:
        iteration_label = f"Iteration {iteration + 1}" if iteration > 0 else "Initial Attempt"
        print(f"\n>>> {iteration_label} <<<\n")
        
        # Agent 1 processing
        print("--- Running Agent 1: Tagger ---")
        tags = tag_report(masked_text, llm, prompt_template=prompts['tagger'], feedback=feedback)
        print(tags)
        print("--- Agent 1: Tagger Complete ---\n")

        if not tags.needs_downstream_disease_analysis:
            print("Skipping Agent 2 and 3 as no downstream disease analysis is needed.")
            return {
                'report_index': index,
                'tags': tags.model_dump(),
                'classification': None,
                'recommendations': None,
                'evaluation': None,
                'iterations': iteration + 1,
                'skipped': True
            }

        # Agent 2 processing
        print("--- Running Agent 2: Classifier ---")
        classification = classify_report(masked_text, tags.model_dump(), llm, prompt_template=prompts['classifier'], feedback=feedback)
        print(classification)
        print("--- Agent 2: Classifier Complete ---\n")

        # Agent 3 processing
        print("--- Running Agent 3: Recommender ---")
        recommendations = recommend_follow_up(
            disease_type=classification.disease_type,
            impression=classification.impression,
            reason=classification.reasoning,
            llm=llm,
            prompt_template=prompts['recommender'],
            feedback=feedback
        )
        print(recommendations)
        print("--- Agent 3: Recommender Complete ---\n")

        # Agent 4 processing
        print("--- Running Agent 4: Final Evaluator ---")
        final_evaluation = evaluate_report(
            actual_report_text=original_text,
            tags=tags.model_dump(),
            classification=classification.model_dump(),
            recommendations=recommendations.model_dump(),
            llm=llm,
            prompt_template=prompts['evaluator'],
        )
        print(final_evaluation)
        print("--- Agent 4: Final Evaluator Complete ---")
        
        # Display iteration scores
        print(f"\n📊 {iteration_label} Scores:")
        print(f"   Classification: {final_evaluation.classification_score:.2f}")
        print(f"   Impression: {final_evaluation.impression_score:.2f}")
        print(f"   Overall: {final_evaluation.overall_report_score:.2f}")
        
        # Check if scores meet quality threshold
        quality_threshold = 0.7
        if (final_evaluation.overall_report_score >= quality_threshold and 
            final_evaluation.classification_score >= quality_threshold):
            print(f"\n✅ Quality threshold met! Completed in {iteration + 1} iteration(s).")
            return {
                'report_index': index,
                'tags': tags.model_dump(),
                'classification': classification.model_dump(),
                'recommendations': recommendations.model_dump(),
                'evaluation': final_evaluation.model_dump(),
                'iterations': iteration + 1,
                'skipped': False
            }
        
        # If we haven't reached max iterations, prepare feedback for next iteration
        if iteration + 1 < max_iterations:
            print(f"\n⚠️  Quality threshold not met. Preparing feedback for next iteration...")
            
            # Generate detailed feedback for the next iteration
            feedback_parts = []
            
            if final_evaluation.classification_score < quality_threshold:
                feedback_parts.append(f"Classification Score ({final_evaluation.classification_score:.2f}) is below threshold.")
            
            if final_evaluation.impression_score < quality_threshold:
                feedback_parts.append(f"Impression Score ({final_evaluation.impression_score:.2f}) is below threshold.")
                feedback_parts.append(f"Impression Reasoning: {final_evaluation.impression_reasoning}")
            
            if final_evaluation.additional_notes:
                feedback_parts.append(f"Safety/Alignment Issues: {final_evaluation.additional_notes}")
            
            feedback_parts.append(f"Follow-up Reasoning: {final_evaluation.follow_up_reasoning}")
            
            feedback = "\n".join(feedback_parts)
            print(f"\nFeedback generated:\n{feedback}")
        
        iteration += 1
    
    # Return best attempt after max iterations
    print(f"\n⚠️  Max iterations ({max_iterations}) reached. Returning best attempt.")
    return {
        'report_index': index,
        'tags': tags.model_dump(),
        'classification': classification.model_dump(),
        'recommendations': recommendations.model_dump(),
        'evaluation': final_evaluation.model_dump(),
        'iterations': max_iterations,
        'warning': 'Max iterations reached without meeting quality threshold',
        'skipped': False
    }

# Load CSV data
masked_report = pd.read_csv("data/postive_chest_ct_synthetic_radiology_reports_masked.csv")
unmasked_report = pd.read_csv("data/postive_chest_ct_synthetic_radiology_reports.csv")

# Initialize LLM
llm = create_llm()

# Load all prompts once
prompts = {
    'tagger': open("prompts/tagger_prompt.txt").read(),
    'classifier': open("prompts/classifier_prompt.txt").read(),
    'recommender': open("prompts/recommender_prompt.txt").read(),
    'evaluator': open("prompts/final_evaluator_prompt.txt").read()
}

# Process first 100 reports
results = []
num_reports = min(100, len(masked_report))

for i in range(num_reports):
    masked_text = masked_report.iloc[i]["Report"]
    original_text = unmasked_report.iloc[i]["Report"]
    
    result = process_report(masked_text, original_text, llm, prompts, i)
    results.append(result)

# Create results DataFrame
results_data = []
for r in results:
    if r['skipped']:
        continue
    
    eval_data = r['evaluation']
    results_data.append({
        'report_index': r['report_index'],
        'iterations': r['iterations'],
        'classification_score': eval_data['classification_score'],
        'impression_score': eval_data['impression_score'],
        'overall_report_score': eval_data['overall_report_score'],
        'follow_up_recommended': eval_data['follow_up_recommended'],
        'impression_reasoning': eval_data['impression_reasoning'],
        'follow_up_reasoning': eval_data['follow_up_reasoning'],
        'additional_notes': eval_data.get('additional_notes', ''),
        'warning': r.get('warning', ''),
        'disease_type': r['classification']['disease_type'],
        'impression': r['classification']['impression'],
        'recommendations': str(r['recommendations']['recommendations'])
    })

results_df = pd.DataFrame(results_data)
results_df.to_csv("logs/evaluation_results.csv", index=False)

# Calculate summary statistics
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Total reports processed: {len(results_data)}/{num_reports}")

if len(results_data) > 0:
    correct_count = (results_df['overall_report_score'] > 0.9).sum()
    acceptable_count = (results_df['overall_report_score'] >= 0.8).sum()
    perfect_count = (results_df['overall_report_score'] == 1.0).sum()
    
    print(f"\nOverall Report Score:")
    print(f"  Acceptable (>= 0.8): {acceptable_count}/{len(results_data)}")
    print(f"  Correct (> 0.9): {correct_count}/{len(results_data)}")
    print(f"  Perfect (= 1.0): {perfect_count}/{len(results_data)}")
    
    print(f"\nMean Scores:")
    print(f"  Classification Score: {results_df['classification_score'].mean():.3f}")
    print(f"  Impression Score: {results_df['impression_score'].mean():.3f}")
    print(f"  Overall Report Score: {results_df['overall_report_score'].mean():.3f}")
    
    print(f"\nIteration Statistics:")
    print(f"  Mean iterations: {results_df['iterations'].mean():.2f}")
    print(f"  Reports requiring refinement: {(results_df['iterations'] > 1).sum()}/{len(results_data)}")
    print(f"  Max iterations reached: {(results_df['warning'] != '').sum()}/{len(results_data)}")
    
    print(f"\nResults saved to: logs/evaluation_results.csv")
else:
    print("No reports were fully processed (all skipped)")