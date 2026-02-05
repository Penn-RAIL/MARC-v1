# Feedback Loop Implementation

## Overview

The MARC framework now includes an **iterative feedback loop** that automatically evaluates and refines agent outputs until quality thresholds are met.

## How It Works

### 1. Pipeline Execution
- All 3 agents run sequentially, producing their outputs
- Outputs are stored for evaluation

### 2. Quality Evaluation
- A specialized **Evaluator Agent** assesses all outputs
- Scores each agent on 4 criteria (0-10 each):
  - **Accuracy**: Correctness and relevance
  - **Completeness**: Coverage of all necessary aspects
  - **Clarity**: Structure and understandability
  - **Consistency**: Alignment with other agents' outputs
- Calculates an **overall score** (0-100)

### 3. Decision Point
- **Score >= Threshold**: Pipeline complete, outputs accepted
- **Score < Threshold**: Generate feedback and iterate

### 4. Refinement Iteration
- Evaluator provides specific feedback for each underperforming agent
- Feedback is injected into agent prompts
- Agents re-run with awareness of what needs improvement
- Process repeats until quality passes or max iterations reached

## Configuration

### In `config/agents.yaml`:

```yaml
# Quality Evaluator Configuration
evaluator:
  enabled: true                # Enable/disable feedback loop
  model: "gemini-2.0-flash"    # Model for evaluation
  prompt_file: "evaluator_prompt.txt"
  quality_threshold: 70        # Minimum score (0-100) to pass
  max_iterations: 3            # Maximum refinement attempts

# Feedback Loop Settings
feedback_loop:
  enabled: true                # Master switch for feedback
  verbose: true                # Print detailed scores
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluator.enabled` | boolean | `false` | Enable evaluator agent |
| `evaluator.model` | string | `gemini-2.0-flash` | LLM for evaluation |
| `evaluator.quality_threshold` | integer | `70` | Minimum passing score (0-100) |
| `evaluator.max_iterations` | integer | `3` | Max refinement loops |
| `feedback_loop.enabled` | boolean | `false` | Enable iteration logic |
| `feedback_loop.verbose` | boolean | `true` | Show detailed scores |

## Example Output

```
============================================================
ITERATION 1: Initial Run
============================================================

--- Agent 1 is working... ---
Output from Agent 1:
{ "entities": ["MRI", "Brain"], "summary": "Medical imaging analysis" }

--- Agent 2 is working... ---
Output from Agent 2:
Category: Neurology. Reason: Brain scan analysis required.

--- Agent 3 is working... ---
Output from Agent 3:
Recommendation: Consult neurologist for detailed review.

============================================================
Quality Evaluator is assessing outputs (Iteration 1)...
============================================================

Evaluation Results:
Overall Score: 65/100

Detailed Scores:
  agent_1: {'accuracy': 7, 'completeness': 6, 'clarity': 7, 'consistency': 7}
  agent_2: {'accuracy': 6, 'completeness': 5, 'clarity': 7, 'consistency': 6}
  agent_3: {'accuracy': 7, 'completeness': 6, 'clarity': 6, 'consistency': 6}

✗ Quality threshold not met (65 < 70)

Feedback for refinement:
  agent_1: Include specific anatomical structures identified
  agent_2: Provide more detailed clinical reasoning
  agent_3: Suggest specific next diagnostic steps

============================================================
ITERATION 2: Refining outputs based on feedback...
============================================================

[Agents re-run with feedback incorporated...]

============================================================
Quality Evaluator is assessing outputs (Iteration 2)...
============================================================

Evaluation Results:
Overall Score: 85/100

✓ Quality threshold met! (85 >= 70)
```

## Benefits

### 1. **Self-Correcting**
- Automatically identifies weaknesses
- No manual review needed for basic quality issues

### 2. **Consistent Quality**
- Enforces minimum standards
- Reduces variance in outputs

### 3. **Transparent**
- Detailed scoring shows exactly what was evaluated
- Feedback is visible and actionable

### 4. **Configurable**
- Adjust thresholds based on use case
- Control iteration limits to manage costs

### 5. **Cost-Aware**
- Limits maximum iterations
- Only refines when necessary
- Can disable for development/testing

## Disabling Feedback Loop

To run in single-pass mode (no evaluation):

```yaml
evaluator:
  enabled: false

feedback_loop:
  enabled: false
```

Or simply set both to `false` and the pipeline runs like the original V1.

## Best Practices

### Threshold Setting
- **High-stakes tasks** (medical, legal): 80-90
- **General classification**: 70-80
- **Exploratory analysis**: 60-70
- **Development/testing**: 50 or disable

### Max Iterations
- **3 iterations**: Good balance of quality vs cost
- **1 iteration**: Single-pass, no refinement
- **5+ iterations**: Very high-quality requirements

### Verbose Mode
- **Development**: `true` - see all scores
- **Production**: `false` - minimal output

## Customizing the Evaluator

Edit `prompts/evaluator_prompt.txt` to:
- Change scoring criteria
- Add domain-specific checks
- Modify output format
- Adjust feedback style

## Technical Details

### Feedback Injection
Feedback is appended to agent prompts:
```
=== FEEDBACK FOR IMPROVEMENT ===
[Specific feedback from evaluator]
=== Please address the above feedback in your response ===
```

### JSON Parsing
Evaluator outputs structured JSON:
```json
{
  "overall_score": 75,
  "agent_scores": {
    "agent_1": {"accuracy": 8, "completeness": 7, ...},
    ...
  },
  "feedback": {
    "agent_1": "Specific improvement needed",
    ...
  },
  "pass": true
}
```

### Error Handling
- Failed JSON parsing returns default low scores
- Triggers refinement automatically
- Graceful degradation on evaluation errors

## Future Enhancements

Potential additions to the feedback loop:
- [ ] Multi-criteria weighting (some criteria more important)
- [ ] Progressive thresholds (higher bar each iteration)
- [ ] Agent-specific thresholds
- [ ] External validation hooks
- [ ] Feedback history tracking
- [ ] Learning from successful refinements

## Troubleshooting

**Issue**: Evaluator always fails to parse JSON  
**Solution**: Check that evaluator prompt produces valid JSON, or adjust regex in `parse_evaluation()`

**Issue**: Never reaches threshold  
**Solution**: Lower threshold, increase max_iterations, or check if agents can see feedback

**Issue**: Too many API calls  
**Solution**: Reduce max_iterations or disable feedback loop for testing

**Issue**: Scores seem random  
**Solution**: Improve evaluator prompt with clearer scoring rubric
