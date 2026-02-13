"""
MARC Pipeline Module

Reusable pipeline orchestrator for the MARC (Medical Agentic Report Classification)
multi-agent system. Extracted from main.py to allow programmatic use by the
MedHELM-compatible API server and HELM client.
"""

import os
import time
import yaml
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from dotenv import load_dotenv
from agents.agent import GenericAgent

logger = logging.getLogger("marc.pipeline")


@dataclass
class PipelineResult:
    """Structured result from a full pipeline execution."""
    final_output: str
    agent_outputs: Dict[str, str]
    iterations: int
    final_score: Optional[float]
    quality_passed: bool
    request_time: float


class MARCPipeline:
    """
    Reusable MARC multi-agent pipeline.

    Runs Agent 1 (Tagger) -> Agent 2 (Classifier) -> Agent 3 (Recommender)
    with an optional evaluator feedback loop for quality refinement.
    """

    def __init__(
        self,
        config_path: str = "config/agents.yaml",
        prompts_dir: str = "prompts",
        base_dir: Optional[str] = None,
    ):
        """
        Initialize the pipeline from configuration.

        Args:
            config_path: Path to agents.yaml (relative to base_dir).
            prompts_dir: Path to prompts directory (relative to base_dir).
            base_dir: Base directory for resolving relative paths.
                      Defaults to the directory containing this file.
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir

        # Load environment variables
        load_dotenv(os.path.join(base_dir, "keys.env"))
        load_dotenv(os.path.join(base_dir, ".env"))
        load_dotenv()

        # Resolve paths
        abs_config_path = os.path.join(base_dir, config_path)
        self.prompts_dir = os.path.join(base_dir, prompts_dir)

        if not os.path.exists(abs_config_path):
            raise FileNotFoundError(f"Configuration file not found: {abs_config_path}")

        with open(abs_config_path, "r") as f:
            config_data = yaml.safe_load(f)

        agents_config = config_data.get("agents", [])
        evaluator_config = config_data.get("evaluator", {})
        feedback_config = config_data.get("feedback_loop", {})

        # Build agent pipeline
        self.pipeline: List[GenericAgent] = []
        for config in agents_config:
            # Resolve context_files relative to base_dir
            context_files = [
                os.path.join(base_dir, cf) if not os.path.isabs(cf) else cf
                for cf in config.get("context_files", [])
            ]
            agent = GenericAgent(
                name=config["name"],
                model_name=config.get("model", "gemini-1.5-flash"),
                prompt_template=self._read_prompt(config["prompt_file"]),
                context_files=context_files,
                backend=config.get("backend", "api"),
            )
            self.pipeline.append(agent)
        logger.info("Pipeline initialized with %d agents", len(self.pipeline))

        # Build evaluator
        self.evaluator: Optional[GenericAgent] = None
        self.quality_threshold = 0
        self.max_iterations = 1
        self.verbose = False

        feedback_enabled = feedback_config.get("enabled", False)
        if evaluator_config.get("enabled", False) and feedback_enabled:
            self.evaluator = GenericAgent(
                name="Quality Evaluator",
                model_name=evaluator_config.get("model", "gemini-2.0-flash"),
                prompt_template=self._read_prompt(evaluator_config["prompt_file"]),
                context_files=[],
                backend=evaluator_config.get("backend", "api"),
            )
            self.quality_threshold = evaluator_config.get("quality_threshold", 70)
            self.max_iterations = evaluator_config.get("max_iterations", 3)
            self.verbose = feedback_config.get("verbose", True)
            logger.info(
                "Evaluator enabled: threshold=%d, max_iterations=%d",
                self.quality_threshold,
                self.max_iterations,
            )
        else:
            logger.info("Feedback loop disabled. Running in single-pass mode.")

    def _read_prompt(self, name: str) -> str:
        """Read a prompt file from the prompts directory."""
        path = os.path.join(self.prompts_dir, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        logger.warning("Prompt file %s not found. Using default prompt.", path)
        return "You are a helpful assistant. Input: {input}"

    def run(
        self,
        input_text: str,
        image_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline on a single input.

        Args:
            input_text: The text prompt to process through all agents.
            image_path: Optional path to an image file for multimodal tasks.

        Returns:
            PipelineResult with all outputs and metadata.
        """
        start_time = time.time()

        current_input = input_text
        previous_output = None

        iteration = 0
        quality_passed = False
        agent_outputs: Dict[str, str] = {}
        agent_feedback: Dict[str, str] = {}
        overall_score: Optional[float] = None

        while iteration < self.max_iterations and not quality_passed:
            iteration += 1

            if iteration > 1:
                logger.info("Iteration %d: Refining outputs based on feedback...", iteration)

            # Run all agents in the pipeline
            for i, agent in enumerate(self.pipeline):
                logger.debug("Running %s...", agent.name)

                agent_specific_feedback = agent_feedback.get(f"agent_{i+1}", None)

                output = agent.run(
                    input_text=current_input,
                    previous_agent_output=previous_output,
                    image_path=image_path,
                    feedback=agent_specific_feedback,
                )

                agent_outputs[f"agent_{i+1}"] = output
                previous_output = output
                logger.debug("%s output: %s", agent.name, output[:200])

            # Quality evaluation
            if self.evaluator:
                logger.info("Evaluator assessing outputs (iteration %d)...", iteration)

                eval_prompt_data = {
                    "original_input": input_text,
                    "agent_1_output": agent_outputs.get("agent_1", "N/A"),
                    "agent_2_output": agent_outputs.get("agent_2", "N/A"),
                    "agent_3_output": agent_outputs.get("agent_3", "N/A"),
                }

                formatted_eval_input = self.evaluator.prompt_template
                for key, value in eval_prompt_data.items():
                    formatted_eval_input = formatted_eval_input.replace(
                        f"{{{key}}}", value
                    )

                if self.evaluator.backend == "local":
                    evaluation_output = self.evaluator.local_model.generate(
                        formatted_eval_input
                    )
                else:
                    evaluation_output = self.evaluator.llm.invoke(
                        formatted_eval_input
                    ).content
                evaluation = GenericAgent.parse_evaluation(evaluation_output)

                overall_score = evaluation.get("overall_score", 0)
                quality_passed = overall_score >= self.quality_threshold

                logger.info("Evaluation score: %s/100 (threshold: %d)", overall_score, self.quality_threshold)

                if self.verbose:
                    for agent_key, scores in evaluation.get("agent_scores", {}).items():
                        logger.debug("  %s: %s", agent_key, scores)

                if quality_passed:
                    logger.info("Quality threshold met.")
                else:
                    logger.info("Quality threshold not met.")
                    if iteration < self.max_iterations:
                        agent_feedback = evaluation.get("feedback", {})
                    else:
                        logger.warning(
                            "Max iterations (%d) reached. Proceeding with current outputs.",
                            self.max_iterations,
                        )
            else:
                quality_passed = True

        elapsed = time.time() - start_time

        # Final output is from the last agent in the pipeline
        final_output = agent_outputs.get(f"agent_{len(self.pipeline)}", "")

        return PipelineResult(
            final_output=final_output,
            agent_outputs=agent_outputs,
            iterations=iteration,
            final_score=overall_score,
            quality_passed=quality_passed,
            request_time=elapsed,
        )
