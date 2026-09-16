import asyncio

from llm_judge import LLMJudge
from continuous_metrics import ContinuousMetricsEvaluator


class PromptSmellDetector:
    """
    Detect the presence of the following six “prompt smells” in English-Language prompts:
    01. Reasoning Suppression
    02. Lack of Self-Reflection
    03. Role Suppression
    04. Unspecified Output Structure
    05. Lack of Examples
    06. Complexity-Length
    07. Grammatical Correctness
    08. Formatting
    09. Readability
    10. Prompt Quality
    """

    def __init__(self):
        self.llm_judge = LLMJudge()
        self.metrics_evaluator = ContinuousMetricsEvaluator()

    async def analyze_prompt(self, prompt: str, judge: bool = True, continuous_metrics: bool = True) -> dict:
        """
        Analyze an individual prompt and return metrics and the presence of smells.
        """

        metrics = {
            "reasoning_score": None,
            "self_reflection_present": None,
            "role_assigned": None,
            "structure_specified": None,
            "examples_count": None,
            "complexity_length_score": None,
            "grammatical_correctness_score": None,
            "readability_score": None,
            "formatting_score": None,
            "prompt_quality_score": None
        }

        if judge:
            llm_results = await self.llm_judge.evaluate(prompt)

            # Map the Pydantic boolean outputs to numeric values (1.0 / 0.0)
            metrics["reasoning_score"] = 1.0 if llm_results.get("is_reasoning_required") else 0.0
            metrics["self_reflection_present"] = 1 if llm_results.get("self_reflection_present") else 0
            metrics["role_assigned"] = 1 if llm_results.get("is_role_assigned") else 0
            metrics["structure_specified"] = 1 if llm_results.get("structure_specified") else 0
            metrics["examples_count"] = llm_results.get("examples_count", 0)

        if continuous_metrics:
            cont_results = self.metrics_evaluator.evaluate(prompt)

            metrics["complexity_length_score"] = cont_results.get("complexity_length_score", 1.0)
            metrics["grammatical_correctness_score"] = cont_results.get("grammatical_correctness_score", 1.0)
            metrics["readability_score"] = cont_results.get("readability_score", 1.0)
            metrics["formatting_score"] = cont_results.get("formatting_score", 1.0)

            # Calculate Prompt Quality Score as the average of the 4 continuous metrics.
            # Using (1.0 - complexity) to align it positively with the other metrics.
            cls = metrics["complexity_length_score"]
            g_score = metrics["grammatical_correctness_score"]
            c_score = metrics["readability_score"]
            f_score = metrics["formatting_score"]

            metrics["prompt_quality_score"] = round(((1.0 - cls) + g_score + c_score + f_score) / 4.0, 4)

        CL_THRESHOLD = 0.75
        G_THRESHOLD = 0.9
        C_THRESHOLD = 0.5
        F_THRESHOLD = 0.75  # Soglia sotto la quale il prompt viene considerato mal formattato
        PQS_THRESHOLD = 0.7

        return {
            "metrics": metrics,
            "smells_detected": {
                # Conditional safe evaluation to prevent NoneType errors if one of the flags was False
                "reasoning_suppression": metrics["reasoning_score"] == 0.0 if metrics[
                                                                                  "reasoning_score"] is not None else False,
                "lack_of_self_reflection": metrics["self_reflection_present"] == 0 if metrics[
                                                                                          "self_reflection_present"] is not None else False,
                "role_suppression": metrics["role_assigned"] == 0 if metrics["role_assigned"] is not None else False,
                "unspecified_output_structure": metrics["structure_specified"] == 0 if metrics[
                                                                                           "structure_specified"] is not None else False,
                "lack_of_examples": metrics["examples_count"] == 0 if metrics["examples_count"] is not None else False,

                "complexity_length": metrics["complexity_length_score"] > CL_THRESHOLD if metrics[
                                                                                              "complexity_length_score"] is not None else False,
                "poor_grammar": metrics["grammatical_correctness_score"] < G_THRESHOLD if metrics[
                                                                                              "grammatical_correctness_score"] is not None else False,
                "poor_readability": metrics["readability_score"] < C_THRESHOLD if metrics[
                                                                                      "readability_score"] is not None else False,
                "poor_formatting": metrics["formatting_score"] < F_THRESHOLD if metrics[
                                                                                    "formatting_score"] is not None else False,
                "low_quality": metrics["prompt_quality_score"] < PQS_THRESHOLD if metrics[
                                                                                      "prompt_quality_score"] is not None else False,
            }
        }