from llm_judge import LLMJudgeEvaluator
from syntactic_metrics import SyntacticMetricsEvaluator


class PromptSmellDetector:
    """
    Detect the presence of the following ten “prompt smells” in English-Language prompts:
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
        self.llm_judge = LLMJudgeEvaluator()
        self.syntactic_eval = SyntacticMetricsEvaluator()

    async def analyze_prompt(
            self,
            prompt: str,
            eval_judge_metrics: bool = True,
            eval_syntactic_metrics: bool = True
    ) -> dict:
        """
        Analyze an individual prompt and return metrics and the presence of smells.

        Parameters:
            prompt (str): The prompt to analyze.
            eval_judge_metrics (bool, optional): Whether to evaluate metrics from 01 to 05.
            eval_syntactic_metrics (bool, optional): Whether to evaluate metrics from 06 to 10.

        Returns:
            metrics (dict): Dictionary containing the ten metrics reported by the evaluator.

            If any of the two flags is set to false, the return value for those five smells is "None".
        """
        if not prompt or not isinstance(prompt, str) or prompt.strip() == "":
            raise ValueError("The prompt provided is empty.")

        metrics: dict = {
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

        if eval_judge_metrics:
            llm_results = await self.llm_judge.evaluate(prompt)
            metrics["reasoning_score"] = 1 if llm_results.get("is_reasoning_required") else 0
            metrics["self_reflection_present"] = 1 if llm_results.get("self_reflection_present") else 0
            metrics["role_assigned"] = 1 if llm_results.get("is_role_assigned") else 0
            metrics["structure_specified"] = 1 if llm_results.get("structure_specified") else 0
            metrics["examples_count"] = llm_results.get("examples_count", 0)

        if eval_syntactic_metrics:
            syntactic_eval_results = self.syntactic_eval.evaluate(prompt)
            metrics["complexity_length_score"] = syntactic_eval_results.get("complexity_length_score", None)
            metrics["grammatical_correctness_score"] = syntactic_eval_results.get("grammatical_correctness_score", None)
            metrics["readability_score"] = syntactic_eval_results.get("readability_score", None)
            metrics["formatting_score"] = syntactic_eval_results.get("formatting_score", None)
            metrics["prompt_quality_score"] = syntactic_eval_results.get("prompt_quality_score", None)

        # The thresholds for determining whether a syntactic prompt smell is present
        CL_THRESHOLD = 0.75  # Complexity Length
        G_THRESHOLD = 0.9  # Grammatical Correctness
        C_THRESHOLD = 0.6  # Readability
        F_THRESHOLD = 0.75  # Formatting
        PQ_THRESHOLD = 0.7  # Prompt Quality

        def check_smell(value, condition):
            return condition(value) if value is not None else None

        return {
            "metrics": metrics,
            "smells_detected": {
                "reasoning_suppression": check_smell(metrics["reasoning_score"], lambda x: x == 0.0),
                "lack_of_self_reflection": check_smell(metrics["self_reflection_present"], lambda x: x == 0),
                "role_suppression": check_smell(metrics["role_assigned"], lambda x: x == 0),
                "unspecified_output_structure": check_smell(metrics["structure_specified"], lambda x: x == 0),
                "lack_of_examples": check_smell(metrics["examples_count"], lambda x: x == 0),
                "complexity_length": check_smell(metrics["complexity_length_score"], lambda x: x > CL_THRESHOLD),
                "poor_grammar": check_smell(metrics["grammatical_correctness_score"], lambda x: x < G_THRESHOLD),
                "poor_readability": check_smell(metrics["readability_score"], lambda x: x < C_THRESHOLD),
                "poor_formatting": check_smell(metrics["formatting_score"], lambda x: x < F_THRESHOLD),
                "low_quality": check_smell(metrics["prompt_quality_score"], lambda x: x < PQ_THRESHOLD),
            }
        }
