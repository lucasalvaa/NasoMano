from __future__ import annotations

from typing import Callable, TypedDict

# To build the docker image
# from .llm_judge import LLMJudgeEvaluator
# from .syntactic_metrics import SyntacticMetricsEvaluator

from llm_judge import LLMJudgeEvaluator
from syntactic_metrics import SyntacticMetricsEvaluator


COMPLEXITY_LENGTH_THRESHOLD = 0.75
GRAMMATICAL_CORRECTNESS_THRESHOLD = 0.9
READABILITY_THRESHOLD = 0.6
FORMATTING_THRESHOLD = 0.75
PROMPT_QUALITY_THRESHOLD = 0.7


class Metrics(TypedDict):
    is_reasoning_required: int | None
    self_reflection_present: int | None
    role_assigned: int | None
    structure_specified: int | None
    examples_count: int | None
    complexity_length_score: float | None
    grammatical_correctness_score: float | None
    readability_score: float | None
    formatting_score: float | None
    prompt_quality_score: float | None


class SmellsDetected(TypedDict):
    reasoning_suppression: bool | None
    lack_of_self_reflection: bool | None
    role_suppression: bool | None
    unspecified_output_structure: bool | None
    lack_of_examples: bool | None
    complexity_length: bool | None
    poor_grammar: bool | None
    poor_readability: bool | None
    poor_formatting: bool | None
    low_quality: bool | None


class DetectionResult(TypedDict):
    llm_reasoning: str | None
    metrics: Metrics
    smells_detected: SmellsDetected


# Associate each metric to the pair (smell key, predicate that detects it)
# To add or remove a smell, simply edit a line here
_SMELL_RULES: dict[str, tuple[str, Callable[[float], bool]]] = {
    "is_reasoning_required": ("reasoning_suppression", lambda v: v == 0),
    "self_reflection_present": ("lack_of_self_reflection", lambda v: v == 0),
    "role_assigned": ("role_suppression", lambda v: v == 0),
    "structure_specified": ("unspecified_output_structure", lambda v: v == 0),
    "examples_count": ("lack_of_examples", lambda v: v == 0),
    "complexity_length_score": ("complexity_length", lambda v: v > COMPLEXITY_LENGTH_THRESHOLD),
    "grammatical_correctness_score": ("poor_grammar", lambda v: v < GRAMMATICAL_CORRECTNESS_THRESHOLD),
    "readability_score": ("poor_readability", lambda v: v < READABILITY_THRESHOLD),
    "formatting_score": ("poor_formatting", lambda v: v < FORMATTING_THRESHOLD),
    "prompt_quality_score": ("low_quality", lambda v: v < PROMPT_QUALITY_THRESHOLD),
}


class Naso:
    def __init__(self, model: str = "qwen2.5:3b") -> None:
        self.llm_judge = LLMJudgeEvaluator(model_name=model)
        self.syntactic_eval = SyntacticMetricsEvaluator()

    async def detect_smells(
        self,
        prompt: str,
        eval_judge_metrics: bool = True,
        eval_syntactic_metrics: bool = True,
    ) -> DetectionResult:
        """
        Analyze an individual prompt and return metrics and the presence of smells.

        Parameters:
            prompt (str): The prompt to analyze.
            eval_judge_metrics (bool, optional): Whether to evaluate metrics from 01 to 05.
            eval_syntactic_metrics (bool, optional): Whether to evaluate metrics from 06 to 10.

        Returns:
            metrics (dict): Dictionary containing the ten metrics reported by the evaluator.

        Raises:
            ValueError: if the prompt provided is empty or an invalid string.
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The prompt provided is empty.")

        metrics: Metrics = self._empty_metrics()
        llm_reasoning: str | None = None

        if eval_judge_metrics:
            llm_reasoning, llm_metrics = await self._evaluate_llm_metrics(prompt)
            metrics.update(llm_metrics)

        if eval_syntactic_metrics:
            syntactic_metrics = await self._evaluate_syntactic_metrics(prompt)
            metrics.update(syntactic_metrics)

        return {
            "llm_reasoning": llm_reasoning,
            "metrics": metrics,
            "smells_detected": self._flag_smells(metrics),
        }

    @staticmethod
    def _empty_metrics() -> Metrics:
        return {
            "is_reasoning_required": None,
            "self_reflection_present": None,
            "role_assigned": None,
            "structure_specified": None,
            "examples_count": None,
            "complexity_length_score": None,
            "grammatical_correctness_score": None,
            "readability_score": None,
            "formatting_score": None,
            "prompt_quality_score": None,
        }

    async def _evaluate_llm_metrics(self, prompt: str) -> tuple[str | None, dict]:
        llm_results = await self.llm_judge.evaluate(prompt)
        if llm_results is None:
            return None, {}

        return llm_results.get("llm_reasoning"), {
            "is_reasoning_required": int(bool(llm_results.get("is_reasoning_required"))),
            "self_reflection_present": int(bool(llm_results.get("self_reflection_present"))),
            "role_assigned": int(bool(llm_results.get("is_role_assigned"))),
            "structure_specified": int(bool(llm_results.get("structure_specified"))),
            "examples_count": llm_results.get("examples_count", 0),
        }

    async def _evaluate_syntactic_metrics(self, prompt: str) -> dict:
        syntactic_results = await self.syntactic_eval.evaluate(prompt)
        return {
            "complexity_length_score": syntactic_results.get("complexity_length_score"),
            "grammatical_correctness_score": syntactic_results.get("grammatical_correctness_score"),
            "readability_score": syntactic_results.get("readability_score"),
            "formatting_score": syntactic_results.get("formatting_score"),
            "prompt_quality_score": syntactic_results.get("prompt_quality_score"),
        }

    @staticmethod
    def _flag_smells(metrics: Metrics) -> SmellsDetected:
        return {
            smell_key: (predicate(metrics[metric_key]) if metrics[metric_key] is not None else None)
            for metric_key, (smell_key, predicate) in _SMELL_RULES.items()
        }