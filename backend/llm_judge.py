from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pydantic import BaseModel, Field
from typing import Any

from ollama import AsyncClient


# Pydantic schema for guided decoding
class PromptEvaluation(BaseModel):
    reasoning: str = Field(description="The reasoning behind your decisions for each dimension.")
    is_role_assigned: bool
    is_reasoning_required: bool
    self_reflection_present: bool
    structure_specified: bool
    examples_count: int = Field(description="The exact number of examples provided. 0 if none.")


JUDGE_INSTRUCTIONS = """
You are an expert data annotator in natural language processing with an in-depth knowledge of prompt engineering.
You will be given a prompt for a Large Language Model (LLM): your job is to analyze the prompt and extract 5 specific dimensions based on the prompt patterns described below.

DEFINITIONS & PATTERNS TO DETECT:
1. is_role_assigned: True if the prompt uses a Persona/Role Pattern, explicitly instructing the LLM to adopt a specific identity, expert role, or persona (e.g., "Act as...", "You are an expert...", "Pretend to be...").
2. is_reasoning_required: True if the prompt explicitly uses a Chain-of-Thought (CoT) pattern, instructing the LLM to break down its reasoning step-by-step before providing a final answer (e.g., "Let's think step by step", "First... Then...", "Explain your reasoning step by step").
3. self_reflection_present: True if the prompt uses a Self-Correction / Self-Reflection pattern, instructing the LLM to verify, review, critique, or double-check its OWN output before finalizing the response (e.g., "check to see if your answer is correct", "review your code before replying"). False if the LLM is only asked to review code provided in the user's input.
4. structure_specified: True if the prompt uses an Output Format Constraint pattern, explicitly requesting a specific structural format or constraint for the response (e.g., "output format: JSON", "structure the output as a table", "respond only in Python code").
5. examples_count: Exact integer count of input-output or few-shot examples provided within the prompt (e.g., "Input: ... Output: ...", "Example 1: ..."). 0 if none.

IMPORTANT:
- Write the 'reasoning' field FIRST in your JSON output to analyze the prompt step-by-step before assigning boolean and integer values.
- Some prompts may contain code snippets. You must not take any code strings or comments into account for your analysis under any circumstances!
- When explaining your reasoning, be careful not to confuse the user's request with your job as an annotator.
- When assigning a value to the `is_reasoning_requested` field, consider only any user requests made to the LLM—not your own reasoning regarding the `reasoning` field.

EXAMPLES:

1. Prompt: Act as an expert Python programmer. Write a Python script to calculate the n-th Fibonacci number. Examples: fib(1)=1, fib(4)=3
Response: {
    "reasoning": "The prompt explicitly assigns the persona of an expert Python programmer ('Act as an expert...'). No Chain-of-Thought or self-reflection is requested. It asks for code, specifying the expected output structure. Two examples are explicitly provided.",
    "is_role_assigned": true,
    "is_reasoning_required": false,
    "self_reflection_present": false,
    "structure_specified": true,
    "examples_count": 2
}

2. Prompt: I want u to write code to calculate the factorial of a number. For example, if the user input is 5 the output must be 120. Explain the reasoning behind the code and double-check that it's correct.
Response: {
    "reasoning": "No persona/role is assigned to the model. The prompt asks to explain the reasoning behind the code (Chain-of-Thought requested) and explicitly instructs the model to double-check its own output (Self-reflection pattern). No specific output structure is constrained. One example of input/output is provided.",
    "is_role_assigned": false,
    "is_reasoning_required": true,
    "self_reflection_present": true,
    "structure_specified": false,
    "examples_count": 1
}
"""

logger = logging.getLogger(__name__)
MAX_BACKOFF_SECONDS = 30.0


class LLMJudgeEvaluator:
    def __init__(
            self,
            model_name: str = "qwen2.5:3b",
            base_url: str | None = None,
            concurrency_limit: int = 50,
            max_retries: int = 3,
            timeout_seconds: float = 45,
    ) -> None:
        """
        Initialize the LLM judge with a global semaphore to protect the
        AI server from API traffic spikes.

         Parameters:
            model_name (str, optional): Name of the Large Language Model to use.
            base_url (str, optional): The URL of the Ollama server. If not specified, it is read
                from the environment variable OLLAMA_BASE_URL, with a fallback to localhost.
            concurrency_limit (int, optional): The maximum number of evaluate() calls allowed
                to run concurrently against the Ollama server. Additional calls wait on the
                semaphore instead of being sent immediately, to avoid overwhelming the server
                with traffic spikes.
            max_retries (int, optional): The maximum number of attempts made to obtain a valid
                response before giving up and returning an error response. Must be at least 1.
            timeout_seconds (float, optional): The maximum time, in seconds, to wait for a single
                response from the model before considering that attempt timed out and retrying.

        Raises:
            ValueError: if concurrency_limit is less than 1, if max_retries is less than 1, or
                if timeout_seconds is not strictly positive.
        """
        if concurrency_limit < 1:
            raise ValueError("concurrency_limit must be at least 1.")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        resolved_base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )

        self.client = AsyncClient(host=resolved_base_url)
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def evaluate(self, prompt: str) -> dict[str, Any]:
        """
        Evaluate a prompt using an LLM as a judge, with retries and exponential
        backoff (with jitter) in the event of a timeout or error.

        Parameters:
            prompt (str): The prompt to analyze.

        Returns:
            A dict with evaluated metrics, or with “error” set
            in the event of failure after all attempts.

        Raises:
            ValueError: if the prompt provided is empty or an invalid string.
        """
        if not prompt or not prompt.strip():
            raise ValueError("The prompt provided is empty.")

        full_prompt = f"{JUDGE_INSTRUCTIONS}\n\nPROMPT TO ANALYZE:\n\"\"\"{prompt}\"\"\""

        async with self.semaphore:
            for attempt in range(self.max_retries):
                is_last_attempt = attempt == self.max_retries - 1
                try:
                    response = await asyncio.wait_for(
                        self.client.chat(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful JSON-outputting assistant."},
                                {"role": "user", "content": full_prompt},
                            ],
                            # Guided Decoding: the LLM is forced to output a specific JSON schema
                            format=PromptEvaluation.model_json_schema(),
                            options={"temperature": 0.1},
                        ),
                        timeout=self.timeout_seconds,
                    )
                    return self._parse_response(response)

                except asyncio.TimeoutError:
                    logger.warning(
                        "LLM judge timeout (attempt %d/%d, model=%s)",
                        attempt + 1, self.max_retries, self.model_name,
                    )
                    if is_last_attempt:
                        return self._error_response("Timeout reached")

                except json.JSONDecodeError:
                    logger.warning(
                        "LLM judge returned malformed JSON (attempt %d/%d, model=%s)",
                        attempt + 1, self.max_retries, self.model_name,
                    )
                    if is_last_attempt:
                        return self._error_response("Malformed response from the model")

                except Exception:
                    logger.exception(
                        "Unexpected error from LLM judge (attempt %d/%d, model=%s)",
                        attempt + 1, self.max_retries, self.model_name,
                    )
                    if is_last_attempt:
                        return self._error_response("Unexpected error contacting the LLM judge")

                await self._backoff(attempt)

        raise AssertionError("unreachable")

    @staticmethod
    def _parse_response(response: dict) -> dict[str, Any]:
        result_text = response["message"]["content"]
        result_json = json.loads(result_text)

        return {
            "is_role_assigned": result_json.get("is_role_assigned"),
            "is_reasoning_required": result_json.get("is_reasoning_required"),
            "self_reflection_present": result_json.get("self_reflection_present"),
            "structure_specified": result_json.get("structure_specified"),
            "examples_count": result_json.get("examples_count", 0),
            "llm_reasoning": result_json.get("reasoning"),
            "error": None,
        }

    @staticmethod
    async def _backoff(attempt: int) -> None:
        delay = min(2 ** attempt, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5)
        await asyncio.sleep(delay)

    @staticmethod
    def _error_response(error_msg: str) -> dict[str, Any]:
        """
        Helper method to return a clean JSON object in case of failure.
        """
        return {
            "is_reasoning_required": None,
            "self_reflection_present": None,
            "is_role_assigned": None,
            "structure_specified": None,
            "examples_count": None,
            "llm_reasoning": None,
            "error": error_msg,
        }
