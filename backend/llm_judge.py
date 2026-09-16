import json
import asyncio
from pydantic import BaseModel, Field
from ollama import AsyncClient

# Pydantic schema for guided decoding
class PromptEvaluation(BaseModel):
    is_role_assigned: bool
    is_reasoning_required: bool
    self_reflection_present: bool
    structure_specified: bool
    examples_count: int = Field(description="The exact number of examples provided. 0 if none.")
    reasoning: str = Field(description="The reasoning behind your decisions for each dimension.")


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
- When explaining your reasoning and , be careful not to confuse the user's request with your job as an annotator.
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


class LLMJudgeEvaluator:
    def __init__(self,
                 model_name="qwen2.5:3b",
                 base_url="http://127.0.0.1:11434",
                 concurrency_limit=50,
                 max_retries=3,
                 timeout_seconds=45):
        """
        Initialize the LLM judge with a global semaphore
        to protect the AI server from API traffic spikes.
        """
        self.client = AsyncClient(host=base_url)
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def evaluate(self, prompt: str) -> dict | None:

        if not prompt or not prompt.strip():
            raise ValueError("The prompt provided is empty.")

        full_prompt = f"{JUDGE_INSTRUCTIONS}\n\nPROMPT TO ANALYZE:\n\"\"\"{prompt}\"\"\""

        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await asyncio.wait_for(
                        self.client.chat(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful JSON-outputting assistant."},
                                {"role": "user", "content": full_prompt}
                            ],
                            # Guided Decoding: the LLM is forced to output a specific JSON schema
                            format=PromptEvaluation.model_json_schema(),
                            options={"temperature": 0.1}
                        ),
                        timeout=self.timeout_seconds
                    )

                    result_text = response['message']['content']
                    result_json = json.loads(result_text)

                    return {
                        "is_role_assigned": result_json.get("is_role_assigned"),
                        "is_reasoning_required": result_json.get("is_reasoning_required"),
                        "self_reflection_present": result_json.get("self_reflection_present"),
                        "structure_specified": result_json.get("structure_specified"),
                        "examples_count": result_json.get("examples_count", 0),
                        "llm_reasoning": result_json.get("reasoning"),
                        "error": None
                    }

                except asyncio.TimeoutError:
                    if attempt == self.max_retries - 1:
                        return self._error_response("Timeout reached")
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return self._error_response(e)
                    await asyncio.sleep(2 ** attempt)


    def _error_response(self, error_msg: str | Exception) -> dict:
        """Helper method to return a clean JSON object in case of failure."""
        return {
            "reasoning_score": None,
            "self_reflection_present": None,
            "role_assigned": None,
            "structure_specified": None,
            "examples_count": None,
            "llm_reasoning": None,
            "error": error_msg
        }