import pandas as pd
import json
import asyncio
import os
import ast
from ollama import AsyncClient
from tqdm.asyncio import tqdm
from pydantic import BaseModel, Field

# Configurazione
DATASET_PATH = "../dataset/final_dataset.parquet"
OUTPUT_FILENAME = "../output/evaluated_prompts1.parquet"

BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen2.5:7b"
CONCURRENCY_LIMIT = 50
MAX_RETRIES = 3

TIMEOUT_SECONDS = 45
CHECKPOINT_INTERVAL = 50


# 1. Definizione dello Schema Pydantic per il Guided Decoding
class PromptEvaluation(BaseModel):
    is_role_assigned: bool
    is_reasoning_required: bool
    self_reflection_present: bool
    structure_specified: bool
    examples_count: int = Field(description="The exact number of examples provided. 0 if none.")
    reasoning: str = Field(description="The reasoning behind your decisions for each dimension.")


# 2. Istruzioni condensate per guidare l'attenzione del modello 7B
JUDGE_INSTRUCTIONS = """
You are an expert data annotator in natural language processing with an in-depth knowledge of prompt engineering.
You will be given a prompt for a Large Language Model (LLM): your job is to analyze the prompt and extract 5 specific dimensions.
After completing a single analysis, briefly explain the reasoning behind the choices you made.

DEFINITIONS:
1. ROLE SUPPRESSION (is_role_assigned): True if the prompt explicitly instruct an LLM to adopt a specific identity, role, or expertise (e.g., "Act as...", "You are an expert...").
2. REASONING SUPPRESSION (is_reasoning_required): True if the prompt is structured in such a way as to allow the LLM to reason through a Chain of Thought (e.g., "Let's think step by step", "First... Then...").
     Chain of Thought is a prompt engineering technique that instructs LLMs to show their step-by-step reasoning before giving a final answer.
3. LACK OF SELF-REFLECTION (self_reflection_present): True if LLM is asked to verify/review its output (e.g., "check to see if your answer is correct", "review the code").
    Self-reflection is a prompting strategy where an LLM is instructed to assess, critique, and revise its own generated outputs (e.g., code) before finalizing a response.
    There is no request for self-reflection if the LLM is asked to check the code in the user's prompt.
4. UNSPECIFIED OUTPUT STRUCTURE (structure_specified): True if a specific output format is requested (e.g., "output format: JSON", "structure the output as...").
5. LACK OF EXAMPLES (examples_count): Count the exact number of examples provided in the prompt (e.g., "Input: ... Output: ...", "Example 1: ..."). 0 if none.

EXAMPLES:
1. Prompt: Act as an expert Python programmer. Write a Python script to calculate the n-th Fibonacci number. Examples: fib(1)=1, fib(4)=3
Response: { "is_role_assigned": true, "is_reasoning_required": false, "self_reflection_present": false, "structure_specified": true, "examples_count": 2,
    "reasoning": As a Python programmer (role), the LLM is asked to generate Python code as output. Two examples are provided. }

2. Prompt: I want u to write code to calculate the factorial of a number. For example, if the user input is 5 the output must be 120.
    Explain the reasoning behind the code and double-check that it's correct. 
Response: { "is_role_assigned": false, "is_reasoning_required": true, "self_reflection_present": true, "structure_specified": false, "examples_count": 1,
    "reasoning": The LLM is asked to double-check its output and explain the reasoning process that produced it. It was also provided with an example of input-output"}

IMPORTANT: Some prompts may contain code snippets. You must not take any code strings or comments into account for you analysis under any circumstances!

PROMPT TO ANALYZE:
"""

def extract_user_prompt(conversation_field) -> str:
    """
    Estrae il contenuto del primo messaggio con role='user'.
    Gestisce il caso in cui il campo sia una stringa JSON, una stringa di dizionari Python o una lista.
    """
    try:
        # Se è una stringa (es. letta da CSV), proviamo a parsarla in una lista di dizionari
        if isinstance(conversation_field, str):
            try:
                # Prova prima con JSON
                convo = json.loads(conversation_field)
            except json.JSONDecodeError:
                # Fallback per rappresentazioni stringa di liste Python (es. create da print)
                convo = ast.literal_eval(conversation_field)
        else:
            convo = conversation_field  # Se è già una lista

        # Cerca il primo messaggio dell'utente
        for msg in convo:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "").strip()

    except Exception as e:
        print(f"Errore nel parsing della conversazione: {e}")

    return None

async def evaluate_prompt_with_llm(client: AsyncClient, user_prompt: str, semaphore: asyncio.Semaphore) -> dict:
    if not user_prompt:
        return {
            "is_role_assigned": None, "is_reasoning_required": None,
            "self_reflection_present": None, "structure_specified": None,
            "examples_count": None, "reasoning": None, "error": "Empty prompt"
        }

    full_prompt = f"{JUDGE_INSTRUCTIONS}\"\"\"{user_prompt}\"\"\""

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    client.chat(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a helpful JSON-outputting assistant."},
                            {"role": "user", "content": full_prompt}
                        ],
                        # 3. Forziamo l'output strutturato passando lo schema Pydantic
                        format=PromptEvaluation.model_json_schema(),
                        options={"temperature": 0.1}
                    ),
                    timeout=TIMEOUT_SECONDS
                )

                result_text = response['message']['content']
                result_json = json.loads(result_text)

                return {
                    "is_role_assigned": result_json.get("is_role_assigned"),
                    "is_reasoning_required": result_json.get("is_reasoning_required"),
                    "self_reflection_present": result_json.get("self_reflection_present"),
                    "structure_specified": result_json.get("structure_specified"),
                    "examples_count": result_json.get("examples_count"),
                    "reasoning" : result_json.get("reasoning"),
                    "error": None
                }
            except asyncio.TimeoutError:
                if attempt == MAX_RETRIES - 1:
                    return {"error": "Timeout raggiunto"}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {"error": str(e)}
                await asyncio.sleep(2 ** attempt)


async def process_dataset(df: pd.DataFrame, client: AsyncClient) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    user_prompts = df[:]['conversation'].apply(extract_user_prompt).tolist()

    results = []
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

    with tqdm(total=len(user_prompts), desc="Valutazione in corso") as pbar:
        for i in range(0, len(user_prompts), CHECKPOINT_INTERVAL):
            batch_prompts = user_prompts[i: i + CHECKPOINT_INTERVAL]

            tasks = [
                evaluate_prompt_with_llm(client, prompt, semaphore)
                for prompt in batch_prompts
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # 4. Aggiorniamo il salvataggio intermedio includendo tutte le colonne
            temp_df = pd.DataFrame({
                "user_prompt": user_prompts[:len(results)],
                "is_role_assigned": [res.get("is_role_assigned") for res in results],
                "is_reasoning_required": [res.get("is_reasoning_required") for res in results],
                "self_reflection_present": [res.get("self_reflection_present") for res in results],
                "structure_specified": [res.get("structure_specified") for res in results],
                "examples_count": [res.get("examples_count") for res in results],
                "reasoning": [res.get("reasoning") for res in results],
                "error": [res.get("error") for res in results]
            })

            temp_df.to_parquet(OUTPUT_FILENAME, index=False)
            pbar.update(len(batch_prompts))

    return temp_df


def main():
    print(f"Uploading dataset from {DATASET_PATH}...")

    try:
        df = pd.read_parquet(DATASET_PATH)
        df = df.reset_index(drop=True)

        # 5. Inizializziamo a None tutte le nuove colonne per prevenire collisioni
        for col in ["is_role_assigned", "is_reasoning_required", "self_reflection_present", "structure_specified",
                    "examples_count", "reasoning", "error"]:
            if col not in df.columns:
                df[col] = None

        print(f"Dataset successfully uploaded. Rows to process: {len(df)}")
    except Exception as e:
        print(f"Critical error while loading the Parquet dataset: {e}")
        return

    client = AsyncClient(host=BASE_URL)
    final_df = asyncio.run(process_dataset(df, client))

    print(f"\nProcessing completed! Results saved in: {OUTPUT_FILENAME}")

    # Check statistico rapido
    if len(final_df) > 0:
        valid_rows = final_df[final_df['error'].isnull()]
        print(f"\n--- Final statistics ({len(valid_rows)} processed without errors) ---")
        print(f"Role Assigned (True): {(valid_rows['is_role_assigned'] == True).sum()}")
        print(f"Self Reflection (True): {(valid_rows['self_reflection_present'] == True).sum()}")
        print(f"Specified Output Structure (True): {(valid_rows['structure_specified'] == True).sum()}")
        print(f"Reasoning (True): {(valid_rows['is_reasoning_required'] == True).sum()}")
        print(f"Examples (True): {(valid_rows['examples_count'] == True).sum()}")

if __name__ == "__main__":
    main()