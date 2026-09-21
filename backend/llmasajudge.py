import pandas as pd
import json
import asyncio
import os
from tqdm.asyncio import tqdm
from llm_judge import LLMJudgeEvaluator

# Configurazione
DATASET_PATH = "../dataset/final_dataset.parquet"
OUTPUT_FILENAME = "../output/test.parquet"

BASE_URL = "http://127.0.0.1:11434" # Ollama socket
judge = LLMJudgeEvaluator(model_name="qwen2.5:7b")
CHECKPOINT_INTERVAL = 50

def extract_user_prompt(conversation_field) -> str:
    """
    Estrae il contenuto del primo messaggio dell'utente dal JSON array della conversazione.
    """
    try:
        convo = json.loads(conversation_field) if isinstance(conversation_field, str) else conversation_field

        if isinstance(convo, list):
            for msg in convo:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "").strip()

    except Exception as e:
        print(f"An error occurred while extracting users' prompts from the conversation field: {e}")

    return None


async def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    conversation_ids = df[:5]['conversation_id'].tolist()
    user_prompts = df[:5]['conversation'].apply(extract_user_prompt).tolist()

    results = []
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

    with tqdm(total=len(user_prompts), desc="Evaluating...") as pbar:
        for i in range(0, len(user_prompts), CHECKPOINT_INTERVAL):
            batch_prompts = user_prompts[i: i + CHECKPOINT_INTERVAL]

            tasks = [
                judge.evaluate(prompt)
                for prompt in batch_prompts
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            temp_df = pd.DataFrame({
                "conversation_id": conversation_ids[:len(results)],
                "user_prompt": user_prompts[:len(results)],
                "is_role_assigned": [res.get("is_role_assigned") for res in results],
                "is_reasoning_required": [res.get("is_reasoning_required") for res in results],
                "self_reflection_present": [res.get("self_reflection_present") for res in results],
                "structure_specified": [res.get("structure_specified") for res in results],
                "examples_count": [res.get("examples_count") for res in results],
                "llm_reasoning": [res.get("llm_reasoning") for res in results],
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
                    "examples_count", "llm_reasoning", "error"]:
            if col not in df.columns:
                df[col] = None

        print(f"Dataset successfully uploaded. Rows to process: {len(df)}")
    except Exception as e:
        print(f"Critical error while loading the Parquet dataset: {e}")
        return

    final_df = asyncio.run(process_dataset(df))

    print(f"\nProcessing completed! Results saved in: {OUTPUT_FILENAME}")

    # Check statistico rapido
    if len(final_df) > 0:
        valid_rows = final_df[final_df['error'].isnull()]
        valid_count = len(valid_rows)

        stats: dict = {
            "role_assigned": (valid_rows['is_role_assigned'] == True).sum(),
            "self_ref_present": (valid_rows['self_reflection_present'] == True).sum(),
            "output_structure_spec": (valid_rows['structure_specified'] == True).sum(),
            "reasoning_req": (valid_rows['is_reasoning_required'] == True).sum(),
            "example_provided": (len(valid_rows['examples_count'] > 0)),

            # Percentage of smelly prompts entries out of the total number of lines
            "role_suppression_pct": (valid_rows['is_role_assigned'] == False).sum() / valid_count * 100,
            "lack_of_self_reflection_pct": (valid_rows['self_reflection_present'] == False).sum() / valid_count * 100,
            "unspecified_output_structure_pct": (valid_rows['structure_specified'] == False).sum() / valid_count * 100,
            "reasoning_suppression_pct": (valid_rows['is_reasoning_required'] == False).sum() / valid_count * 100,
            "lack_of_examples_pct": (len(valid_rows['examples_count'] == 0)) / valid_count * 100
        }

        print(f"\n--- Final statistics ({len(valid_rows)} rows processed without errors) ---")
        print(f"Role Assigned in {stats['role_assigned']} rows")
        print(f"Self-Reflection required in {stats['self_ref_present']} rows")
        print(f"Output Structure specified in {stats['output_structure_spec']} rows")
        print(f"Reasoning required in: {stats['reasoning_req']} rows")
        print(f"At least one example provided in {stats['example_provided']} rows")

        print(f"\n--- Prompt Smells Detected (%) ---")
        print(f"Role Suppression: {stats['role_suppression_pct']:.2f}%")
        print(f"Lack of Self-Reflection: {stats['lack_of_self_reflection_pct']:.2f}%")
        print(f"Unspecified Output Structure: {stats['unspecified_output_structure_pct']:.2f}%")
        print(f"Reasoning Suppression: {stats['reasoning_suppression_pct']:.2f}%")
        print(f"Lack of Examples: {stats['lack_of_examples_pct']:.2f}%")



if __name__ == "__main__":
    main()