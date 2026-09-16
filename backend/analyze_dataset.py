import json
import time
import random
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from naso import PromptSmellDetector
from tqdm import tqdm

INPUT_PATH = "../dataset/final_dataset.parquet"
OUTPUT_PATH_PARQUET = "../dataset/final_dataset_from_conversation.parquet"
CONVERSATION_COLUMN = "conversation"
MAX_WORKERS = 8  # Puoi aumentare o diminuire in base ai core della tua CPU

# Variabile globale per mantenere un'istanza del detector in ciascun processo isolato
global_detector = None


def init_worker():
    """Inizializza un'istanza del detector isolata all'interno di ogni processo lavoratore."""
    global global_detector
    global_detector = PromptSmellDetector()


def extract_user_content(conversation_raw: str) -> str:
    """
    Extracts and concatenates the content of all messages with role == 'user'
    from a conversation saved as a JSON string.
    Returns an empty string if parsing fails or there are no user messages.
    """
    if not isinstance(conversation_raw, str) or not conversation_raw.strip():
        return ""
    try:
        messages = json.loads(conversation_raw)
    except (json.JSONDecodeError, TypeError):
        return ""

    user_contents = [
        m.get("content", "")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
    ]
    return "\n".join(user_contents)


def process_single_prompt(text: str) -> dict:
    """
    Worker function executed by each process to analyze a single prompt string.
    Returns a dictionary containing all metrics and smells. Empty/Failed prompts return None values.
    """
    # Dizionario di default con valori nulli per mantenere la struttura del DataFrame
    empty_result = {
        "reasoning_score": None,
        "self_reflection_present": None,
        "role_assigned": None,
        "structure_specified": None,
        "examples_count": None,
        "reasoning_suppression": None,
        "lack_of_self_reflection": None,
        "role_suppression": None,
        "unspecified_output_structure": None,
        "lack_of_examples": None,
        "complexity_length": None,
        "poor_grammar": None,
        "poor_formatting": None,
        "poor_readability": None,
        "low_quality": None,
        "total_smells": None,
    }

    if not isinstance(text, str) or not text.strip():
        return empty_result

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Usa l'istanza globale isolata per questo specifico processo
            global global_detector
            result = global_detector.analyze_prompt(text)
            metrics = result["metrics"]
            smells = result["smells_detected"]

            return {
                "reasoning_score": metrics.get("reasoning_score"),
                "self_reflection_present": metrics.get("self_reflection_present"),
                "role_assigned": metrics.get("role_assigned"),
                "structure_specified": metrics.get("structure_specified"),
                "examples_count": metrics.get("examples_count"),
                "reasoning_suppression": smells.get("reasoning_suppression"),
                "lack_of_self_reflection": smells.get("lack_of_self_reflection"),
                "role_suppression": smells.get("role_suppression"),
                "unspecified_output_structure": smells.get("unspecified_output_structure"),
                "lack_of_examples": smells.get("lack_of_examples"),
                "complexity_length": smells.get("complexity_length"),
                "poor_grammar": smells.get("poor_grammar"),
                "poor_formatting": smells.get("poor_formatting"),
                "poor_readability": smells.get("poor_readability"),
                "low_quality": smells.get("low_quality"),
                "total_smells": sum(smells.values()) if smells else 0,
            }
        except Exception as e:
            error_str = str(e)

            # Se l'errore è causato dal crash o riavvio del container Docker
            if "Connection" in error_str or "10061" in error_str or "Max retries" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff: aspetta sempre di più prima di riprovare (es. 2.5s, 5s...)
                    # Il "random.uniform" aggiunge un jitter per sfalsare i thread e non sovraccaricare Docker al riavvio
                    sleep_time = (attempt + 1) * 2.5 + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    continue  # Riprova il ciclo for

            # Se non è un errore di connessione o abbiamo esaurito i tentativi, restituisci null
            # print(f"Failed after {attempt+1} attempts. Error: {e}")
            return empty_result


def analyze_dataset(input_path: str, max_workers: int = 4) -> pd.DataFrame:
    """
    Loads dataset, extracts user prompts, and runs multi-processing analysis.
    """
    print(f"Reading dataset from: {input_path}")
    df = pd.read_parquet(input_path)

    print("Extracting user prompts from 'conversation' column...")
    df["user_prompt"] = df[CONVERSATION_COLUMN].apply(extract_user_content)

    prompts = df["user_prompt"].tolist()

    print(f"Analyzing {len(prompts)} prompts using {max_workers} isolated processes...")

    # Esecuzione multiprocesso tramite ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
        results = list(
            tqdm(
                executor.map(process_single_prompt, prompts, chunksize=10),
                total=len(prompts),
                desc="Analyzing prompts (Multiprocessing)",
                unit="prompt",
            )
        )

    res_df = pd.DataFrame(results)
    for col in res_df.columns:
        df[col] = res_df[col]

    return df


def main():
    print(f"Loading dataset from: {INPUT_PATH}")
    df_enriched = analyze_dataset(INPUT_PATH)

    n_total = len(df_enriched)
    n_extracted = df_enriched["user_prompt"].str.strip().astype(bool).sum()
    print(f"\nTotal rows: {n_total}")
    print(f"Rows with 'user' content extracted correctly: {n_extracted}")

    smell_cols = [
        "reasoning_suppression",
        "lack_of_self_reflection",
        "role_suppression",
        "unspecified_output_structure",
        "lack_of_examples",
        "complexity_length",
        "poor_grammar",
        "poor_formatting",
        "poor_readability",
        "low_quality",
        "total_smells",
    ]

    print("\n--- Percentage of prompts with each smell (from 'conversation') ---")
    for col in smell_cols:
        if col in df_enriched.columns:
            pct = df_enriched[col].mean() * 100
            print(f"{col:35s} {pct:5.1f}%")

    df_enriched.to_parquet(OUTPUT_PATH_PARQUET, index=False)
    print(f"\nSaved enriched dataset to: {OUTPUT_PATH_PARQUET}")


if __name__ == "__main__":
    main()