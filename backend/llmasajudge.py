import pandas as pd
import json
import asyncio
# import ast
import os
from ollama import AsyncClient
from tqdm.asyncio import tqdm

# Configurazione
DATASET_PATH = "../dataset/final_dataset.parquet"
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
BASE_URL = "http://localhost:11434/v1"  # Porta standard di Ollama locale
MODEL_NAME = "qwen2.5:3b"  # IMPORTANTE: Sostituisci con l'identificativo esatto con cui è stato lanciato vLLM
CONCURRENCY_LIMIT = 4  # Quante chiamate API fare in parallelo (aggiusta in base al tuo rate limit)
MAX_RETRIES = 3  # Numero di tentativi in caso di errore dell'API

TIMEOUT_SECONDS = 45  # Tempo massimo di attesa (in secondi) prima di dichiarare la richiesta "morta"
CHECKPOINT_INTERVAL = 50  # Ogni quante righe salvare il dataset intermedio
OUTPUT_FILENAME = "../output/evaluated_prompts.parquet"

JUDGE_INSTRUCTIONS = """
You are an impartial annotator. Your only job: decide whether the prompt
explicitly assigned the AI a persona/professional role before or during
a software-related request.

DEFINITION — is_role_assigned = true when the prompt contains a clear
instruction directed at the assistant's own identity, e.g.:
- "You are a senior [role]...", "Act as a...", "I want you to act as...",
  "Pretend/imagine/assume you are...", "As an experienced [X], ..."
- A pasted system-prompt-like persona definition for the assistant to embody.

is_role_assigned = false when there is no such instruction, including these
common false-positive traps — do NOT count them as role assignment:
- "You are given an array/string/..." → describes problem INPUT, not the
  assistant's identity.
- References to roles inside the software/domain itself ("admin role",
  "the Player class", "as a user of this API") → domain entities, not the
  assistant.
- The user describing their OWN background ("I'm a junior dev...") → role of
  the user, not the model.
- Generic boilerplate with no real persona/expertise attached ("you are a
  helpful assistant") → weak/borderline; still counts as true but with lower
  confidence, since a label is technically present.

RULE: Scan the user's message for second-person/role language ("you are",
"act as", "pretend", "assume", "your role", "as a/an..."). If at least one
match is genuinely directed at the assistant's identity (not a false-positive
trap above), return true; otherwise return false.

CONFIDENCE (0-100, integer): how unambiguous the evidence is, regardless of
direction (a clear absence also deserves a high score).
- 90-100: unambiguous (explicit persona instruction, or unambiguous absence).
- 70-89: clear but with minor ambiguity or unusual phrasing.
- 50-69: genuinely borderline / could be read either way.
- <50: use only when forced to guess on very unclear input.

OUTPUT — strict, nothing else before/after:
{"is_role_assigned": <true|false>, "confidence_score": <integer<=100>}

PROMPT: 
"""


async def evaluate_prompt_with_llm(client: AsyncClient, user_prompt: str, semaphore: asyncio.Semaphore) -> dict:
    """
    Invia il prompt all'LLM e gestisce il ritorno in formato JSON.
    Usa un semaforo per limitare la concorrenza e gestisce i retry.
    """
    if not user_prompt:
        return {"is_role_assigned": None, "confidence_score": None, "error": "Empty prompt"}

    # Passo 3: Concatenare le istruzioni al prompt dell'utente
    full_prompt = f"{JUDGE_INSTRUCTIONS}\"\"\"{user_prompt}\"\"\""

    # Limitiamo il numero di chiamate concorrenti
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
                        format="json",
                        options={"temperature": 0.1}
                    ),
                    timeout=TIMEOUT_SECONDS
                )

                # Passo 4: Estrarre i due campi richiesti
                result_text = response['message']['content']
                result_json = json.loads(result_text)

                return {
                    "is_role_assigned": result_json.get("is_role_assigned", None),
                    "confidence_score": result_json.get("confidence_score", None),
                    "error": None
                }
            except asyncio.TimeoutError:
                if attempt == MAX_RETRIES - 1:
                    return {"is_role_assigned": None, "confidence_score": None, "error": "Timeout raggiunto"}
                print(f"Timeout al tentativo {attempt + 1}. Ritento...")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {"is_role_assigned": None, "confidence_score": None, "error": str(e)}
                await asyncio.sleep(2 ** attempt)  # Exponential backoff prima di riprovare


async def process_dataset(df: pd.DataFrame, client: AsyncClient) -> pd.DataFrame:
    """
    Itera su tutto il dataframe, prepara i task asincroni ed esegue l'LLM come giudice.
    """
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # print("Estrazione dei prompt utente (Passo 1 e 2)...")
    # user_prompts = df['conversation'].apply(extract_user_prompt).tolist()

    user_prompts = df[:]['natural_language_text'].tolist()

    # Eseguiamo i task con una progress bar per monitorare
    results = []
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

    with tqdm(total=len(user_prompts), desc="Valutazione in corso") as pbar:
        # Loop principale che suddivide in blocchi (chunks)
        for i in range(0, len(user_prompts), CHECKPOINT_INTERVAL):
            batch_prompts = user_prompts[i: i + CHECKPOINT_INTERVAL]

            # Prepariamo i task per questo singolo lotto
            tasks = [
                evaluate_prompt_with_llm(client, prompt, semaphore)
                for prompt in batch_prompts
            ]

            # Attendiamo che tutti i task di questo lotto finiscano
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # SALVATAGGIO INTERMEDIO (CHECKPOINT)
            temp_df = pd.DataFrame({
                "user_prompt": user_prompts[:len(results)],
                "is_role_assigned": [res.get("is_role_assigned") for res in results],
                "confidence_score": [res.get("confidence_score") for res in results],
                "error": [res.get("error") for res in results]
            })
            # Sovrascrive il file ad ogni checkpoint con i dati aggiornati
            temp_df.to_parquet(OUTPUT_FILENAME, index=False)

            # Aggiorniamo la barra di caricamento visiva
            pbar.update(len(batch_prompts))

    return temp_df


def main():
    # Lettura del dataset reale dal file parquet
    print(f"Uploading dataset from {DATASET_PATH}...")

    try:
        # Nota: assicurati di avere installato 'pyarrow' o 'fastparquet' (es. pip install pyarrow)
        df = pd.read_parquet(DATASET_PATH)
        df = df.reset_index(drop=True)  # FONDAMENTALE: previene collisioni di scrittura
        print(f"Dataset successfully uploaded. Rows to process: {len(df)}")
    except Exception as e:
        print(f"Critical error while loading the Parquet dataset: {e}")
        return

    # Inizializza il client Ollama puntando al server vLLM locale
    client = AsyncClient(host="http://localhost:11434")

    # Esegui il loop asincrono
    final_df = asyncio.run(process_dataset(df, client))

    # Salvataggio del risultato
    final_df.to_parquet(OUTPUT_FILENAME, index=False)
    print(f"\nProcessing completed! Results saved in: {OUTPUT_FILENAME}")

    total_prompts = len(final_df)
    if total_prompts > 0:
        # Usa pandas per sommare rapidamente tutti i valori True
        true_count = (final_df['is_role_assigned'] == True).sum()
        percentage = (true_count / total_prompts) * 100

        print(f"\n--- Statistiche Finali ---")
        print(f"Totale prompt analizzati: {total_prompts}")
        print(f"Ruoli assegnati (True): {true_count} ({percentage:.2f}%)")
    else:
        print("\nNessun prompt analizzato.")


if __name__ == "__main__":
    main()