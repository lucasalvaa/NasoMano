"""
Apply PromptSmellDetector (naso.py) to the message content with
role == “user” extracted from the JSON in the ‘conversation’ column.

It does not use the `natural_language_text` column in any way.
"""

import json
import pandas as pd
from naso import PromptSmellDetector

INPUT_PATH = "../dataset/final_dataset.parquet"
OUTPUT_PATH_PARQUET = "../dataset/final_dataset_from_conversation.parquet"
CONVERSATION_COLUMN = "conversation"


def extract_user_content(conversation_raw) -> str:
    """
    Extracts and concatenates the content of all messages with role == “user”
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


def analyze_dataset(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    detector = PromptSmellDetector()

    df["user_prompt"] = df[CONVERSATION_COLUMN].apply(extract_user_content)

    metric_cols = {
        "reasoning_score": [],
        "self_reflection_present": [],
        "role_assigned": [],
        "structure_specified": [],
        "examples_count": [],
        "reasoning_suppression": [],
        "lack_of_self_reflection": [],
        "role_suppression": [],
        "unspecified_output_structure": [],
        "lack_of_examples": [],
        "total_smells": [],
    }

    for text in df["user_prompt"]:
        if not isinstance(text, str) or not text.strip():
            for col in metric_cols:
                metric_cols[col].append(None)
            continue

        result = detector.analyze_prompt(text)
        metrics = result["metrics"]
        smells = result["smells_detected"]

        metric_cols["reasoning_score"].append(metrics["reasoning_score"])
        metric_cols["self_reflection_present"].append(metrics["self_reflection_present"])
        metric_cols["role_assigned"].append(metrics["role_assigned"])
        metric_cols["structure_specified"].append(metrics["structure_specified"])
        metric_cols["examples_count"].append(metrics["examples_count"])

        metric_cols["reasoning_suppression"].append(smells["reasoning_suppression"])
        metric_cols["lack_of_self_reflection"].append(smells["lack_of_self_reflection"])
        metric_cols["role_suppression"].append(smells["role_suppression"])
        metric_cols["unspecified_output_structure"].append(smells["unspecified_output_structure"])
        metric_cols["lack_of_examples"].append(smells["lack_of_examples"])

        metric_cols["total_smells"].append(sum(smells.values()))

    for col, values in metric_cols.items():
        df[col] = values

    return df


def main():
    print(f"Loading dataset from: {INPUT_PATH}")
    df_enriched = analyze_dataset(INPUT_PATH)

    n_total = len(df_enriched)
    n_extracted = df_enriched["user_prompt"].str.strip().astype(bool).sum()
    print(f"Total rows: {n_total}")
    print(f"Rows with 'user' content extracted correctly: {n_extracted}")

    smell_cols = [
        "reasoning_suppression",
        "lack_of_self_reflection",
        "role_suppression",
        "unspecified_output_structure",
        "lack_of_examples",
    ]
    print("\n--- Percentage of prompts with each smell (from “conversation”) ---")
    for col in smell_cols:
        pct = df_enriched[col].mean() * 100
        print(f"{col:35s} {pct:5.1f}%")

    df_enriched.to_parquet(OUTPUT_PATH_PARQUET, index=False)
    print(f"\nSaved to: {OUTPUT_PATH_PARQUET}")


if __name__ == "__main__":
    main()