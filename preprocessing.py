# dementia_utils/preprocessing.py

import re
import pandas as pd
from pathlib import Path

def preprocess_transcript(raw_text: str) -> list:
    """
    Extract participant speech lines from a raw transcript and clean them.
    Preserves contractions and filler tokens like &-um.
    """
    sentences = []

    # Extract participant lines
    for line in raw_text.splitlines():
        if line.startswith("*PAR:"):
            line = line[len("*PAR:"):].strip()
            sentences.append(line)

    # Clean up control characters, tabs, and timestamps
    sentences = [re.sub(r'[\x00-\x1F\x7F]', '', s) for s in sentences]
    sentences = [s.replace("\\t", "") for s in sentences]
    sentences = [re.sub(r'\[\+ exc\] [0-9]+_[0-9]+', '', s) for s in sentences]

    cleaned_sentences = []

    for sentence in sentences:
        # Split sentence into tokens
        tokens = sentence.split()

        # Keep alphabetic words, contractions, and filler tokens (&-um etc.)
        filtered_tokens = [t for t in tokens if re.match(r"[A-Za-z]+(?:'[A-Za-z]+)?|[&+-].*", t)]

        # Recombine into cleaned sentence
        cleaned_sentences.append(" ".join(filtered_tokens))

    return cleaned_sentences


from pathlib import Path
import re
import pandas as pd


def load_patient_data(
    df: pd.DataFrame,
    base_folder: str = "./Pitt/Dementia/",
    task_folders = ("cookie", "fluency", "recall", "sentence"),
) -> dict:
    """
    Load patient data and match it to available .cha transcripts across tasks.
    
    Returns:
        patient_data[patient_id][visit_date] = {
            "mmse": int,
            "visit_index": int,
            "tasks": {
                task_name: {
                    "preprocessed_transcript": str
                }
            }
        }
    """

    patient_data = {}
    DATE_PATTERN = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"

    for row_idx in range(len(df)):
        patient = df.loc[row_idx]
        patient_id = str(patient["id"]).zfill(3)

        visits, mmses = [], []
        visit_num = 1

        # ---- Collect visit dates + MMSE values ----
        while True:
            visit_col = f"visit{visit_num}"
            mmse_col = f"mmse{visit_num}"

            if visit_col not in patient or mmse_col not in patient:
                break

            raw_visit = patient[visit_col]
            raw_mmse = patient[mmse_col]

            if pd.isna(raw_visit) or pd.isna(raw_mmse):
                break

            match = re.search(DATE_PATTERN, str(raw_visit))
            if not match:
                break

            visits.append(match.group())
            mmses.append(int(raw_mmse))
            visit_num += 1

        visit_data = {}

        # ---- Load transcripts for each visit ----
        for idx, (visit_date, mmse) in enumerate(zip(visits, mmses)):
            task_data = {}

            # Try each task folder
            for task in task_folders:
                transcript_path = (
                    Path(base_folder) / task / f"{patient_id}-{idx}.cha"
                )

                try:
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        transcript = f.read()
                except FileNotFoundError:
                    continue  # skip missing tasks

                preprocessed = preprocess_transcript(transcript)

                task_data[task] = {
                    "preprocessed_transcript": preprocessed
                }

            # Only store visit if at least one task transcript exists
            if task_data:
                visit_data[visit_date] = {
                    "mmse": mmse,
                    "visit_index": idx,
                    "tasks": task_data
                }

        if visit_data:
            patient_data[patient_id] = visit_data

    return patient_data

