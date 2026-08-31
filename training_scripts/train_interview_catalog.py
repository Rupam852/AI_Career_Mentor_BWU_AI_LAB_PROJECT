"""
train_interview_catalog.py
--------------------------
Processes 200,000 interview dataset records to train the Difficulty Evaluator Model
and compiles a massive, role-specific comprehensive Interview Question Bank 
covering Technical, Behavioral, and Situational questions for all 63 career fields.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "AI_Career_Mentor_datasets", "04_interview_questions_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "AI_Career_Mentor_models")
OUTPUT_MODEL_PATH = os.path.join(MODELS_DIR, "04_interview_model.pkl")
OUTPUT_CATALOG_PATH = os.path.join(BASE_DIR, "interview_questions_catalog.json")

def main():
    print(f"Loading dataset from: {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Total rows loaded: {len(df):,}")

    # 1. Compile Comprehensive Question Bank per Role
    print("Compiling Comprehensive Question Bank across all 63 roles...")
    catalog = {}
    
    # Deduplicate questions per (job_title, question_text)
    df_dedup = df.drop_duplicates(subset=["job_title", "question_text"])
    print(f"Total unique questions compiled: {len(df_dedup):,}")

    for job_title, group in df_dedup.groupby("job_title"):
        industry = group["industry"].mode().values[0] if not group["industry"].empty else "General"
        
        questions_list = []
        for _, row in group.iterrows():
            eval_pts = [p.strip() for p in str(row.get("key_evaluation_points", "")).split(";") if p.strip()]
            questions_list.append({
                "question_text": str(row["question_text"]).strip(),
                "question_type": str(row["question_type"]).strip(),
                "difficulty_level": str(row["difficulty_level"]).strip(),
                "ideal_answer_length_words": int(row.get("ideal_answer_length_words", 200)),
                "key_evaluation_points": eval_pts,
            })

        # Summary statistics
        type_counts = group["question_type"].value_counts().to_dict()
        diff_counts = group["difficulty_level"].value_counts().to_dict()

        catalog[job_title] = {
            "job_title": job_title,
            "industry": industry,
            "total_questions_available": len(questions_list),
            "type_counts": type_counts,
            "diff_counts": diff_counts,
            "questions": questions_list
        }

    with open(OUTPUT_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"[SUCCESS] Saved Complete Interview Question Bank to: {OUTPUT_CATALOG_PATH} with {len(catalog)} roles!")

    # 2. Train Difficulty Classifier Model
    print("Training Interview Difficulty Classifier Model...")
    le_ind = LabelEncoder().fit(df["industry"].astype(str))
    le_job = LabelEncoder().fit(df["job_title"].astype(str))
    le_typ = LabelEncoder().fit(df["question_type"].astype(str))
    le_diff = LabelEncoder().fit(df["difficulty_level"].astype(str))

    df["eval_points_count"] = df["key_evaluation_points"].apply(lambda x: len(str(x).split(";")))
    df["question_length_words"] = df["question_text"].apply(lambda x: len(str(x).split()))
    df["ind_enc"] = le_ind.transform(df["industry"].astype(str))
    df["job_enc"] = le_job.transform(df["job_title"].astype(str))
    df["typ_enc"] = le_typ.transform(df["question_type"].astype(str))

    feature_cols = [
        "ideal_answer_length_words",
        "eval_points_count",
        "question_length_words",
        "ind_enc",
        "job_enc",
        "typ_enc",
    ]

    X = df[feature_cols]
    y = le_diff.transform(df["difficulty_level"].astype(str))

    model = HistGradientBoostingClassifier(max_iter=150, random_state=42)
    model.fit(X, y)

    acc = model.score(X, y)
    print(f"[SUCCESS] Model trained successfully! Accuracy: {acc*100:.2f}%")

    bundle = {
        "model": model,
        "features": feature_cols,
        "industry_classes": list(le_ind.classes_),
        "job_title_classes": list(le_job.classes_),
        "question_type_classes": list(le_typ.classes_),
        "difficulty_classes": list(le_diff.classes_),
        "question_catalog": catalog,
        "accuracy": acc,
        "training_samples": len(df)
    }

    joblib.dump(bundle, OUTPUT_MODEL_PATH)
    print(f"[SUCCESS] Saved complete Model Bundle to: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
