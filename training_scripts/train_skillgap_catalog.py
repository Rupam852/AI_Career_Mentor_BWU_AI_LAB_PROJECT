"""
train_skillgap_catalog.py
-------------------------
Processes 200,000 skill gap records to train the Estimated Transition Time Model
and compiles a canonical Required Skills & Resource Recommendation catalog for all 47 target career paths.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "AI_Career_Mentor_datasets", "02_skill_gap_analysis_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "AI_Career_Mentor_models")
OUTPUT_MODEL_PATH = os.path.join(MODELS_DIR, "02_skillgap_model.pkl")
OUTPUT_CATALOG_PATH = os.path.join(BASE_DIR, "skillgap_roles_catalog.json")

def main():
    print(f"Loading dataset from: {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Total rows loaded: {len(df):,}")

    # 1. Compile Required Skills & Learning Resources Catalog per Target Role
    print("Compiling Target Role Required Skills and Learning Resources Catalog...")
    catalog = {}

    for target_role, group in df.groupby("target_role"):
        industry = group["industry"].mode().values[0] if not group["industry"].empty else "General"
        
        # Extract most common required skills
        all_req_skills = []
        for s in group["required_skills_for_target"].dropna():
            all_req_skills.extend([item.strip() for item in str(s).split(",") if item.strip()])
        top_required_skills = [s for s, _ in Counter(all_req_skills).most_common(12)]

        # Extract resource recommendations per skill
        resource_map = {}
        for res_str in group["recommended_resources"].dropna().head(50):
            # Format: "Skill -> Resource; Skill2 -> Resource2"
            items = str(res_str).split(";")
            for it in items:
                if "->" in it:
                    sk, src = it.split("->", 1)
                    sk_clean = sk.strip()
                    src_clean = src.strip()
                    if sk_clean and src_clean and sk_clean not in resource_map:
                        resource_map[sk_clean] = src_clean

        catalog[target_role] = {
            "target_role": target_role,
            "industry": industry,
            "required_skills": top_required_skills,
            "resource_map": resource_map,
            "avg_gap_months": round(float(group["estimated_months_to_close_gap"].mean()), 1)
        }

    with open(OUTPUT_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"[SUCCESS] Saved Skill Gap Catalog to: {OUTPUT_CATALOG_PATH} with {len(catalog)} roles!")

    # 2. Train Transition Time Model
    print("Training Skill Gap Transition Time Regression Model...")
    le_ind = LabelEncoder().fit(df["industry"].astype(str))
    le_cur = LabelEncoder().fit(df["current_role"].astype(str))
    le_tar = LabelEncoder().fit(df["target_role"].astype(str))
    le_edu = LabelEncoder().fit(df["education_level"].astype(str))

    priority_map = {"Low": 0, "Medium": 1, "High": 2}

    df["curr_skills_count"] = df["current_skills"].apply(lambda x: len(str(x).split(",")))
    df["req_skills_count"] = df["required_skills_for_target"].apply(lambda x: len(str(x).split(",")))
    df["priority_enc"] = df["learning_priority"].map(priority_map).fillna(1)

    df["ind_enc"] = le_ind.transform(df["industry"].astype(str))
    df["cur_enc"] = le_cur.transform(df["current_role"].astype(str))
    df["tar_enc"] = le_tar.transform(df["target_role"].astype(str))
    df["edu_enc"] = le_edu.transform(df["education_level"].astype(str))

    feature_cols = [
        "skill_gap_count",
        "gap_score",
        "readiness_percentage",
        "curr_skills_count",
        "req_skills_count",
        "priority_enc",
        "ind_enc",
        "cur_enc",
        "tar_enc",
        "edu_enc",
    ]

    X = df[feature_cols]
    y = df["estimated_months_to_close_gap"]

    model = HistGradientBoostingRegressor(max_iter=150, random_state=42)
    model.fit(X, y)

    score = model.score(X, y)
    print(f"[SUCCESS] Model trained successfully! R2 Score: {score:.4f}")

    bundle = {
        "model": model,
        "features": feature_cols,
        "industry_classes": list(le_ind.classes_),
        "current_role_classes": list(le_cur.classes_),
        "target_role_classes": list(le_tar.classes_),
        "education_level_classes": list(le_edu.classes_),
        "skills_catalog": catalog,
        "r2_score": score,
        "training_samples": len(df)
    }

    joblib.dump(bundle, OUTPUT_MODEL_PATH)
    print(f"[SUCCESS] Saved complete Model Bundle to: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
