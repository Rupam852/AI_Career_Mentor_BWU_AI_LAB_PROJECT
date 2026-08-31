"""
train_roadmap_curriculum.py
----------------------------
Trains regression models for duration and compiles a rich, domain-specific 
Basic-to-Advanced multi-phase curriculum & milestone catalog from 200,000 roadmap dataset records.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "AI_Career_Mentor_datasets", "03_roadmap_generator_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "AI_Career_Mentor_models")
OUTPUT_MODEL_PATH = os.path.join(MODELS_DIR, "03_roadmap_model.pkl")
OUTPUT_CATALOG_PATH = os.path.join(BASE_DIR, "roadmap_curriculum_catalog.json")

def main():
    print(f"Loading dataset from: {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Total rows loaded: {len(df):,}")

    # 1. Compile Domain Curriculum Catalog per (target_role, industry)
    print("Compiling Basic-to-Advanced Curriculum & Milestone catalog...")
    catalog = {}

    for target_role, group in df.groupby("target_role"):
        industry = group["industry"].mode().values[0]
        avg_duration = round(float(group["total_duration_months"].mean()), 1)
        avg_phases = int(round(group["number_of_phases"].mean()))
        
        # Extract top certifications
        certs = group["target_certification"].dropna().tolist()
        top_certs = [c for c, _ in Counter(certs).most_common(3)]
        
        # Extract top focus skills
        all_skills = []
        for s in group["focus_skills"].dropna():
            all_skills.extend([item.strip() for item in str(s).split(",") if item.strip()])
        top_skills = [s for s, _ in Counter(all_skills).most_common(12)]

        # Extract structured roadmap steps
        step_templates = group["roadmap_steps"].dropna().head(10).tolist()
        milestone_templates = group["milestones"].dropna().head(10).tolist()

        # Parse phases from roadmap steps
        phases_list = []
        if step_templates:
            raw_steps = step_templates[0].split(" | ")
            for idx, raw_step in enumerate(raw_steps, 1):
                # e.g. Foundation Building (3 mo): Python via Official documentation
                parts = raw_step.split(":")
                phase_title = parts[0].strip() if len(parts) > 0 else f"Phase {idx}: Foundation"
                phase_desc = parts[1].strip() if len(parts) > 1 else "Core skills and foundational concepts"
                
                # Derive phase level
                if idx == 1:
                    level = "Beginner (Fundamentals & Syntax)"
                    badge = "🟢 Basic Fundamentals"
                elif idx == 2:
                    level = "Intermediate (Core Frameworks & Tools)"
                    badge = "🟡 Core Development"
                elif idx == 3:
                    level = "Upper-Intermediate (Real-world Architecture)"
                    badge = "🟠 Architecture & Systems"
                elif idx == 4:
                    level = "Advanced (Production & Optimization)"
                    badge = "🔴 Advanced Specialization"
                else:
                    level = "Mastery (Capstone Projects & Certification)"
                    badge = "🏆 Capstone & Job-Readiness"

                phases_list.append({
                    "phase_number": idx,
                    "title": phase_title,
                    "level": level,
                    "badge": badge,
                    "details": phase_desc,
                    "key_topics": [s for s in top_skills[((idx-1)*2):(idx*2+1)] if s] or top_skills[:3]
                })

        # Parse milestones
        milestones_list = []
        if milestone_templates:
            raw_ms = milestone_templates[0].split(" | ")
            for ms in raw_ms:
                milestones_list.append(ms.strip())

        catalog[target_role] = {
            "target_role": target_role,
            "industry": industry,
            "typical_duration_months": avg_duration,
            "default_phases_count": avg_phases,
            "top_skills": top_skills,
            "top_certifications": top_certs or ["Industry Standard Professional Certificate"],
            "phases": phases_list,
            "milestones": milestones_list,
            "recommended_projects": [
                f"Beginner {target_role} Starter Application (Clean architecture, CRUD, Unit testing)",
                f"Production-Ready Enterprise {target_role} Platform (Scalable microservices, CI/CD, Cloud deploy)",
                f"End-to-End Capstone Portfolio Demonstration & Case Study"
            ]
        }

    # Save compiled catalog
    with open(OUTPUT_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"[SUCCESS] Saved Curriculum Catalog to: {OUTPUT_CATALOG_PATH} with {len(catalog)} target roles!")

    # 2. Train HistGradientBoosting Regression Model for Duration
    print("Training Duration Prediction Model...")
    
    # Encoders
    le_ind = LabelEncoder().fit(df["industry"].astype(str))
    le_cur = LabelEncoder().fit(df["current_role"].astype(str))
    le_tar = LabelEncoder().fit(df["target_role"].astype(str))
    
    diff_map = {"Easy": 0, "Moderate": 1, "Challenging": 2, "Intensive": 3}
    
    df["focus_skills_count"] = df["focus_skills"].apply(lambda x: len(str(x).split(",")))
    df["difficulty_enc"] = df["difficulty_level"].map(diff_map).fillna(1)
    df["has_cert"] = df["target_certification"].notna().astype(int)
    df["ind_enc"] = le_ind.transform(df["industry"].astype(str))
    df["cur_enc"] = le_cur.transform(df["current_role"].astype(str))
    df["tar_enc"] = le_tar.transform(df["target_role"].astype(str))

    feature_cols = [
        "weekly_hours_commitment",
        "number_of_phases",
        "focus_skills_count",
        "difficulty_enc",
        "has_cert",
        "ind_enc",
        "cur_enc",
        "tar_enc",
    ]

    X = df[feature_cols]
    y = df["total_duration_months"]

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
        "curriculum_catalog": catalog,
        "r2_score": score,
        "training_samples": len(df)
    }

    joblib.dump(bundle, OUTPUT_MODEL_PATH)
    print(f"[SUCCESS] Saved complete Model Bundle to: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
