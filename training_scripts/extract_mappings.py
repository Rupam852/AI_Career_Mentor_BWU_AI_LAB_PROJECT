"""
extract_mappings.py
-------------------
Extracts all categorical values and exact LabelEncoder mappings from the 8 CSV datasets,
saving them to `category_mappings.json`.
"""
import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "AI_Career_Mentor_datasets"
MODELS_DIR = "AI_Career_Mentor_models"

mappings = {}

# 1. Resume Dataset
p1 = os.path.join(DATA_DIR, "01_resume_analysis_dataset.csv")
if os.path.exists(p1):
    print("Reading 01_resume_analysis_dataset.csv...")
    df = pd.read_csv(p1)
    mappings["01_resume"] = {
        "education_level": sorted(df["education_level"].dropna().unique().tolist()),
        "degree_field": sorted(df["degree_field"].dropna().unique().tolist()),
        "industry": sorted(df["industry"].dropna().unique().tolist()),
        "current_job_title": sorted(df["current_job_title"].dropna().unique().tolist()),
    }
    for col in ["education_level", "degree_field", "industry", "current_job_title"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        mappings["01_resume"][f"{col}_classes"] = le.classes_.tolist()

# 2. Skill Gap Dataset
p2 = os.path.join(DATA_DIR, "02_skill_gap_analysis_dataset.csv")
if os.path.exists(p2):
    print("Reading 02_skill_gap_analysis_dataset.csv...")
    df = pd.read_csv(p2)
    mappings["02_skillgap"] = {
        "industry": sorted(df["industry"].dropna().unique().tolist()),
        "current_role": sorted(df["current_role"].dropna().unique().tolist()),
        "target_role": sorted(df["target_role"].dropna().unique().tolist()),
        "education_level": sorted(df["education_level"].dropna().unique().tolist()),
        "learning_priority": ["Low", "Medium", "High"]
    }
    for col in ["industry", "current_role", "target_role", "education_level"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        mappings["02_skillgap"][f"{col}_classes"] = le.classes_.tolist()

# 3. Roadmap Dataset
p3 = os.path.join(DATA_DIR, "03_roadmap_generator_dataset.csv")
if os.path.exists(p3):
    print("Reading 03_roadmap_generator_dataset.csv...")
    df = pd.read_csv(p3)
    mappings["03_roadmap"] = {
        "industry": sorted(df["industry"].dropna().unique().tolist()),
        "current_role": sorted(df["current_role"].dropna().unique().tolist()),
        "target_role": sorted(df["target_role"].dropna().unique().tolist()),
        "difficulty_level": ["Easy", "Moderate", "Challenging", "Hard"]
    }
    for col in ["industry", "current_role", "target_role"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        mappings["03_roadmap"][f"{col}_classes"] = le.classes_.tolist()

# 4. Interview Dataset
p4 = os.path.join(DATA_DIR, "04_interview_questions_dataset.csv")
if os.path.exists(p4):
    print("Reading 04_interview_questions_dataset.csv...")
    df = pd.read_csv(p4)
    mappings["04_interview"] = {
        "industry": sorted(df["industry"].dropna().unique().tolist()),
        "job_title": sorted(df["job_title"].dropna().unique().tolist()),
        "question_type": sorted(df["question_type"].dropna().unique().tolist())
    }
    for col in ["industry", "job_title", "question_type"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        mappings["04_interview"][f"{col}_classes"] = le.classes_.tolist()

# 5. LinkedIn Dataset
p5 = os.path.join(DATA_DIR, "05_linkedin_review_dataset.csv")
if os.path.exists(p5):
    print("Reading 05_linkedin_review_dataset.csv...")
    df = pd.read_csv(p5)
    mappings["05_linkedin"] = {
        "industry": sorted(df["industry"].dropna().unique().tolist()),
        "current_job_title": sorted(df["current_job_title"].dropna().unique().tolist()),
        "review_rating": ["Poor", "Needs Improvement", "Average", "Good", "Excellent"]
    }
    for col in ["industry", "current_job_title"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        mappings["05_linkedin"][f"{col}_classes"] = le.classes_.tolist()

# 6. GitHub Dataset
p6 = os.path.join(DATA_DIR, "06_github_review_dataset.csv")
if os.path.exists(p6):
    print("Reading 06_github_review_dataset.csv...")
    df = pd.read_csv(p6)
    mappings["06_github"] = {
        "focus_area": sorted(df["focus_area"].dropna().unique().tolist()),
        "review_rating": ["Poor", "Needs Improvement", "Average", "Good", "Excellent"]
    }
    le = LabelEncoder()
    le.fit(df["focus_area"].astype(str))
    mappings["06_github"]["focus_area_classes"] = le.classes_.tolist()

# 7. Salary Dataset
p7 = os.path.join(DATA_DIR, "07_salary_prediction_dataset_enriched.csv")
if not os.path.exists(p7):
    p7 = os.path.join(MODELS_DIR, "07_salary_prediction_dataset_enriched.csv")
if os.path.exists(p7):
    print("Reading 07_salary_prediction_dataset_enriched.csv...")
    df = pd.read_csv(p7)
    mappings["07_salary"] = {
        "industry": sorted(df["industry"].dropna().unique().tolist()),
        "job_title": sorted(df["job_title"].dropna().unique().tolist()),
        "education_level": sorted(df["education_level"].dropna().unique().tolist()),
        "degree_field": sorted(df["degree_field"].dropna().unique().tolist()),
        "country": sorted(df["country"].dropna().unique().tolist()),
        "company_size": sorted(df["company_size"].dropna().unique().tolist()),
        "work_type": sorted(df["work_type"].dropna().unique().tolist()),
    }
    for col in ["industry", "job_title", "education_level", "degree_field", "country", "company_size", "work_type"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        mappings["07_salary"][f"{col}_classes"] = le.classes_.tolist()

# 8. Career Recommendation Dataset
p8 = os.path.join(DATA_DIR, "08_career_recommendation_dataset.csv")
if os.path.exists(p8):
    print("Reading 08_career_recommendation_dataset.csv...")
    df = pd.read_csv(p8)
    mappings["08_career"] = {
        "interest_categories": [
            'Business & Strategy', 'Communication & Media', 'Data & Analytics', 'Design & Creativity',
            'Healthcare & Wellbeing', 'People & Culture', 'Problem Solving', 'Technology'
        ],
        "work_style": sorted(df["work_style"].dropna().unique().tolist()),
        "recommended_industry": sorted(df["recommended_industry"].dropna().unique().tolist()),
        "education_level": sorted(df["education_level"].dropna().unique().tolist()),
        "recommended_career": sorted(df["recommended_career"].dropna().unique().tolist()),
    }
    for col in ["work_style", "recommended_industry", "education_level"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        mappings["08_career"][f"{col}_classes"] = le.classes_.tolist()

with open("category_mappings.json", "w") as f:
    json.dump(mappings, f, indent=2)

print("Saved category_mappings.json successfully!")
