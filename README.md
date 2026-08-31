# 🚀 AI Career Mentor — Intelligence Suite

> **Production-Ready, Enterprise AI Career Mentorship Platform** powered by **8 Machine Learning Models** trained on **1.6+ Million Industry Career Records** across 63 Specialized Professions. Features a **Material 3 + Glassmorphism Web Application** (FastAPI) and an **Interactive Analytics Dashboard** (Streamlit).

---

## 📁 Project Architecture & Folder Structure

```
d:/AI_Career_Mentor/
│
├── AI_Career_Mentor_datasets/       # 📊 8 Raw Industry Datasets (200,000 Rows Each)
│   ├── 01_resume_analysis_dataset.csv
│   ├── 02_skill_gap_analysis_dataset.csv
│   ├── 03_roadmap_generator_dataset.csv
│   ├── 04_interview_questions_dataset.csv
│   ├── 05_linkedin_review_dataset.csv
│   ├── 06_github_review_dataset.csv
│   ├── 07_salary_prediction_dataset_enriched.csv
│   └── 08_career_recommendation_dataset.csv
│
├── AI_Career_Mentor_models/         # 🧠 8 Trained Machine Learning Model Bundles (.pkl) & Metrics
│   ├── 01_resume_ats_model.pkl              (HistGradientBoostingRegressor - ATS & Keyword Match)
│   ├── 02_skillgap_model.pkl                (HistGradientBoostingRegressor - Transition Timeline)
│   ├── 03_roadmap_model.pkl                 (HistGradientBoostingRegressor - 5-Phase Multi-Milestone Curriculum)
│   ├── 04_interview_model.pkl               (HistGradientBoostingClassifier - 9,300+ Question Bank & STAR Guide)
│   ├── 05_linkedin_model.pkl                (HistGradientBoostingClassifier - Profile Rating & Audit)
│   ├── 06_github_model.pkl                  (HistGradientBoostingClassifier - Open Source & Repo Quality)
│   ├── 07_salary_prediction_model.pkl       (HistGradientBoostingRegressor - PPP-Adjusted Global Salary)
│   ├── 08_career_recommendation_model.pkl   (HistGradientBoostingClassifier - Top 3 & Top 5 Career Shortlist)
│   └── *_metrics.json                       (Accuracy, R² Scores & Validation Metrics)
│
├── catalogs/                        # 📚 Structured Curated Catalogs & Knowledge Bases
│   ├── category_mappings.json               (Canonical dropdown lists & role mappings)
│   ├── roadmap_curriculum_catalog.json      (47-Role 5-Phase Basic-to-Advanced Roadmaps & Project Blueprints)
│   ├── interview_questions_catalog.json     (9,345 Unique Interview Questions with STAR Guidelines)
│   └── skillgap_roles_catalog.json          (47 Target Roles with Standard Required Skills & Course Resources)
│
├── training_scripts/                # ⚙️ Data Processing & Model Training Pipelines
│   ├── train_roadmap_curriculum.py          (Trains Roadmap Model & builds 5-Phase Curriculums)
│   ├── train_interview_catalog.py           (Compiles 9,300+ Role Questions & Trains Difficulty Classifier)
│   ├── train_skillgap_catalog.py            (Compiles Required Skills Benchmarks & Trains Transition Model)
│   └── extract_mappings.py                  (Extracts unique categories across all 8 datasets)
│
├── static/                          # 🎨 Modern Material 3 + Glassmorphism UI (Responsive)
│   ├── index.html                           (Single-page Web Application)
│   ├── css/
│   │   └── m3-glass.css                     (Custom Glassmorphism, Dark Mode, Animations & Mobile Drawer)
│   └── js/
│       └── app.js                           (REST Client, GitHub Live Auto-Fetch, Timeline & Question Renderers)
│
├── mentor_engine.py                 # ⚡ Core Python Inference Engine for all 8 ML Models
├── server.py                        # 🌐 FastAPI REST API Server (Port 8000)
├── app.py                           # 📊 Streamlit Interactive Analytics Dashboard (Port 8501)
├── country_data.py                  # 🌍 Global Economic Matrix, Currencies & Numbeo 2026 PPP Indices
├── career_utils.py                  # 🛠️ Encoding & Text Utility Helpers
├── requirements.txt                 # 📦 Project Dependencies
├── run_webapp.bat                   # 🚀 1-Click Launcher: FastAPI Web App (http://localhost:8000)
└── run_app.bat                      # 🚀 1-Click Launcher: Streamlit App (http://localhost:8501)
```

---

## ⚡ 8 AI Intelligence Modules Overview

1. **📄 Resume ATS & Keyword Scanner (`01_resume`)**:
   - Analyzes resume word count, keywords, certifications, and skills to predict ATS Pass Score & Category rating.
2. **🎯 Skill Gap & Career Readiness Analyzer (`02_skillgap`)**:
   - Automatically benchmarks user's current skills against target role standards, computes readiness score, missing skills count, transition time, and recommends specific courses.
3. **🗺️ Career Roadmap Duration & Curriculum Studio (`03_roadmap`)**:
   - Generates full 5-Phase Basic-to-Advanced learning paths (Phase 1 Fundamentals ➔ Phase 5 Capstone & Certifications) with weekly hours, milestones, and project blueprints.
4. **💡 AI Interview Question Generator & Prep Studio (`04_interview`)**:
   - Role-specific curated question bank (Top 25 High-Impact questions by default, expandable to 300+), categorized into Technical, Behavioral, and Situational questions with STAR answer frameworks.
5. **💼 LinkedIn Profile Reviewer (`05_linkedin`)**:
   - Evaluates profile completeness, connections network, summary word count, and engagement consistency.
6. **🐙 GitHub Portfolio Reviewer & Live Auto-Fetch (`06_github`)**:
   - Automatically fetches live public repositories, stars, commit streak, and languages via GitHub REST API with 1-click URL input.
7. **💵 Real Global Salary Predictor (`07_salary`)**:
   - Multi-currency compensation estimation with Numbeo 2026 Cost of Living, Purchasing Power Parity (PPP), and monthly breakdown across 10+ countries.
8. **🚀 AI Career Path Recommender (`08_career`)**:
   - Interactive 1-click passion pills & skill matching to recommend Top 3 & Top 5 career specializations with custom AI reasoning.

---

## 🚀 How to Run Locally

### 1. Launch FastAPI Web Application (Material 3 + Glassmorphism):
Double-click [`run_webapp.bat`](file:///d:/AI_Career_Mentor/run_webapp.bat) or run:
```bash
.\.venv\Scripts\uvicorn.exe server:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser.

### 2. Launch Streamlit Analytics Dashboard:
Double-click [`run_app.bat`](file:///d:/AI_Career_Mentor/run_app.bat) or run:
```bash
.\.venv\Scripts\streamlit.exe run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🛠️ Tech Stack
- **Backend & ML Inference**: Python 3.13, Scikit-Learn (HistGradientBoosting), Pandas, NumPy, Joblib
- **Web API Server**: FastAPI, Uvicorn, Pydantic
- **Web Frontend**: Vanilla HTML5, Material 3 Glassmorphism CSS, Vanilla ES6+ JavaScript, Google Material Symbols, Google Outfit Typography
- **Analytics Dashboard**: Streamlit, Plotly Express
