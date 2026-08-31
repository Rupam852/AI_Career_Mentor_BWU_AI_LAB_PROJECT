"""
mentor_engine.py
----------------
Unified inference engine for all 8 AI Career Mentor machine learning models.
Handles model loading, scikit-learn version compatibility, input feature transformation,
and economic/currency conversions.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Compatibility shim for scikit-learn HistGradientBoosting models across versions
try:
    import sklearn._loss
    if '_loss' not in sys.modules:
        sys.modules['_loss'] = sklearn._loss._loss if hasattr(sklearn._loss, '_loss') else sklearn._loss
except Exception as e:
    print(f"Warning setting _loss shim: {e}")

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "AI_Career_Mentor_models")
CATALOGS_DIR = os.path.join(BASE_DIR, "catalogs")
MAPPINGS_FILE = os.path.join(CATALOGS_DIR, "category_mappings.json") if os.path.exists(os.path.join(CATALOGS_DIR, "category_mappings.json")) else os.path.join(BASE_DIR, "category_mappings.json")

# Add models dir to sys.path so country_data and career_utils can be imported
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from country_data import (
    COUNTRY_ECONOMIC_DATA,
    EXCHANGE_RATE_TO_USD,
    CURRENCY_SYMBOL,
    get_country_info,
    to_local_currency,
    format_local_currency,
)
from career_utils import ORDINAL_MAPS, count_items, ordinal_encode

class CareerMentorEngine:
    _instance = None

    def __init__(self):
        self.models = {}
        self.mappings = {}
        self._load_mappings()
        self._load_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CareerMentorEngine()
        return cls._instance

    def _load_mappings(self):
        if os.path.exists(MAPPINGS_FILE):
            with open(MAPPINGS_FILE, "r") as f:
                self.mappings = json.load(f)
        else:
            print("Warning: category_mappings.json not found! Running with empty mappings.")
            self.mappings = {}

    def _load_models(self):
        model_files = {
            "resume_ats": "01_resume_ats_model.pkl",
            "skillgap": "02_skillgap_model.pkl",
            "roadmap": "03_roadmap_model.pkl",
            "interview": "04_interview_model.pkl",
            "linkedin": "05_linkedin_model.pkl",
            "github": "06_github_model.pkl",
            "salary": "07_salary_prediction_model.pkl",
            "career_rec": "08_career_recommendation_model.pkl",
        }
        for key, fname in model_files.items():
            path = os.path.join(MODELS_DIR, fname)
            if os.path.exists(path):
                try:
                    self.models[key] = joblib.load(path)
                except Exception as ex:
                    print(f"Error loading model {fname}: {ex}")
            else:
                print(f"Model file {path} not found.")

    def _encode_val(self, val, classes_list):
        if not classes_list:
            return -1
        val_str = str(val).strip()
        if val_str in classes_list:
            return classes_list.index(val_str)
        # Case-insensitive fallback
        lower_classes = [c.lower() for c in classes_list]
        if val_str.lower() in lower_classes:
            return lower_classes.index(val_str.lower())
        # Smart mapping for Student / Fresher / Intern / Entry Level
        val_lower = val_str.lower()
        if any(k in val_lower for k in ["student", "fresher", "intern", "graduate", "beginner", "entry"]):
            for role_candidate in ["junior engineer", "teaching assistant", "software engineer", "junior developer", "data analyst", "associate"]:
                if role_candidate in lower_classes:
                    return lower_classes.index(role_candidate)
            return 0  # Fallback to the first baseline class
        return 0  # Default safe index if unseen

    # =========================================================================
    # 1. Resume ATS Score Predictor
    # =========================================================================
    def predict_resume_ats(
        self,
        age: int,
        years_experience: float,
        word_count: int,
        keyword_match_percentage: float,
        skills_text: str,
        certifications_text: str,
        missing_keywords_text: str,
        overall_rating: str,
        education_level: str,
        degree_field: str,
        industry: str,
        current_job_title: str,
    ) -> dict:
        bundle = self.models.get("resume_ats")
        if not bundle:
            return {"error": "Model 01_resume_ats_model.pkl not loaded"}

        m1_map = self.mappings.get("01_resume", {})
        
        # Feature Engineering
        skills_count = len([s for s in skills_text.split(",") if s.strip()])
        certs_count = len([s for s in certifications_text.split(",") if s.strip()])
        missing_kw_count = len([s for s in missing_keywords_text.split(",") if s.strip()])
        
        overall_rating_enc = ORDINAL_MAPS["overall_rating"].get(overall_rating, 2)
        education_level_enc = self._encode_val(education_level, m1_map.get("education_level_classes", []))
        degree_field_enc = self._encode_val(degree_field, m1_map.get("degree_field_classes", []))
        industry_enc = self._encode_val(industry, m1_map.get("industry_classes", []))
        current_job_title_enc = self._encode_val(current_job_title, m1_map.get("current_job_title_classes", []))

        features = [
            age,
            years_experience,
            word_count,
            keyword_match_percentage,
            skills_count,
            certs_count,
            missing_kw_count,
            overall_rating_enc,
            education_level_enc,
            degree_field_enc,
            industry_enc,
            current_job_title_enc,
        ]

        df_feat = pd.DataFrame([features], columns=bundle["features"])
        score = float(bundle["model"].predict(df_feat)[0])
        score = max(0.0, min(100.0, score))

        if score >= 80:
            category = "Excellent (Interview Ready)"
            badge_color = "green"
        elif score >= 65:
            category = "Good (Minor Improvements Needed)"
            badge_color = "blue"
        elif score >= 50:
            category = "Average (Requires Optimization)"
            badge_color = "orange"
        else:
            category = "Needs Major Revision"
            badge_color = "red"

        return {
            "ats_score": round(score, 1),
            "category": category,
            "badge_color": badge_color,
            "skills_detected": skills_count,
            "certifications_detected": certs_count,
            "missing_keywords_count": missing_kw_count,
            "word_count": word_count,
            "keyword_match_percentage": keyword_match_percentage,
        }

    # =========================================================================
    # 2. Skill Gap Predictor
    # =========================================================================
    def predict_skill_gap(
        self,
        current_skills: str,
        industry: str,
        current_role: str,
        target_role: str,
        education_level: str,
        learning_priority: str = "Medium",
        required_skills: str = None,
        skill_gap_count: int = None,
        gap_score: float = None,
        readiness_percentage: float = None,
    ) -> dict:
        bundle = self.models.get("skillgap")
        if not bundle:
            return {"error": "Model 02_skillgap_model.pkl not loaded"}

        catalog = bundle.get("skills_catalog", {})
        target_info = catalog.get(target_role)
        if not target_info:
            for k in catalog:
                if target_role.lower() in k.lower() or k.lower() in target_role.lower():
                    target_info = catalog[k]
                    break

        # Standard required benchmark skills
        benchmark_req_list = target_info.get("required_skills", []) if target_info else []
        resource_map = target_info.get("resource_map", {}) if target_info else {}

        # Parse user's current skills
        user_skills_list = [s.strip() for s in current_skills.replace(";", ",").split(",") if s.strip()]
        user_skills_lower = {s.lower() for s in user_skills_list}

        if required_skills and required_skills.strip():
            req_list = [s.strip() for s in required_skills.replace(";", ",").split(",") if s.strip()]
        else:
            req_list = benchmark_req_list if benchmark_req_list else [
                "Problem Solving", "System Design", "Cloud Infrastructure", "API Development", "Version Control", "Database Optimization"
            ]

        # Compute matched vs missing skills
        matched_skills = []
        missing_skills = []
        for req in req_list:
            if req.lower() in user_skills_lower or any(req.lower() in u or u in req.lower() for u in user_skills_lower):
                matched_skills.append(req)
            else:
                missing_skills.append(req)

        # Auto-compute counts if not manually provided
        if skill_gap_count is None:
            skill_gap_count = len(missing_skills)

        total_req_count = max(len(req_list), 1)
        if readiness_percentage is None:
            readiness_percentage = round((len(matched_skills) / total_req_count) * 100, 1)

        if gap_score is None:
            gap_score = round((len(missing_skills) / total_req_count) * 100, 1)

        curr_skills_count = len(user_skills_list)
        req_skills_count = len(req_list)
        learning_priority_enc = ORDINAL_MAPS["learning_priority"].get(learning_priority, 1)

        m2_map = self.mappings.get("02_skillgap", {})
        industry_enc = self._encode_val(industry, m2_map.get("industry_classes", []))
        current_role_enc = self._encode_val(current_role, m2_map.get("current_role_classes", []))
        target_role_enc = self._encode_val(target_role, m2_map.get("target_role_classes", []))
        education_level_enc = self._encode_val(education_level, m2_map.get("education_level_classes", []))

        features = [
            skill_gap_count,
            gap_score,
            readiness_percentage,
            curr_skills_count,
            req_skills_count,
            learning_priority_enc,
            industry_enc,
            current_role_enc,
            target_role_enc,
            education_level_enc,
        ]

        df_feat = pd.DataFrame([features], columns=bundle["features"])
        months = float(bundle["model"].predict(df_feat)[0])
        months = max(0.5, round(months, 1))

        # Build missing skills with recommended resources
        missing_with_resources = []
        for sk in missing_skills:
            res = resource_map.get(sk)
            if not res:
                # Default learning pathways
                res = f"Comprehensive {sk} Masterclass & Hands-on Project"
            missing_with_resources.append({"skill": sk, "resource": res})

        return {
            "estimated_months": months,
            "readiness_percentage": readiness_percentage,
            "gap_score": gap_score,
            "skill_gap_count": skill_gap_count,
            "current_skills_count": curr_skills_count,
            "required_skills_count": req_skills_count,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "missing_skills_with_resources": missing_with_resources,
            "benchmark_required_skills": req_list,
            "target_role": target_role,
            "current_role": current_role,
        }

    # =========================================================================
    # 3. Roadmap Duration Predictor
    # =========================================================================
    def predict_roadmap_duration(
        self,
        weekly_hours_commitment: float,
        number_of_phases: int,
        focus_skills: str,
        difficulty_level: str,
        has_target_cert: bool,
        industry: str,
        current_role: str,
        target_role: str,
    ) -> dict:
        bundle = self.models.get("roadmap")
        if not bundle:
            return {"error": "Model 03_roadmap_model.pkl not loaded"}

        m3_map = self.mappings.get("03_roadmap", {})
        focus_skills_count = len([s for s in focus_skills.split(",") if s.strip()])
        difficulty_level_enc = ORDINAL_MAPS["difficulty_level"].get(difficulty_level, 1)
        has_target_cert_int = 1 if has_target_cert else 0

        industry_enc = self._encode_val(industry, m3_map.get("industry_classes", []))
        current_role_enc = self._encode_val(current_role, m3_map.get("current_role_classes", []))
        target_role_enc = self._encode_val(target_role, m3_map.get("target_role_classes", []))

        features = [
            weekly_hours_commitment,
            number_of_phases,
            focus_skills_count,
            difficulty_level_enc,
            has_target_cert_int,
            industry_enc,
            current_role_enc,
            target_role_enc,
        ]

        df_feat = pd.DataFrame([features], columns=bundle["features"])
        total_months = float(bundle["model"].predict(df_feat)[0])
        total_months = max(1.0, round(total_months, 1))

        weeks_total = round(total_months * 4.33, 1)
        hours_total = round(weekly_hours_commitment * weeks_total)

        # Retrieve Basic-to-Advanced Curriculum from Trained Knowledge Catalog
        catalog = bundle.get("curriculum_catalog", {})
        target_info = catalog.get(target_role)
        if not target_info and catalog:
            # Fallback to closest match in catalog
            target_info = list(catalog.values())[0]

        phases = []
        if target_info and "phases" in target_info:
            phases = target_info["phases"][:number_of_phases] if number_of_phases else target_info["phases"]
        
        milestones = target_info.get("milestones", []) if target_info else []
        top_certifications = target_info.get("top_certifications", []) if target_info else []
        recommended_projects = target_info.get("recommended_projects", []) if target_info else []
        top_skills = target_info.get("top_skills", []) if target_info else []

        return {
            "total_duration_months": total_months,
            "total_weeks": weeks_total,
            "total_study_hours": hours_total,
            "number_of_phases": number_of_phases,
            "focus_skills_count": focus_skills_count,
            "target_role": target_role,
            "industry": industry,
            "current_role": current_role,
            "phases": phases,
            "milestones": milestones,
            "top_certifications": top_certifications,
            "recommended_projects": recommended_projects,
            "top_skills": top_skills,
        }

    # =========================================================================
    # 4. Interview Question Generator & Comprehensive Question Bank
    # =========================================================================
    def get_interview_question_bank(
        self,
        job_title: str,
        industry: str = None,
        question_type: str = "All",
        difficulty_level: str = "All",
        limit: int = 25,
        search_query: str = None,
    ) -> dict:
        bundle = self.models.get("interview")
        catalog = bundle.get("question_catalog", {}) if bundle else {}
        
        # Match job title in catalog
        role_data = catalog.get(job_title)
        if not role_data:
            # Fallback to closest match
            for k in catalog:
                if job_title.lower() in k.lower() or k.lower() in job_title.lower():
                    role_data = catalog[k]
                    break
        if not role_data and catalog:
            role_data = list(catalog.values())[0]

        if not role_data:
            return {"error": f"No question bank found for {job_title}", "questions": []}

        all_questions = role_data.get("questions", [])
        
        # Balance ranking: ensure Technical, Behavioral, and Situational are all represented in top results
        tech_q = [q for q in all_questions if q["question_type"] == "Technical"]
        behav_q = [q for q in all_questions if q["question_type"] == "Behavioral"]
        sit_q = [q for q in all_questions if q["question_type"] == "Situational"]

        # Interleave for balanced top curriculum: 3 Technical, 1 Behavioral, 1 Situational
        balanced_all = []
        max_len = max(len(tech_q), len(behav_q), len(sit_q), 1)
        t_idx, b_idx, s_idx = 0, 0, 0
        for _ in range(max_len):
            for _ in range(3):
                if t_idx < len(tech_q):
                    balanced_all.append(tech_q[t_idx])
                    t_idx += 1
            if b_idx < len(behav_q):
                balanced_all.append(behav_q[b_idx])
                b_idx += 1
            if s_idx < len(sit_q):
                balanced_all.append(sit_q[s_idx])
                s_idx += 1

        filtered = []
        for q in balanced_all:
            # Question Type filter
            if question_type and question_type != "All":
                if q["question_type"].lower() != question_type.lower():
                    continue
            # Difficulty filter
            if difficulty_level and difficulty_level != "All":
                if q["difficulty_level"].lower() != difficulty_level.lower():
                    continue
            # Search filter
            if search_query:
                sq = search_query.lower()
                if sq not in q["question_text"].lower() and not any(sq in p.lower() for p in q.get("key_evaluation_points", [])):
                    continue
            filtered.append(q)

        # Slice limit if requested (default top 25)
        if limit and limit > 0:
            display_questions = filtered[:limit]
        else:
            display_questions = filtered

        return {
            "job_title": role_data.get("job_title", job_title),
            "industry": role_data.get("industry", industry),
            "total_questions_in_bank": role_data.get("total_questions_available", len(all_questions)),
            "filtered_count": len(filtered),
            "display_count": len(display_questions),
            "limit_applied": limit,
            "type_breakdown": role_data.get("type_counts", {}),
            "difficulty_breakdown": role_data.get("diff_counts", {}),
            "questions": display_questions,
        }

    def predict_interview_difficulty(
        self,
        ideal_answer_length_words: int,
        eval_points_text: str,
        question_text: str,
        industry: str,
        job_title: str,
        question_type: str,
    ) -> dict:
        bundle = self.models.get("interview")
        if not bundle:
            return {"error": "Model 04_interview_model.pkl not loaded"}

        m4_map = self.mappings.get("04_interview", {})
        eval_points_count = len([s for s in eval_points_text.replace(",", ";").split(";") if s.strip()])
        question_text_len = len(question_text.strip().split())

        industry_enc = self._encode_val(industry, m4_map.get("industry_classes", []))
        job_title_enc = self._encode_val(job_title, m4_map.get("job_title_classes", []))
        question_type_enc = self._encode_val(question_type, m4_map.get("question_type_classes", []))

        features = [
            ideal_answer_length_words,
            eval_points_count,
            question_text_len,
            industry_enc,
            job_title_enc,
            question_type_enc,
        ]

        df_feat = pd.DataFrame([features], columns=bundle["features"])
        pred_label = bundle["model"].predict(df_feat)[0]
        
        probas = {}
        if hasattr(bundle["model"], "predict_proba"):
            p = bundle["model"].predict_proba(df_feat)[0]
            classes = bundle["model"].classes_
            for cls, prob in zip(classes, p):
                probas[str(cls)] = round(float(prob) * 100, 1)

        return {
            "predicted_difficulty": str(pred_label),
            "class_probabilities": probas,
            "question_word_count": question_text_len,
            "eval_points_count": eval_points_count,
        }

    # =========================================================================
    # 5. LinkedIn Profile Rating
    # =========================================================================
    def predict_linkedin_rating(
        self,
        has_profile_photo: bool,
        has_banner_image: bool,
        summary_word_count: int,
        connections_count: int,
        skills_count: int,
        total_endorsements: int,
        recommendations_count: int,
        posts_last_90_days: int,
        avg_engagement_per_post: float,
        profile_completeness_score: float,
        industry: str,
        current_job_title: str,
    ) -> dict:
        bundle = self.models.get("linkedin")
        if not bundle:
            return {"error": "Model 05_linkedin_model.pkl not loaded"}

        m5_map = self.mappings.get("05_linkedin", {})
        industry_enc = self._encode_val(industry, m5_map.get("industry_classes", []))
        current_job_title_enc = self._encode_val(current_job_title, m5_map.get("current_job_title_classes", []))

        features = [
            1 if has_profile_photo else 0,
            1 if has_banner_image else 0,
            summary_word_count,
            connections_count,
            skills_count,
            total_endorsements,
            recommendations_count,
            posts_last_90_days,
            avg_engagement_per_post,
            profile_completeness_score,
            industry_enc,
            current_job_title_enc,
        ]

        df_feat = pd.DataFrame([features], columns=bundle["features"])
        pred_rating = str(bundle["model"].predict(df_feat)[0])

        probas = {}
        if hasattr(bundle["model"], "predict_proba"):
            p = bundle["model"].predict_proba(df_feat)[0]
            for cls, prob in zip(bundle["model"].classes_, p):
                probas[str(cls)] = round(float(prob) * 100, 1)

        return {
            "predicted_rating": pred_rating,
            "rating_probabilities": probas,
            "completeness_score": profile_completeness_score,
        }

    # =========================================================================
    # 6. GitHub Profile Rating
    # =========================================================================
    def predict_github_rating(
        self,
        public_repos: int,
        followers: int,
        following: int,
        total_stars: int,
        total_forks: int,
        contributions_last_year: int,
        longest_streak_days: int,
        readme_coverage_percentage: float,
        pinned_repos_count: int,
        top_repo_stars: int,
        has_bio: bool,
        open_source_contributions: int,
        profile_score: float,
        languages_used_text: str,
        focus_area: str,
    ) -> dict:
        bundle = self.models.get("github")
        if not bundle:
            return {"error": "Model 06_github_model.pkl not loaded"}

        m6_map = self.mappings.get("06_github", {})
        languages_used_count = len([s for s in languages_used_text.split(",") if s.strip()])
        focus_area_enc = self._encode_val(focus_area, m6_map.get("focus_area_classes", []))

        features = [
            public_repos,
            followers,
            following,
            total_stars,
            total_forks,
            contributions_last_year,
            longest_streak_days,
            readme_coverage_percentage,
            pinned_repos_count,
            top_repo_stars,
            1 if has_bio else 0,
            open_source_contributions,
            profile_score,
            languages_used_count,
            focus_area_enc,
        ]

        df_feat = pd.DataFrame([features], columns=bundle["features"])
        pred_rating = str(bundle["model"].predict(df_feat)[0])

        probas = {}
        if hasattr(bundle["model"], "predict_proba"):
            p = bundle["model"].predict_proba(df_feat)[0]
            for cls, prob in zip(bundle["model"].classes_, p):
                probas[str(cls)] = round(float(prob) * 100, 1)

        return {
            "predicted_rating": pred_rating,
            "rating_probabilities": probas,
            "profile_score": profile_score,
            "total_stars": total_stars,
            "contributions_last_year": contributions_last_year,
        }

    # =========================================================================
    # GitHub Real-time API Fetcher
    # =========================================================================
    def fetch_github_profile(self, profile_input: str) -> dict:
        """
        Fetches live profile metrics and repository data from GitHub's public REST API.
        Extracts features, determines primary focus area, and executes ML prediction.
        """
        import urllib.request
        import json
        from collections import Counter

        # Extract clean username from input (URL or raw username)
        cleaned = profile_input.strip().rstrip("/")
        if "/" in cleaned:
            username = cleaned.split("/")[-1]
        else:
            username = cleaned
        username = username.lstrip("@").strip()

        if not username:
            return {"error": "Please provide a valid GitHub username or profile URL."}

        headers = {
            "User-Agent": "AI-Career-Mentor-Suite-Engine",
            "Accept": "application/vnd.github.v3+json"
        }

        try:
            # 1. Fetch user core profile
            user_req = urllib.request.Request(f"https://api.github.com/users/{username}", headers=headers)
            with urllib.request.urlopen(user_req, timeout=10) as resp:
                user_data = json.loads(resp.read().decode())

            # 2. Fetch public repos (up to 100 recent)
            repos_req = urllib.request.Request(
                f"https://api.github.com/users/{username}/repos?per_page=100&sort=pushed",
                headers=headers
            )
            with urllib.request.urlopen(repos_req, timeout=10) as resp:
                repos_data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"error": f"GitHub user '{username}' was not found."}
            elif e.code == 403:
                return {"error": "GitHub API rate limit reached. Please wait a minute or enter details manually."}
            return {"error": f"GitHub API returned error {e.code}: {e.reason}"}
        except Exception as e:
            return {"error": f"Failed to connect to GitHub: {str(e)}"}

        # Extract features
        public_repos = user_data.get("public_repos", 0)
        followers = user_data.get("followers", 0)
        following = user_data.get("following", 0)
        bio = user_data.get("bio") or ""
        has_bio = bool(bio.strip())
        avatar_url = user_data.get("avatar_url", "")
        display_name = user_data.get("name") or username
        html_url = user_data.get("html_url", f"https://github.com/{username}")
        company = user_data.get("company") or ""
        location = user_data.get("location") or ""

        total_stars = sum(r.get("stargazers_count", 0) for r in repos_data)
        total_forks = sum(r.get("forks_count", 0) for r in repos_data)
        top_repo_stars = max([r.get("stargazers_count", 0) for r in repos_data], default=0)

        # Languages breakdown
        lang_list = [r.get("language") for r in repos_data if r.get("language")]
        lang_counts = Counter(lang_list)
        top_languages = [lang for lang, _ in lang_counts.most_common(6)]
        languages_used_text = ", ".join(top_languages) if top_languages else "Python, JavaScript"

        # Determine Primary Focus Area matching exact category_mappings classes
        focus_area = "Web Development"
        lang_set = set([l.lower() for l in top_languages])
        if any(l in lang_set for l in ["python", "r", "julia", "jupyter notebook"]):
            focus_area = "Data Science / ML"
        elif any(l in lang_set for l in ["dockerfile", "hcl", "shell", "yaml", "go", "terraform"]):
            focus_area = "DevOps / Infrastructure"
        elif any(l in lang_set for l in ["kotlin", "swift", "dart", "flutter", "objective-c"]):
            focus_area = "Mobile Development"
        elif any(l in lang_set for l in ["c", "c++", "rust", "assembly", "verilog"]):
            focus_area = "Embedded Systems"
        elif any(l in lang_set for l in ["c#", "gdscript", "lua", "shaderlab"]):
            focus_area = "Game Development"
        elif any(l in lang_set for l in ["java", "scala", "elixir", "clojure"]):
            focus_area = "Backend Systems"
        elif any(l in lang_set for l in ["javascript", "typescript", "html", "css", "php", "ruby"]):
            focus_area = "Web Development"
        else:
            focus_area = "Web Development"

        # Quality heuristics
        repos_with_readme = sum(1 for r in repos_data if (r.get("size", 0) > 10 and not r.get("fork", False)))
        total_sample = max(1, len(repos_data))
        readme_coverage = min(100.0, max(40.0, round((repos_with_readme / total_sample) * 100, 1)))

        pinned_repos_count = min(6, max(1, sum(1 for r in repos_data if r.get("stargazers_count", 0) > 0 or not r.get("fork", False))))
        
        # Estimate streak and contributions from repo activity & account age
        contributions_last_year = min(3000, max(30, public_repos * 18 + total_stars * 3 + followers * 2))
        longest_streak = min(365, max(5, int(contributions_last_year / 12)))
        
        # Estimate open source contributions (forked repos or external collaboration)
        forked_repos = sum(1 for r in repos_data if r.get("fork", False))
        open_source_contribs = max(1, forked_repos + int(followers / 15))

        # Overall profile score (0-100)
        profile_score = min(100.0, max(30.0, round(
            (25 if has_bio else 10) +
            min(25, public_repos * 2) +
            min(25, total_stars * 1.5) +
            min(15, followers * 0.5) +
            (10 if len(top_languages) >= 2 else 5),
            1
        )))

        # Run ML Model Evaluation
        ml_eval = self.predict_github_rating(
            public_repos=public_repos,
            followers=followers,
            following=following,
            total_stars=total_stars,
            total_forks=total_forks,
            contributions_last_year=contributions_last_year,
            longest_streak_days=longest_streak,
            readme_coverage_percentage=readme_coverage,
            pinned_repos_count=pinned_repos_count,
            top_repo_stars=top_repo_stars,
            has_bio=has_bio,
            open_source_contributions=open_source_contribs,
            profile_score=profile_score,
            languages_used_text=languages_used_text,
            focus_area=focus_area,
        )

        return {
            "success": True,
            "username": username,
            "display_name": display_name,
            "avatar_url": avatar_url,
            "bio": bio,
            "company": company,
            "location": location,
            "html_url": html_url,
            "metrics": {
                "public_repos": public_repos,
                "followers": followers,
                "following": following,
                "total_stars": total_stars,
                "total_forks": total_forks,
                "top_repo_stars": top_repo_stars,
                "languages_used": languages_used_text,
                "top_languages": top_languages,
                "focus_area": focus_area,
                "readme_coverage_percentage": readme_coverage,
                "pinned_repos_count": pinned_repos_count,
                "has_bio": has_bio,
                "contributions_last_year": contributions_last_year,
                "longest_streak_days": longest_streak,
                "open_source_contributions": open_source_contribs,
                "profile_score": profile_score,
            },
            "ml_evaluation": ml_eval,
        }

    # =========================================================================
    # 7. Global Salary Predictor (Real-Data Enriched & Multi-Currency)
    # =========================================================================
    def predict_salary(
        self,
        years_experience: float,
        skills_count: int,
        certifications_count: int,
        industry: str,
        job_title: str,
        education_level: str,
        degree_field: str,
        country: str,
        company_size: str,
        work_type: str,
    ) -> dict:
        bundle = self.models.get("salary")
        if not bundle:
            return {"error": "Model 07_salary_prediction_model.pkl not loaded"}

        m7_map = self.mappings.get("07_salary", {})
        c_info = get_country_info(country)
        col_index = c_info["cost_of_living_index"]
        pp_index = c_info["purchasing_power_index"]

        industry_enc = self._encode_val(industry, m7_map.get("industry_classes", []))
        job_title_enc = self._encode_val(job_title, m7_map.get("job_title_classes", []))
        education_level_enc = self._encode_val(education_level, m7_map.get("education_level_classes", []))
        degree_field_enc = self._encode_val(degree_field, m7_map.get("degree_field_classes", []))
        country_enc = self._encode_val(country, m7_map.get("country_classes", []))
        company_size_enc = self._encode_val(company_size, m7_map.get("company_size_classes", []))
        work_type_enc = self._encode_val(work_type, m7_map.get("work_type_classes", []))

        features = [
            years_experience,
            skills_count,
            certifications_count,
            industry_enc,
            job_title_enc,
            education_level_enc,
            degree_field_enc,
            country_enc,
            company_size_enc,
            work_type_enc,
            col_index,
            pp_index,
        ]

        df_feat = pd.DataFrame([features], columns=bundle["features"])
        predicted_usd = float(bundle["model"].predict(df_feat)[0])
        predicted_usd = max(5000.0, round(predicted_usd, 2))

        local_amount = to_local_currency(predicted_usd, country)
        currency_code = c_info["currency_code"]
        currency_sym = CURRENCY_SYMBOL.get(currency_code, currency_code)
        formatted_local = format_local_currency(predicted_usd, country)

        us_lpp = COUNTRY_ECONOMIC_DATA["United States"]["purchasing_power_index"]
        ppp_adjusted_usd = round(predicted_usd * (pp_index / us_lpp), 2)

        # Monthly figures
        monthly_usd = round(predicted_usd / 12, 2)
        monthly_local = round(local_amount / 12, 2)

        return {
            "predicted_salary_usd": predicted_usd,
            "predicted_salary_local": local_amount,
            "currency_code": currency_code,
            "currency_symbol": currency_sym,
            "formatted_local_salary": formatted_local,
            "monthly_salary_usd": monthly_usd,
            "monthly_salary_local": monthly_local,
            "ppp_adjusted_salary_usd": ppp_adjusted_usd,
            "cost_of_living_index": col_index,
            "purchasing_power_index": pp_index,
            "country": country,
        }

    # =========================================================================
    # 8. Career Recommendation (Top-3 / Top-5 Shortlist Recommender)
    # =========================================================================
    def predict_career_recommendations(
        self,
        years_experience: float = 1.0,
        match_score: float = None,
        current_skills_text: str = "Python, Problem Solving",
        work_style: str = "Hybrid",
        recommended_industry: str = "Technology",
        education_level: str = "Bachelor's Degree",
        selected_interests: list = None,
        top_k: int = 5,
    ) -> dict:
        bundle = self.models.get("career_rec")
        if not bundle:
            return {"error": "Model 08_career_recommendation_model.pkl not loaded"}

        if selected_interests is None:
            selected_interests = ["Technology", "Problem Solving"]

        m8_map = self.mappings.get("08_career", {})
        current_skills_count = len([s for s in current_skills_text.split(",") if s.strip()])

        # Auto-compute realistic match score if not provided
        if match_score is None:
            interests_count = len(selected_interests)
            match_score = round(min(96.0, max(70.0, 68.0 + min(14.0, current_skills_count * 2.5) + min(10.0, interests_count * 2.0) + min(6.0, years_experience * 1.2))), 1)

        work_style_enc = self._encode_val(work_style, m8_map.get("work_style_classes", []))
        recommended_industry_enc = self._encode_val(recommended_industry, m8_map.get("recommended_industry_classes", []))
        education_level_enc = self._encode_val(education_level, m8_map.get("education_level_classes", []))

        # Interest flags
        interest_cats = [
            'Business & Strategy', 'Communication & Media', 'Data & Analytics', 'Design & Creativity',
            'Healthcare & Wellbeing', 'People & Culture', 'Problem Solving', 'Technology'
        ]
        interest_values = {}
        for cat in interest_cats:
            col = 'interest_' + cat.split(' ')[0].lower().replace('&', 'and')
            interest_values[col] = 1 if cat in selected_interests else 0

        features = [
            years_experience,
            match_score,
            current_skills_count,
            work_style_enc,
            recommended_industry_enc,
            education_level_enc,
            interest_values.get("interest_business", 0),
            interest_values.get("interest_communication", 0),
            interest_values.get("interest_data", 0),
            interest_values.get("interest_design", 0),
            interest_values.get("interest_healthcare", 0),
            interest_values.get("interest_people", 0),
            interest_values.get("interest_problem", 0),
            interest_values.get("interest_technology", 0),
        ]

        df_feat = pd.DataFrame([features], columns=bundle["features"])
        model = bundle["model"]
        label_classes = bundle.get("label_encoder_classes", [])

        interests_str = ", ".join(selected_interests[:3]) if selected_interests else "Technology"

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(df_feat)[0]
            # Map probabilities to classes
            top_indices = np.argsort(probas)[::-1][:top_k]
            recommendations = []
            for rank, idx in enumerate(top_indices, 1):
                class_label = label_classes[idx] if idx < len(label_classes) else f"Role #{idx}"
                conf = round(float(probas[idx]) * 100, 1)
                
                reason = f"High affinity with your passion for {interests_str} and skill profile in {current_skills_text[:45]}."
                recommendations.append({
                    "rank": rank,
                    "career_title": class_label,
                    "confidence_percentage": conf,
                    "reasoning": reason
                })
        else:
            pred_idx = model.predict(df_feat)[0]
            title = label_classes[pred_idx] if pred_idx < len(label_classes) else str(pred_idx)
            recommendations = [{
                "rank": 1,
                "career_title": title,
                "confidence_percentage": 92.5,
                "reasoning": f"Recommended based on your focus in {interests_str}."
            }]

        return {
            "top_recommendations": recommendations,
            "selected_interests_count": len(selected_interests),
            "match_score": match_score,
            "work_style": work_style,
            "recommended_industry": recommended_industry,
            "education_level": education_level,
        }

        return {
            "top_recommendations": recommendations,
            "selected_interests_count": len(selected_interests),
            "match_score": match_score,
        }
