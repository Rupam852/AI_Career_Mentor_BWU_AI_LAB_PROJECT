"""
server.py
---------
FastAPI REST backend serving all 8 AI Career Mentor machine learning models
and hosting the Material 3 + Glassmorphic Web Application.
"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure models directory is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from mentor_engine import CareerMentorEngine
from country_data import COUNTRY_ECONOMIC_DATA, EXCHANGE_RATE_TO_USD, CURRENCY_SYMBOL

app = FastAPI(
    title="AI Career Mentor Intelligence Suite API",
    description="Material 3 + Glassmorphism AI Career Suite powered by 8 ML Models",
    version="1.0.0",
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = CareerMentorEngine.get_instance()

# =============================================================================
# PYDANTIC REQUEST MODELS
# =============================================================================
class ResumeATSRequest(BaseModel):
    age: int = 26
    years_experience: float = 3.5
    word_count: int = 450
    keyword_match_percentage: float = 78.0
    skills: str = "Python, SQL, Machine Learning, Git, Docker, REST APIs"
    certifications: str = "AWS Certified Solutions Architect"
    missing_keywords: str = "Kubernetes, CI/CD"
    overall_rating: str = "Good"
    education_level: str = "Bachelor's Degree"
    degree_field: str = "Computer Science"
    industry: str = "Technology"
    current_job_title: str = "Software Engineer"

class SkillGapRequest(BaseModel):
    current_skills: str = "Python, SQL, Git, Pandas"
    industry: str = "Technology"
    current_role: str = "Junior Developer"
    target_role: str = "Senior Data Scientist"
    education_level: str = "Bachelor's Degree"
    learning_priority: str = "High"
    required_skills: Optional[str] = None
    skill_gap_count: Optional[int] = None
    gap_score: Optional[float] = None
    readiness_percentage: Optional[float] = None

class RoadmapRequest(BaseModel):
    weekly_hours_commitment: float = 15.0
    number_of_phases: int = 4
    focus_skills: str = "Cloud Architecture, Terraform, Kubernetes, Linux, Python"
    difficulty_level: str = "Moderate"
    has_target_cert: bool = True
    industry: str = "Technology"
    current_role: str = "Junior Developer"
    target_role: str = "Cloud Solutions Architect"

class InterviewRequest(BaseModel):
    ideal_answer_length_words: int = 200
    eval_points: str = "Core algorithm explanation; Trade-off analysis; Edge cases"
    question_text: str = "Explain how Gradient Boosting works and compare its loss minimization strategy with Random Forest."
    industry: str = "Technology"
    job_title: str = "Data Scientist"
    question_type: str = "Technical"

class InterviewBankRequest(BaseModel):
    job_title: str = "Full Stack Developer"
    industry: Optional[str] = None
    question_type: Optional[str] = "All"
    difficulty_level: Optional[str] = "All"
    limit: Optional[int] = None
    search_query: Optional[str] = None

class LinkedInRequest(BaseModel):
    has_profile_photo: bool = True
    has_banner_image: bool = True
    summary_word_count: int = 180
    connections_count: int = 500
    skills_count: int = 18
    total_endorsements: int = 45
    recommendations_count: int = 4
    posts_last_90_days: int = 12
    avg_engagement_per_post: float = 8.5
    profile_completeness_score: float = 85.0
    industry: str = "Technology"
    current_job_title: str = "Software Engineer"

class GitHubRequest(BaseModel):
    public_repos: int = 22
    followers: int = 85
    following: int = 45
    total_stars: int = 120
    total_forks: int = 35
    contributions_last_year: int = 480
    longest_streak_days: int = 32
    readme_coverage_percentage: float = 85.0
    pinned_repos_count: int = 4
    top_repo_stars: int = 65
    has_bio: bool = True
    open_source_contributions: int = 15
    profile_score: float = 82.0
    languages_used: str = "Python, TypeScript, Go, Dockerfile, SQL"
    focus_area: str = "Data Science & ML"

class GitHubFetchRequest(BaseModel):
    profile_input: str

class SalaryRequest(BaseModel):
    years_experience: float = 4.0
    skills_count: int = 8
    certifications_count: int = 2
    industry: str = "Technology"
    job_title: str = "Data Scientist"
    education_level: str = "Bachelor's Degree"
    degree_field: str = "Computer Science"
    country: str = "India"
    company_size: str = "Medium"
    work_type: str = "Hybrid"

class CareerRecRequest(BaseModel):
    years_experience: float = 1.0
    current_skills: str = "Python, Problem Solving, Data Analysis"
    work_style: str = "Hybrid"
    recommended_industry: str = "Technology"
    education_level: str = "Bachelor's Degree"
    selected_interests: List[str] = ["Technology", "Problem Solving"]
    match_score: Optional[float] = None
    top_k: int = 5

# =============================================================================
# API ROUTES
# =============================================================================
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "models_loaded": len(engine.models),
        "total_expected": 8,
    }

@app.get("/api/mappings")
def get_mappings():
    return {
        "categories": engine.mappings,
        "countries": list(COUNTRY_ECONOMIC_DATA.keys()),
        "country_economic_data": COUNTRY_ECONOMIC_DATA,
        "currency_symbols": CURRENCY_SYMBOL,
    }

@app.post("/api/predict/resume-ats")
def predict_resume_ats(req: ResumeATSRequest):
    return engine.predict_resume_ats(
        age=req.age,
        years_experience=req.years_experience,
        word_count=req.word_count,
        keyword_match_percentage=req.keyword_match_percentage,
        skills_text=req.skills,
        certifications_text=req.certifications,
        missing_keywords_text=req.missing_keywords,
        overall_rating=req.overall_rating,
        education_level=req.education_level,
        degree_field=req.degree_field,
        industry=req.industry,
        current_job_title=req.current_job_title,
    )

@app.post("/api/predict/skillgap")
def predict_skillgap(req: SkillGapRequest):
    return engine.predict_skill_gap(
        current_skills=req.current_skills,
        required_skills=req.required_skills,
        skill_gap_count=req.skill_gap_count,
        gap_score=req.gap_score,
        readiness_percentage=req.readiness_percentage,
        learning_priority=req.learning_priority,
        industry=req.industry,
        current_role=req.current_role,
        target_role=req.target_role,
        education_level=req.education_level,
    )

@app.post("/api/predict/roadmap")
def predict_roadmap(req: RoadmapRequest):
    return engine.predict_roadmap_duration(
        weekly_hours_commitment=req.weekly_hours_commitment,
        number_of_phases=req.number_of_phases,
        focus_skills=req.focus_skills,
        difficulty_level=req.difficulty_level,
        has_target_cert=req.has_target_cert,
        industry=req.industry,
        current_role=req.current_role,
        target_role=req.target_role,
    )

@app.post("/api/predict/interview")
def predict_interview(req: InterviewRequest):
    return engine.predict_interview_difficulty(
        ideal_answer_length_words=req.ideal_answer_length_words,
        eval_points_text=req.eval_points,
        question_text=req.question_text,
        industry=req.industry,
        job_title=req.job_title,
        question_type=req.question_type,
    )

@app.post("/api/interview/questions")
def get_interview_questions(req: InterviewBankRequest):
    return engine.get_interview_question_bank(
        job_title=req.job_title,
        industry=req.industry,
        question_type=req.question_type,
        difficulty_level=req.difficulty_level,
        limit=req.limit,
        search_query=req.search_query,
    )

@app.post("/api/predict/linkedin")
def predict_linkedin(req: LinkedInRequest):
    return engine.predict_linkedin_rating(
        has_profile_photo=req.has_profile_photo,
        has_banner_image=req.has_banner_image,
        summary_word_count=req.summary_word_count,
        connections_count=req.connections_count,
        skills_count=req.skills_count,
        total_endorsements=req.total_endorsements,
        recommendations_count=req.recommendations_count,
        posts_last_90_days=req.posts_last_90_days,
        avg_engagement_per_post=req.avg_engagement_per_post,
        profile_completeness_score=req.profile_completeness_score,
        industry=req.industry,
        current_job_title=req.current_job_title,
    )

@app.post("/api/predict/github")
def predict_github(req: GitHubRequest):
    return engine.predict_github_rating(
        public_repos=req.public_repos,
        followers=req.followers,
        following=req.following,
        total_stars=req.total_stars,
        total_forks=req.total_forks,
        contributions_last_year=req.contributions_last_year,
        longest_streak_days=req.longest_streak_days,
        readme_coverage_percentage=req.readme_coverage_percentage,
        pinned_repos_count=req.pinned_repos_count,
        top_repo_stars=req.top_repo_stars,
        has_bio=req.has_bio,
        open_source_contributions=req.open_source_contributions,
        profile_score=req.profile_score,
        languages_used_text=req.languages_used,
        focus_area=req.focus_area,
    )

@app.post("/api/github/fetch-profile")
def fetch_github_profile(req: GitHubFetchRequest):
    return engine.fetch_github_profile(req.profile_input)

@app.post("/api/predict/salary")
def predict_salary(req: SalaryRequest):
    return engine.predict_salary(
        years_experience=req.years_experience,
        skills_count=req.skills_count,
        certifications_count=req.certifications_count,
        industry=req.industry,
        job_title=req.job_title,
        education_level=req.education_level,
        degree_field=req.degree_field,
        country=req.country,
        company_size=req.company_size,
        work_type=req.work_type,
    )

@app.post("/api/predict/career-recommendations")
def predict_career_recommendations(req: CareerRecRequest):
    return engine.predict_career_recommendations(
        years_experience=req.years_experience,
        match_score=req.match_score,
        current_skills_text=req.current_skills,
        work_style=req.work_style,
        recommended_industry=req.recommended_industry,
        education_level=req.education_level,
        selected_interests=req.selected_interests,
        top_k=req.top_k,
    )

# =============================================================================
# STATIC FILES SERVING
# =============================================================================
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Material 3 Web App frontend initializing..."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.py:app", host="0.0.0.0", port=8000, reload=True)
