"""
AI Career Mentor - Full Interactive Web Application
Built with Streamlit & 8 Machine Learning Models
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config FIRST before any other streamlit calls
st.set_page_config(
    page_title="AI Career Mentor - 8 ML Models Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern glassmorphic & premium dashboard styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #f8fafc;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 0.25rem;
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    .result-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border-left: 5px solid #38bdf8;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .badge-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-blue { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-green { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* Custom buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.4);
    }
    /* Mobile and Tablet responsiveness */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem 1.25rem;
            margin-bottom: 1.25rem;
        }
        .main-header h1 {
            font-size: 1.6rem !important;
        }
        .main-header p {
            font-size: 0.9rem !important;
        }
        .metric-card {
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }
        .metric-val {
            font-size: 1.5rem !important;
        }
        .result-box {
            padding: 1.25rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Import our engine
from mentor_engine import CareerMentorEngine
from country_data import COUNTRY_ECONOMIC_DATA, EXCHANGE_RATE_TO_USD, CURRENCY_SYMBOL

@st.cache_resource
def get_engine():
    return CareerMentorEngine.get_instance()

engine = get_engine()
mappings = engine.mappings

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("### **AI Career Mentor**")
    st.caption("Powered by 8 HistGradientBoosting ML Models")
    
    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
            "🏠 Dashboard Overview",
            "📄 1. Resume ATS Analyzer",
            "🎯 2. Skill Gap Analyzer",
            "🗺️ 3. Roadmap Generator",
            "💡 4. Interview Prep Evaluator",
            "💼 5. LinkedIn Profile Reviewer",
            "🐙 6. GitHub Portfolio Reviewer",
            "💰 7. Global Salary Predictor",
            "🚀 8. Career Recommender",
        ],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.markdown("🟢 **8/8 Models Loaded**")
    st.markdown("🟢 **Numbeo 2026 Live Data**")
    st.markdown("🟢 **15 Currencies Supported**")
    st.caption("v1.0 • Multi-Currency AI Career Suite")

# =============================================================================
# 0. DASHBOARD OVERVIEW
# =============================================================================
if menu == "🏠 Dashboard Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🎓 AI Career Mentor Intelligence Suite</h1>
        <p>An end-to-end career guidance platform powered by 8 state-of-the-art machine learning models trained on 1.6M+ career trajectories, real-world Numbeo cost-of-living indices, and global currency metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">Trained ML Models</div>
            <div class="metric-val">8 Models</div>
            <div class="metric-sub">100% Operational</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">Salary Model R²</div>
            <div class="metric-val" style="color: #34d399;">0.984</div>
            <div class="metric-sub">Real Economic Enrichment</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">Resume ATS Model R²</div>
            <div class="metric-val" style="color: #c084fc;">0.935</div>
            <div class="metric-sub">Industry Standard Scoring</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">Career Rec Top-5</div>
            <div class="metric-val" style="color: #fbbf24;">83.0%</div>
            <div class="metric-sub">Across 63 Career Paths</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🛠️ All 8 AI Modules Available")
    
    grid1, grid2 = st.columns(2)
    with grid1:
        st.markdown("""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h4>📄 1. Resume ATS Score Analyzer</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Predicts applicant tracking system score (0-100), detects skills & keywords match, and provides optimization suggestions.</p>
            <span class="badge-pill badge-blue">R²: 0.935</span> <span class="badge-pill badge-green">Regression</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h4>🎯 2. Skill Gap Analyzer</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Compares current skill inventory against target dream roles and computes required study months and readiness score.</p>
            <span class="badge-pill badge-blue">R²: 0.771</span> <span class="badge-pill badge-purple">Timeline Estimator</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h4>🗺️ 3. Roadmap Duration Generator</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Calculates total months, study weeks, and learning hours needed based on weekly commitments and certifications.</p>
            <span class="badge-pill badge-blue">R²: 0.900</span> <span class="badge-pill badge-green">Planning</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h4>💡 4. Interview Prep Evaluator</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Assesses technical/behavioral interview question difficulty and provides ideal response length guidelines.</p>
            <span class="badge-pill badge-amber">Classification</span> <span class="badge-pill badge-blue">Multi-Class</span>
        </div>
        """, unsafe_allow_html=True)

    with grid2:
        st.markdown("""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h4>💼 5. LinkedIn Profile Reviewer</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Audits headline, summary, recommendations, and engagement metrics to assign a profile rating (Good, Excellent, etc.).</p>
            <span class="badge-pill badge-green">Accuracy: 99.4%</span> <span class="badge-pill badge-purple">Social Audit</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h4>🐙 6. GitHub Portfolio Reviewer</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Analyzes commit streaks, public repositories, stars, forks, and README coverage to rate open-source portfolio strength.</p>
            <span class="badge-pill badge-green">Accuracy: 99.7%</span> <span class="badge-pill badge-blue">Dev Audit</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h4>💰 7. Global Salary Predictor</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Enriched with Numbeo 2026 economic data and live FX rates. Predicts compensation in USD and <strong>Local Currency (INR ₹, USD $, EUR €, etc.)</strong>.</p>
            <span class="badge-pill badge-blue">R²: 0.984</span> <span class="badge-pill badge-green">PPP Adjusted</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <h4>🚀 8. AI Career Recommender</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Shortlists Top 3 & Top 5 personalized career recommendations from 63 specialized roles based on interests and skills.</p>
            <span class="badge-pill badge-amber">Top-5 Acc: 83.0%</span> <span class="badge-pill badge-purple">Recommender</span>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# 1. RESUME ATS SCORE ANALYZER
# =============================================================================
elif menu == "📄 1. Resume ATS Analyzer":
    st.markdown("""
    <div class="main-header">
        <h1>📄 Resume ATS Score Analyzer</h1>
        <p>Evaluate your resume against Applicant Tracking Systems (ATS) and get a predictive score (0-100) with keyword recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    m1_map = mappings.get("01_resume", {})
    industries = m1_map.get("industry", ["Technology", "Finance", "Healthcare", "Education", "Marketing"])
    roles = m1_map.get("current_job_title", ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer"])
    edu_levels = ["High School", "Diploma", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"]
    degrees = m1_map.get("degree_field", ["Computer Science", "Engineering", "Business Administration", "Data Science"])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("👤 Candidate Profile")
        age = st.number_input("Age", min_value=18, max_value=70, value=26)
        years_exp = st.number_input("Years of Professional Experience", min_value=0.0, max_value=40.0, value=3.5, step=0.5)
        industry = st.selectbox("Target Industry", industries, index=0)
        job_title = st.selectbox("Current / Desired Job Title", roles, index=0)
        education = st.selectbox("Education Level", edu_levels, index=3)
        degree = st.selectbox("Degree Field", degrees, index=0)

    with col2:
        st.subheader("📝 Resume Content Metrics")
        word_count = st.slider("Resume Word Count", min_value=100, max_value=1200, value=350, step=25)
        keyword_match = st.slider("Target Job Keyword Match (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        overall_rating = st.selectbox("Self-Assessment / Initial Quality", ["Good", "Excellent", "Average", "Below Average", "Poor"], index=0)
        skills_input = st.text_area("Skills (Comma-separated)", placeholder="e.g. Python, SQL, Machine Learning, Git, Docker, REST APIs")
        certs_input = st.text_area("Certifications (Comma-separated)", placeholder="e.g. AWS Certified Solutions Architect, Scikit-Learn Specialist")
        missing_kw = st.text_input("Missing Keywords (Comma-separated)", placeholder="e.g. Kubernetes, CI/CD")

    if st.button("🚀 Calculate ATS Score", use_container_width=True):
        res = engine.predict_resume_ats(
            age=age,
            years_experience=years_exp,
            word_count=word_count,
            keyword_match_percentage=keyword_match,
            skills_text=skills_input,
            certifications_text=certs_input,
            missing_keywords_text=missing_kw,
            overall_rating=overall_rating,
            education_level=education,
            degree_field=degree,
            industry=industry,
            current_job_title=job_title,
        )

        st.markdown("---")
        score = res["ats_score"]
        
        # Display ATS gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': f"Predicted ATS Score<br><span style='font-size:0.9em;color:#94a3b8'>{res['category']}</span>"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                    {'range': [50, 75], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [75, 100], 'color': "rgba(52, 211, 153, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "#34d399", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
        st.plotly_chart(fig, use_container_width=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Skills Identified", res["skills_detected"])
        k2.metric("Certifications", res["certifications_detected"])
        k3.metric("Keyword Match", f"{keyword_match:.0f}%")
        k4.metric("Missing Keywords", res["missing_keywords_count"])

        if score >= 80:
            st.success("🎉 **Great job!** Your resume structure and keywords strongly match ATS screening algorithms.")
        elif score >= 65:
            st.info("💡 **Good foundation!** Consider adding the missing keywords and quantifying your project achievements to break into the 80+ tier.")
        else:
            st.warning("⚠️ **Optimization Recommended:** Increase keyword density, specify core certifications, and expand your technical skill list.")

# =============================================================================
# 2. SKILL GAP ANALYZER
# =============================================================================
elif menu == "🎯 2. Skill Gap Analyzer":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Skill Gap & Readiness Analyzer</h1>
        <p>Compare your current technical skillset with requirements for your dream career role and predict learning duration.</p>
    </div>
    """, unsafe_allow_html=True)

    m2_map = mappings.get("02_skillgap", {})
    industries = m2_map.get("industry", ["Technology", "Data Science", "Finance", "Healthcare"])
    curr_roles = m2_map.get("current_role", ["Junior Developer", "Data Analyst", "QA Tester", "Student"])
    target_roles = m2_map.get("target_role", ["Senior Data Scientist", "Full Stack Architect", "ML Engineer", "DevOps Lead"])
    edu_levels = ["High School", "Diploma", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"]

    c1, c2 = st.columns(2)
    with c1:
        industry = st.selectbox("Industry Domain", industries)
        current_role = st.selectbox("Current Role / Background", curr_roles)
        target_role = st.selectbox("Target Dream Role", target_roles)
        education = st.selectbox("Education Level", edu_levels, index=3)
        priority = st.select_slider("Learning Pace / Priority", options=["Low", "Medium", "High"], value="High")

    with c2:
        curr_skills = st.text_area(
            "Your Current Skills (Comma-separated)", 
            placeholder="e.g. HTML, CSS, JavaScript, Git, Python, SQL",
            value="HTML, CSS, JavaScript, Git"
        )
        st.caption("💡 Enter the skills/tools you know. AI will automatically compare against target role benchmarks from 200,000 profiles.")

    if st.button("🚀 Analyze Skill Gap & Career Readiness", use_container_width=True):
        res = engine.predict_skill_gap(
            current_skills=curr_skills,
            industry=industry,
            current_role=current_role,
            target_role=target_role,
            education_level=education,
            learning_priority=priority,
        )

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("⏱️ Estimated Transition Duration", f"{res['estimated_months']} Months", delta=f"{priority} Priority")
        m2.metric("🎯 Role Readiness Score", f"{res['readiness_percentage']:.0f}%", delta="Target: 100%")
        m3.metric("🧩 Missing Skill Count", f"{res['skill_gap_count']} Skills Missing", delta_color="inverse")

        # Visual progress bar
        st.progress(res['readiness_percentage'] / 100.0, text=f"Role Readiness Match: {res['readiness_percentage']:.0f}%")
        
        c_str, c_bench = st.columns(2)
        with c_str:
            st.markdown("#### 🟢 Verified Strengths (You Have)")
            matched = res.get("matched_skills", [])
            if matched:
                st.write(" ".join([f"`✓ {s}`" for s in matched]))
            else:
                st.info("No direct overlap with target role benchmarks yet. Start with core fundamentals!")

        with c_bench:
            st.markdown("#### 🎯 Target Role Benchmark Skills")
            bench = res.get("benchmark_required_skills", [])
            st.write(" ".join([f"`{s}`" for s in bench]))

        st.markdown("#### 📚 Priority Missing Skills & Recommended Learning Resources")
        missing_res = res.get("missing_skills_with_resources", [])
        if missing_res:
            for idx, item in enumerate(missing_res, 1):
                st.markdown(f"- 🔴 **{item['skill']}** (Priority {idx}) ➔ 📖 *{item['resource']}*")
        else:
            st.success("🎉 Outstanding! You already possess all standard core competencies for this role!")

# =============================================================================
# 3. ROADMAP DURATION GENERATOR
# =============================================================================
elif menu == "🗺️ 3. Roadmap Generator":
    st.markdown("""
    <div class="main-header">
        <h1>🗺️ Career Roadmap Duration & Milestones</h1>
        <p>Predict the total learning timeline and study commitment needed to execute a career transition roadmap.</p>
    </div>
    """, unsafe_allow_html=True)

    m3_map = mappings.get("03_roadmap", {})
    industries = m3_map.get("industry", ["Technology", "Healthcare", "Finance", "Education"])
    curr_roles = m3_map.get("current_role", ["Junior Developer", "Data Analyst", "Student", "IT Support"])
    target_roles = m3_map.get("target_role", ["Cloud Solutions Architect", "Machine Learning Specialist", "Cybersecurity Lead"])

    c1, c2 = st.columns(2)
    with c1:
        industry = st.selectbox("Industry Domain", industries)
        current_role = st.selectbox("Starting Role", curr_roles)
        target_role = st.selectbox("Goal Role", target_roles)
        difficulty = st.select_slider("Target Topic Difficulty", options=["Easy", "Moderate", "Challenging", "Hard"], value="Moderate")

    with c2:
        weekly_hours = st.slider("Weekly Study Hours Commitment", min_value=5, max_value=40, value=15, step=1)
        num_phases = st.slider("Number of Planned Roadmap Phases", min_value=2, max_value=8, value=4)
        has_cert = st.checkbox("Includes Preparing for a Professional Industry Certification", value=False)
        focus_skills = st.text_input("Focus Skills (Comma-separated)", placeholder="e.g. Cloud Architecture, Terraform, Kubernetes, Linux, Python")

    if st.button("🚀 Generate Roadmap Timeline", use_container_width=True):
        res = engine.predict_roadmap_duration(
            weekly_hours_commitment=weekly_hours,
            number_of_phases=num_phases,
            focus_skills=focus_skills,
            difficulty_level=difficulty,
            has_target_cert=has_cert,
            industry=industry,
            current_role=current_role,
            target_role=target_role,
        )

        st.markdown("---")
        k1, k2, k3 = st.columns(3)
        k1.metric("📅 Total Duration", f"{res['total_duration_months']} Months")
        k2.metric("📆 Total Study Weeks", f"{res['total_weeks']} Weeks")
        k3.metric("⏳ Estimated Study Hours", f"{res['total_study_hours']} Hours", delta=f"{weekly_hours} hrs/week")

        st.markdown("### 🗺️ Basic to Advanced Multi-Phase Curriculum")
        
        phases = res.get("phases", [])
        if phases:
            for p in phases:
                with st.expander(f"📍 Phase {p['phase_number']}: {p['title']} — {p['badge']}", expanded=True):
                    st.markdown(f"**Level:** `{p['level']}`")
                    st.markdown(f"**Details:** {p['details']}")
                    if p.get("key_topics"):
                        st.markdown(f"**Key Topics:** " + " • ".join([f"`{t}`" for t in p["key_topics"]]))
        else:
            phase_duration = round(res['total_duration_months'] / num_phases, 1)
            phases_data = []
            for i in range(1, num_phases + 1):
                phases_data.append({
                    "Phase": f"Phase {i}",
                    "Duration": f"~{phase_duration} months",
                    "Weekly Commitment": f"{weekly_hours} hours/week",
                    "Milestone Goal": f"Complete Milestone {i} & Capstone Module",
                })
            st.dataframe(pd.DataFrame(phases_data), use_container_width=True, hide_index=True)

        # Milestones & Projects
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("#### 🎯 Key Milestones Target")
            for ms in res.get("milestones", []):
                st.markdown(f"- ✅ **{ms}**")

        with rc2:
            st.markdown("#### 🚀 Recommended Portfolio Projects")
            for proj in res.get("recommended_projects", []):
                st.markdown(f"- 🛠️ **{proj}**")

        if res.get("top_certifications"):
            st.markdown("#### 📜 Target Industry Certifications")
            st.info(" • ".join([f"🏆 **{c}**" for c in res["top_certifications"]]))

# =============================================================================
# 4. INTERVIEW PREP EVALUATOR
# =============================================================================
elif menu == "💡 4. Interview Prep Evaluator":
    st.markdown("""
    <div class="main-header">
        <h1>💡 AI Interview Question Generator & Prep Studio</h1>
        <p>Generate and master all essential technical, behavioral, and situational interview questions for your chosen role.</p>
    </div>
    """, unsafe_allow_html=True)

    m4_map = mappings.get("04_interview", {})
    industries = m4_map.get("industry", ["Information Technology", "Finance", "Healthcare", "Data Science", "Engineering"])
    job_titles = m4_map.get("job_title", ["Full Stack Developer", "Data Scientist", "AI Research Engineer", "Cloud Engineer", "Product Designer"])

    c1, c2 = st.columns(2)
    with c1:
        industry = st.selectbox("Industry Domain", industries)
        job_title = st.selectbox("Target Role for Interview", job_titles)
        q_type = st.selectbox("Question Category Filter", ["All", "Technical", "Behavioral", "Situational"])

    with c2:
        diff_filter = st.selectbox("Difficulty Filter", ["All", "Easy", "Medium", "Hard"])
        search_kw = st.text_input("Search Keywords (Optional)", placeholder="e.g. React, SQL, Microservices, Conflict")

    if st.button("🚀 ⚡ Generate Role Interview Questions", use_container_width=True):
        bank = engine.get_interview_question_bank(
            job_title=job_title,
            industry=industry,
            question_type=q_type,
            difficulty_level=diff_filter,
            search_query=search_kw.strip() if search_kw else None,
        )

        st.markdown("---")
        st.subheader(f"📋 {bank['job_title']} Interview Question Bank")
        
        # Stat chips
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📦 Displayed", f"{bank['display_count']} Questions")
        types = bank.get("type_breakdown", {})
        k2.metric("💻 Technical", types.get("Technical", 0))
        k3.metric("🧠 Behavioral", types.get("Behavioral", 0))
        k4.metric("🎯 Situational", types.get("Situational", 0))

        st.info("💡 **STAR Framework Tip:** Structure your answers: **Situation** (context) ➔ **Task** (goal) ➔ **Action** (technical implementation) ➔ **Result** (metrics & outcome).")

        questions = bank.get("questions", [])
        if not questions:
            st.warning("No questions matched your filter. Try choosing 'All' categories or clearing search keywords.")
        else:
            for idx, q in enumerate(questions, 1):
                diff_emoji = "🔴" if q["difficulty_level"] == "Hard" else ("🟡" if q["difficulty_level"] == "Medium" else "🟢")
                with st.expander(f"#{idx} [{q['question_type']}] {diff_emoji} {q['difficulty_level']}: {q['question_text'][:80]}...", expanded=(idx <= 3)):
                    st.markdown(f"### **{q['question_text']}**")
                    st.markdown(f"- **Category:** `{q['question_type']}` | **Difficulty:** `{q['difficulty_level']}` | **Expected Answer Length:** `~{q.get('ideal_answer_length_words', 200)} words`")
                    
                    st.markdown("#### 🔍 What Hiring Managers Look For:")
                    for pt in q.get("key_evaluation_points", []):
                        st.markdown(f"- ✅ {pt}")

        st.markdown("""
        <div class="result-box">
            <h4>💡 Recommended Answer Structuring (STAR Method)</h4>
            <ul>
                <li><strong>Situation:</strong> Set the context and specify the engineering constraints.</li>
                <li><strong>Task:</strong> State your objective and requirements clearly.</li>
                <li><strong>Action:</strong> Walk step-by-step through the technical implementation and architecture choices.</li>
                <li><strong>Result:</strong> Quantify the outcome (latency reduction, accuracy improvement, cost savings).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# 5. LINKEDIN PROFILE REVIEWER
# =============================================================================
elif menu == "💼 5. LinkedIn Profile Reviewer":
    st.markdown("""
    <div class="main-header">
        <h1>💼 LinkedIn Profile Reviewer & Optimizer</h1>
        <p>Audit your LinkedIn metrics, completeness score, and engagement to predict your profile rating.</p>
    </div>
    """, unsafe_allow_html=True)

    m5_map = mappings.get("05_linkedin", {})
    industries = m5_map.get("industry", ["Technology", "Finance", "Healthcare", "Marketing", "Consulting"])
    job_titles = m5_map.get("current_job_title", ["Software Engineer", "Data Scientist", "Product Manager", "Tech Lead"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📋 Profile Setup & Presence")
        industry = st.selectbox("Industry", industries)
        job_title = st.selectbox("Current Job Title / Headline Target", job_titles)
        has_photo = st.checkbox("Professional Profile Photo Uploaded", value=True)
        has_banner = st.checkbox("Custom Background Banner Image Added", value=True)
        completeness = st.slider("Profile Completeness Score (%)", min_value=0.0, max_value=100.0, value=85.0)
        summary_words = st.slider("Summary / About Section Word Count", min_value=0, max_value=500, value=180)

    with c2:
        st.subheader("🌐 Network & Activity")
        connections = st.number_input("Number of Connections", min_value=0, max_value=30000, value=500, step=50)
        skills_listed = st.number_input("Listed Skills Count", min_value=0, max_value=50, value=18)
        endorsements = st.number_input("Total Skill Endorsements", min_value=0, max_value=500, value=45)
        recommendations = st.number_input("Written Recommendations Count", min_value=0, max_value=30, value=4)
        posts_90d = st.number_input("Posts in Last 90 Days", min_value=0, max_value=100, value=12)
        avg_eng = st.number_input("Average Reactions/Comments per Post", min_value=0.0, max_value=500.0, value=8.5)

    if st.button("🚀 Audit LinkedIn Profile", use_container_width=True):
        res = engine.predict_linkedin_rating(
            has_profile_photo=has_photo,
            has_banner_image=has_banner,
            summary_word_count=summary_words,
            connections_count=connections,
            skills_count=skills_listed,
            total_endorsements=endorsements,
            recommendations_count=recommendations,
            posts_last_90_days=posts_90d,
            avg_engagement_per_post=avg_eng,
            profile_completeness_score=completeness,
            industry=industry,
            current_job_title=job_title,
        )

        st.markdown("---")
        st.subheader(f"Predicted Profile Rating: **{res['predicted_rating']}**")

        if res["rating_probabilities"]:
            df_prob = pd.DataFrame(list(res["rating_probabilities"].items()), columns=["Rating Category", "Probability (%)"])
            fig = px.pie(
                df_prob,
                names="Rating Category",
                values="Probability (%)",
                title="Profile Rating Probability Distribution",
                hole=0.45,
                color_discrete_sequence=["#38bdf8", "#34d399", "#fbbf24", "#a855f7", "#f87171"]
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="result-box">
            <h4>✨ Actionable Optimization Tips</h4>
            <ul>
                <li><strong>Headline:</strong> Use a high-converting formula (Role | Core Tech Stack | Impact/Value Proposition).</li>
                <li><strong>Endorsements & Recommendations:</strong> Reach out to 3-5 former colleagues for reciprocal recommendations.</li>
                <li><strong>Activity:</strong> Sharing 1-2 insightful technical posts weekly boosts recruiter search impressions by 3x.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# 6. GITHUB PORTFOLIO REVIEWER
# =============================================================================
elif menu == "🐙 6. GitHub Portfolio Reviewer":
    st.markdown("""
    <div class="main-header">
        <h1>🐙 GitHub Portfolio Reviewer</h1>
        <p>Analyze developer repository activity, stars, commit consistency, and open-source contributions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-fetch bar
    st.markdown("### ⚡ Instant Auto-Fetch from GitHub Profile")
    gh_col1, gh_col2 = st.columns([3, 1])
    with gh_col1:
        gh_input_url = st.text_input("GitHub Profile URL or Username", placeholder="e.g. https://github.com/torvalds or octocat", label_visibility="collapsed")
    with gh_col2:
        fetch_btn = st.button("🔍 Fetch & Sync", use_container_width=True)

    fetched_data = None
    if fetch_btn and gh_input_url:
        with st.spinner("Connecting to GitHub API and syncing repositories..."):
            fetched_data = engine.fetch_github_profile(gh_input_url)
            if "error" in fetched_data:
                st.error(fetched_data["error"])
            else:
                st.success(f"✅ Successfully synced profile for **@{fetched_data['username']}** ({fetched_data['display_name']})!")
                st.session_state["gh_fetched"] = fetched_data

    gh_preset = st.session_state.get("gh_fetched", {}).get("metrics", {})
    if "gh_fetched" in st.session_state and st.session_state["gh_fetched"].get("avatar_url"):
        u_prev = st.session_state["gh_fetched"]
        st.markdown(f"""
        <div class="metric-card" style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
            <img src="{u_prev['avatar_url']}" style="width:50px; height:50px; border-radius:50%; border:2px solid #38bdf8;">
            <div>
                <h4 style="margin:0; color:#fff;">{u_prev['display_name']} <span style="color:#38bdf8; font-size:0.9rem;">(@{u_prev['username']})</span></h4>
                <p style="margin:0; font-size:0.85rem; color:#94a3b8;">{u_prev.get('bio') or 'GitHub Public Profile'}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    m6_map = mappings.get("06_github", {})
    focus_areas = m6_map.get("focus_area", ["Web Development", "Data Science & ML", "DevOps & Cloud", "Mobile Apps", "Systems & Security"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📦 Repositories & Social")
        focus_idx = focus_areas.index(gh_preset["focus_area"]) if "focus_area" in gh_preset and gh_preset["focus_area"] in focus_areas else 0
        focus_area = st.selectbox("Primary Developer Focus Area", focus_areas, index=focus_idx)
        public_repos = st.number_input("Public Repositories", min_value=0, max_value=500, value=gh_preset.get("public_repos", 15))
        pinned_repos = st.slider("Pinned High-Quality Repositories", min_value=0, max_value=6, value=gh_preset.get("pinned_repos_count", 3))
        total_stars = st.number_input("Total Repository Stars", min_value=0, max_value=50000, value=gh_preset.get("total_stars", 25))
        total_forks = st.number_input("Total Forks", min_value=0, max_value=10000, value=gh_preset.get("total_forks", 8))
        top_repo_stars = st.number_input("Stars on Most Popular Repo", min_value=0, max_value=30000, value=gh_preset.get("top_repo_stars", 15))
        followers = st.number_input("GitHub Followers", min_value=0, max_value=100000, value=gh_preset.get("followers", 20))
        following = st.number_input("Following", min_value=0, max_value=50000, value=gh_preset.get("following", 20))

    with c2:
        st.subheader("🔥 Contributions & Quality")
        contributions = st.number_input("Contributions in Last Year", min_value=0, max_value=10000, value=gh_preset.get("contributions_last_year", 250))
        longest_streak = st.number_input("Longest Contribution Streak (Days)", min_value=0, max_value=365, value=gh_preset.get("longest_streak_days", 14))
        readme_coverage = st.slider("Repositories with Detailed README (%)", min_value=0.0, max_value=100.0, value=float(gh_preset.get("readme_coverage_percentage", 75.0)))
        os_contribs = st.number_input("External Open Source Contributions", min_value=0, max_value=1000, value=gh_preset.get("open_source_contributions", 5))
        has_bio = st.checkbox("Custom GitHub Profile README / Bio Configured", value=gh_preset.get("has_bio", False))
        profile_score = st.slider("Calculated Developer Profile Score", min_value=0.0, max_value=100.0, value=float(gh_preset.get("profile_score", 70.0)))
        languages = st.text_input("Languages Used (Comma-separated)", value=gh_preset.get("languages_used", ""), placeholder="e.g. Python, TypeScript, Go, Dockerfile, SQL")

    if st.button("🚀 Analyze GitHub Portfolio", use_container_width=True):
        res = engine.predict_github_rating(
            public_repos=public_repos,
            followers=followers,
            following=following,
            total_stars=total_stars,
            total_forks=total_forks,
            contributions_last_year=contributions,
            longest_streak_days=longest_streak,
            readme_coverage_percentage=readme_coverage,
            pinned_repos_count=pinned_repos,
            top_repo_stars=top_repo_stars,
            has_bio=has_bio,
            open_source_contributions=os_contribs,
            profile_score=profile_score,
            languages_used_text=languages,
            focus_area=focus_area,
        )

        st.markdown("---")
        st.subheader(f"Portfolio Strength Rating: **{res['predicted_rating']}**")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("⭐ Stars Earned", total_stars)
        k2.metric("🔥 Contributions", contributions)
        k3.metric("📖 README Coverage", f"{readme_coverage:.0f}%")
        k4.metric("🏆 Profile Score", f"{profile_score:.0f}/100")

        if res["rating_probabilities"]:
            df_prob = pd.DataFrame(list(res["rating_probabilities"].items()), columns=["Rating", "Probability (%)"])
            fig = px.bar(
                df_prob,
                x="Rating",
                y="Probability (%)",
                color="Rating",
                title="Model Rating Classification Confidence",
                color_discrete_sequence=["#38bdf8", "#34d399", "#fbbf24", "#f87171"]
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 7. GLOBAL SALARY PREDICTOR (REAL-DATA ENRICHED & MULTI-CURRENCY)
# =============================================================================
elif menu == "💰 7. Global Salary Predictor":
    st.markdown("""
    <div class="main-header">
        <h1>💰 Global Salary Predictor (Multi-Currency & Real Data)</h1>
        <p>Predict global market compensation in USD and <strong>Local Currency (INR ₹, USD $, EUR €, GBP £, etc.)</strong> enriched with real Numbeo cost-of-living indices and purchasing power parity.</p>
    </div>
    """, unsafe_allow_html=True)

    m7_map = mappings.get("07_salary", {})
    industries = m7_map.get("industry", ["Technology", "Finance", "Healthcare", "Education", "Consulting"])
    job_titles = m7_map.get("job_title", ["Data Scientist", "Software Engineer", "Machine Learning Engineer", "DevOps Engineer", "Product Manager"])
    edu_levels = ["High School", "Diploma", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"]
    degrees = m7_map.get("degree_field", ["Computer Science", "Engineering", "Business", "Mathematics", "Data Science"])
    countries = list(COUNTRY_ECONOMIC_DATA.keys())
    company_sizes = m7_map.get("company_size", ["Startup", "Small", "Medium", "Large", "Enterprise"])
    work_types = m7_map.get("work_type", ["Remote", "Hybrid", "On-site"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💼 Role & Experience")
        industry = st.selectbox("Industry", industries)
        job_title = st.selectbox("Job Title", job_titles, index=min(1, len(job_titles)-1))
        years_exp = st.number_input("Years of Professional Experience", min_value=0.0, max_value=35.0, value=4.0, step=0.5)
        skills_cnt = st.slider("Key Technical Skills Count", min_value=1, max_value=30, value=8)
        certs_cnt = st.slider("Professional Certifications", min_value=0, max_value=10, value=2)

    with c2:
        st.subheader("🌍 Location & Company")
        # Default to India if available in list
        india_idx = countries.index("India") if "India" in countries else 0
        country = st.selectbox("Country of Employment", countries, index=india_idx)
        education = st.selectbox("Education Level", edu_levels, index=3)
        degree = st.selectbox("Degree Field", degrees)
        company_size = st.selectbox("Company Size", company_sizes, index=min(2, len(company_sizes)-1))
        work_type = st.selectbox("Work Arrangement", work_types, index=1)

    if st.button("🚀 Predict Global & Local Salary", use_container_width=True):
        res = engine.predict_salary(
            years_experience=years_exp,
            skills_count=skills_cnt,
            certifications_count=certs_cnt,
            industry=industry,
            job_title=job_title,
            education_level=education,
            degree_field=degree,
            country=country,
            company_size=company_size,
            work_type=work_type,
        )

        st.markdown("---")
        sym = res["currency_symbol"]
        code = res["currency_code"]

        st.markdown(f"""
        <div class="result-box" style="border-left-color: #34d399;">
            <h2 style="color: #34d399; margin-bottom: 0.25rem;">💵 {res['formatted_local_salary']}</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">Estimated Annual Base Compensation in {country} (Equivalent to <strong>${res['predicted_salary_usd']:,.0f} USD</strong>)</p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Monthly Salary ({code})", f"{sym}{res['monthly_salary_local']:,.0f}")
        k2.metric("Monthly Salary (USD)", f"${res['monthly_salary_usd']:,.0f}")
        k3.metric("Cost of Living Index", f"{res['cost_of_living_index']}", help="Numbeo 2026 Index (100 = NYC Baseline)")
        k4.metric("Purchasing Power Index", f"{res['purchasing_power_index']}", help="Higher means local currency goes further")

        st.markdown("### 📊 Real Economic Comparison")
        comp_df = pd.DataFrame([
            {"Metric": "Nominal Local Annual Salary", "Amount": f"{sym}{res['predicted_salary_local']:,.0f} {code}"},
            {"Metric": "Nominal USD Annual Salary", "Amount": f"${res['predicted_salary_usd']:,.0f} USD"},
            {"Metric": "PPP-Adjusted Equivalent Salary", "Amount": f"${res['ppp_adjusted_salary_usd']:,.0f} USD"},
            {"Metric": "Local Purchasing Power Index", "Amount": f"{res['purchasing_power_index']} (Numbeo 2026)"},
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

# =============================================================================
# 8. CAREER RECOMMENDER
# =============================================================================
elif menu == "🚀 8. Career Recommender":
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Career Recommender (Shortlist Recommender)</h1>
        <p>Shortlist the <strong>Top 3 & Top 5 Best-Fit Career Paths</strong> across 63 specialized professions based on your passions and skills.</p>
    </div>
    """, unsafe_allow_html=True)

    m8_map = mappings.get("08_career", {})
    work_styles = m8_map.get("work_style", ["Remote", "Hybrid", "Collaborative", "Independent", "Structured"])
    industries = m8_map.get("recommended_industry", ["Information Technology", "Data Science", "Design", "Engineering", "Finance", "Healthcare"])
    edu_levels = ["High School", "Diploma", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Background & Work Style")
        skills_text = st.text_area("Your Core Skills (Comma-separated)", placeholder="e.g. Python, SQL, Problem Solving, Communication, Design", value="Python, Problem Solving, SQL")
        industry_pref = st.selectbox("Preferred Industry Domain", industries)
        work_style = st.selectbox("Preferred Work Style", work_styles)
        years_exp = st.number_input("Years of Experience (0 for Fresher)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
        education = st.selectbox("Education Level", edu_levels, index=3)

    with c2:
        st.subheader("💡 Select Your Passions & Interests")
        interest_options = [
            'Technology', 'Problem Solving', 'Data & Analytics', 'Design & Creativity',
            'Business & Strategy', 'People & Culture', 'Communication & Media', 'Healthcare & Wellbeing'
        ]
        
        selected_interests = []
        for cat in interest_options:
            default_chk = cat in ['Technology', 'Problem Solving']
            if st.checkbox(cat, value=default_chk):
                selected_interests.append(cat)

        top_k_select = st.radio("Shortlist Recommendations Size", [3, 5], index=1, horizontal=True)

    if st.button("🚀 ✨ Discover My Best Career Matches", use_container_width=True):
        res = engine.predict_career_recommendations(
            years_experience=years_exp,
            current_skills_text=skills_text,
            work_style=work_style,
            recommended_industry=industry_pref,
            education_level=education,
            selected_interests=selected_interests if selected_interests else ["Technology", "Problem Solving"],
            top_k=top_k_select,
        )

        st.markdown("---")
        st.subheader(f"🏆 Top {top_k_select} Career Recommendations for You")

        recs = res["top_recommendations"]
        df_recs = pd.DataFrame(recs)

        # Plotly horizontal bar chart of confidence
        fig = px.bar(
            df_recs,
            x="confidence_percentage",
            y="career_title",
            orientation="h",
            text="confidence_percentage",
            title=f"AI Match Confidence Distribution (Top-{top_k_select})",
            labels={"confidence_percentage": "Confidence (%)", "career_title": "Career Path"},
            color="confidence_percentage",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        for rec in recs:
            st.markdown(f"""
            <div class="result-box" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;">
                <h4 style="margin-bottom: 0.2rem; color: #38bdf8;">#{rec['rank']} — {rec['career_title']}</h4>
                <p style="margin-bottom: 0; color: #94a3b8; font-size: 0.95rem;">Match Confidence: <strong>{rec['confidence_percentage']}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
