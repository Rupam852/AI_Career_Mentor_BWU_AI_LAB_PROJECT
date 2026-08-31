/**
 * AI Career Mentor - Material 3 + Glassmorphism Web App Controller
 */

// Global State & Mappings
let appMappings = {};
let activeCharts = {};

// DOM Ready
document.addEventListener("DOMContentLoaded", async () => {
    initNavigation();
    await loadMappings();
    initForms();
});

// Navigation Handling
function initNavigation() {
    const navItems = document.querySelectorAll(".m3-nav-item");
    const sections = document.querySelectorAll(".app-section");
    const topTitle = document.getElementById("top-page-title");
    const menuBtn = document.getElementById("mobile-menu-btn");
    const navRail = document.querySelector(".m3-nav-rail");
    const backdrop = document.getElementById("drawer-backdrop");

    // Toggle Mobile Drawer
    function toggleDrawer(open) {
        if (!navRail || !backdrop) return;
        const isOpen = open !== undefined ? open : !navRail.classList.contains("drawer-open");
        if (isOpen) {
            navRail.classList.add("drawer-open");
            backdrop.classList.add("active");
        } else {
            navRail.classList.remove("drawer-open");
            backdrop.classList.remove("active");
        }
    }

    menuBtn?.addEventListener("click", () => toggleDrawer());
    backdrop?.addEventListener("click", () => toggleDrawer(false));

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const target = item.getAttribute("data-target");
            
            navItems.forEach(i => i.classList.remove("active"));
            item.classList.add("active");

            sections.forEach(s => {
                s.style.display = (s.id === target) ? "block" : "none";
            });

            if (topTitle) {
                const label = item.querySelector(".nav-label")?.textContent || "Dashboard";
                topTitle.textContent = label;
            }

            // Close mobile drawer after selection on mobile
            if (window.innerWidth <= 768) {
                toggleDrawer(false);
            }

            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    });
}

// Load Dropdown Mappings from FastAPI
async function loadMappings() {
    try {
        const res = await fetch("/api/mappings");
        const data = await res.json();
        appMappings = data;

        populateSelect("resume-industry", data.categories["01_resume"]?.industry);
        populateSelect("resume-job-title", data.categories["01_resume"]?.current_job_title);
        populateSelect("resume-degree", data.categories["01_resume"]?.degree_field);

        populateSelect("skillgap-industry", data.categories["02_skillgap"]?.industry);
        populateSelect("skillgap-current-role", data.categories["02_skillgap"]?.current_role);
        populateSelect("skillgap-curr-role", data.categories["02_skillgap"]?.current_role);
        populateSelect("skillgap-target-role", data.categories["02_skillgap"]?.target_role);
        populateSelect("skillgap-edu", data.categories["02_skillgap"]?.education_level);

        populateSelect("roadmap-industry", data.categories["03_roadmap"]?.industry);
        populateSelect("roadmap-curr-role", data.categories["03_roadmap"]?.current_role);
        populateSelect("roadmap-target-role", data.categories["03_roadmap"]?.target_role);

        populateSelect("interview-industry", data.categories["04_interview"]?.industry);
        populateSelect("interview-job-title", data.categories["04_interview"]?.job_title);
        populateSelect("interview-type", data.categories["04_interview"]?.question_type);

        populateSelect("linkedin-industry", data.categories["05_linkedin"]?.industry);
        populateSelect("linkedin-job-title", data.categories["05_linkedin"]?.current_job_title);

        populateSelect("github-focus", data.categories["06_github"]?.focus_area);

        populateSelect("salary-industry", data.categories["07_salary"]?.industry);
        populateSelect("salary-job-title", data.categories["07_salary"]?.job_title);
        populateSelect("salary-degree", data.categories["07_salary"]?.degree_field);
        populateSelect("salary-country", data.countries, "India");
        populateSelect("salary-company-size", data.categories["07_salary"]?.company_size);

        populateSelect("career-work-style", data.categories["08_career"]?.work_style);
        populateSelect("career-industry", data.categories["08_career"]?.recommended_industry);

        initInterestChips();
    } catch (err) {
        console.error("Failed to load mappings:", err);
    }
}

function populateSelect(elemId, items, defaultVal = null) {
    const sel = document.getElementById(elemId);
    if (!sel || !items) return;
    sel.innerHTML = "";
    items.forEach(item => {
        const opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item;
        if (defaultVal && item === defaultVal) opt.selected = true;
        sel.appendChild(opt);
    });
}

function initInterestChips() {
    const chips = document.querySelectorAll(".interest-chip");
    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            chip.classList.toggle("selected");
        });
    });
}

// Helper to get sanitized inputs
function getStr(id, fallback = "") {
    const el = document.getElementById(id);
    return el ? (el.value || "").trim() : fallback;
}

function getNum(id, fallback = 0) {
    const el = document.getElementById(id);
    if (!el || el.value === "") return fallback;
    const n = parseInt(el.value, 10);
    return isNaN(n) ? fallback : n;
}

function getFloat(id, fallback = 0.0) {
    const el = document.getElementById(id);
    if (!el || el.value === "") return fallback;
    const n = parseFloat(el.value);
    return isNaN(n) ? fallback : n;
}

// Form Handlers
function initForms() {
    // 1. Resume ATS
    document.getElementById("form-resume")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            age: getNum("resume-age", 25),
            years_experience: getFloat("resume-exp", 2.0),
            word_count: getNum("resume-words", 350),
            keyword_match_percentage: getFloat("resume-kw-match", 60.0),
            skills: getStr("resume-skills", "Python, SQL"),
            certifications: getStr("resume-certs", ""),
            missing_keywords: getStr("resume-missing-kw", ""),
            overall_rating: getStr("resume-rating", "Good"),
            education_level: getStr("resume-edu", "Bachelor's Degree"),
            degree_field: getStr("resume-degree", "Computer Science"),
            industry: getStr("resume-industry", "Technology"),
            current_job_title: getStr("resume-job-title", "Software Engineer"),
        };
        const res = await apiPost("/api/predict/resume-ats", payload);
        if (res) renderResumeResult(res);
    });

    // 2. Skill Gap
    document.getElementById("form-skillgap")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            current_skills: getStr("skillgap-curr-skills", "HTML, CSS, JavaScript, Git"),
            industry: getStr("skillgap-industry", "Information Technology"),
            current_role: getStr("skillgap-current-role", "Student / Fresher"),
            target_role: getStr("skillgap-target-role", "Full Stack Developer"),
            education_level: getStr("skillgap-edu", "Bachelor's Degree"),
            learning_priority: getStr("skillgap-priority", "Medium"),
        };
        const res = await apiPost("/api/predict/skillgap", payload);
        if (res) renderSkillgapResult(res, payload);
    });

    // 3. Roadmap
    document.getElementById("form-roadmap")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            weekly_hours_commitment: getFloat("roadmap-hours", 15.0),
            number_of_phases: getNum("roadmap-phases", 4),
            focus_skills: getStr("roadmap-skills", "Python, Cloud, Architecture"),
            difficulty_level: getStr("roadmap-diff", "Moderate"),
            has_target_cert: document.getElementById("roadmap-cert")?.checked || false,
            industry: getStr("roadmap-industry", "Technology"),
            current_role: getStr("roadmap-curr-role", "Junior Developer"),
            target_role: getStr("roadmap-target-role", "Cloud Solutions Architect"),
        };
        const res = await apiPost("/api/predict/roadmap", payload);
        if (res) renderRoadmapResult(res, payload);
    });

    // 4. Interview Question Bank Loader
    document.getElementById("btn-load-interview-bank")?.addEventListener("click", async () => {
        const limitVal = parseInt(document.getElementById("interview-limit-filter")?.value || "25");
        const payload = {
            industry: getStr("interview-industry", "Information Technology"),
            job_title: getStr("interview-job-title", "Full Stack Developer"),
            question_type: getStr("interview-type-filter", "All"),
            difficulty_level: getStr("interview-diff-filter", "All"),
            limit: limitVal > 0 ? limitVal : null,
            search_query: getStr("interview-search-input", "").trim(),
        };
        const res = await apiPost("/api/interview/questions", payload);
        if (res) renderInterviewQuestionBank(res);
    });

    // 5. LinkedIn
    document.getElementById("form-linkedin")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            has_profile_photo: document.getElementById("li-photo")?.checked || false,
            has_banner_image: document.getElementById("li-banner")?.checked || false,
            summary_word_count: getNum("li-words", 100),
            connections_count: getNum("li-conn", 250),
            skills_count: getNum("li-skills", 10),
            total_endorsements: getNum("li-endorse", 20),
            recommendations_count: getNum("li-recs", 2),
            posts_last_90_days: getNum("li-posts", 5),
            avg_engagement_per_post: getFloat("li-eng", 5.0),
            profile_completeness_score: getFloat("li-comp", 75.0),
            industry: getStr("linkedin-industry", "Technology"),
            current_job_title: getStr("linkedin-job-title", "Software Engineer"),
        };
        const res = await apiPost("/api/predict/linkedin", payload);
        if (res) renderLinkedInResult(res);
    });

    // 6. GitHub Auto-Fetch from URL/Username
    document.getElementById("btn-gh-autofetch")?.addEventListener("click", async () => {
        const input = document.getElementById("gh-url-input")?.value?.trim();
        const statusEl = document.getElementById("gh-fetch-status");
        const btn = document.getElementById("btn-gh-autofetch");
        const previewCard = document.getElementById("gh-user-preview");

        if (!input) {
            if (statusEl) {
                statusEl.style.display = "block";
                statusEl.style.color = "#f87171";
                statusEl.innerHTML = "⚠️ Please enter a GitHub Profile URL or Username.";
            }
            return;
        }

        if (statusEl) {
            statusEl.style.display = "block";
            statusEl.style.color = "var(--glow-cyan)";
            statusEl.innerHTML = '<span class="material-symbols-outlined" style="vertical-align:middle; animation:spin 1s linear infinite;">sync</span> Connecting to GitHub API and syncing portfolio...';
        }
        if (btn) btn.disabled = true;

        const res = await apiPost("/api/github/fetch-profile", { profile_input: input });
        if (btn) btn.disabled = false;

        if (!res || res.error) {
            if (statusEl) {
                statusEl.style.display = "block";
                statusEl.style.color = "#f87171";
                statusEl.innerHTML = `❌ ${res?.error || "Failed to fetch GitHub profile."}`;
            }
            return;
        }

        // Show Success
        if (statusEl) {
            statusEl.style.display = "block";
            statusEl.style.color = "var(--glow-emerald)";
            statusEl.innerHTML = `✅ Successfully fetched @${res.username}! Form auto-populated & ML evaluation rendered.`;
        }

        const m = res.metrics || {};

        // Fill Preview Card
        if (previewCard) {
            previewCard.style.display = "flex";
            const avatarEl = document.getElementById("gh-preview-avatar");
            if (avatarEl) avatarEl.src = res.avatar_url || "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png";
            
            const nameEl = document.getElementById("gh-preview-name");
            if (nameEl) nameEl.textContent = res.display_name || res.username;
            
            const loginEl = document.getElementById("gh-preview-login");
            if (loginEl) loginEl.textContent = `@${res.username}`;
            
            const bioEl = document.getElementById("gh-preview-bio");
            if (bioEl) bioEl.textContent = res.bio || (res.company ? `🏢 ${res.company}` : (res.location ? `📍 ${res.location}` : "Public GitHub Developer Profile"));
            
            const linkEl = document.getElementById("gh-preview-link");
            if (linkEl) linkEl.href = res.html_url || `https://github.com/${res.username}`;

            // Update stats row badges with actual fetched numbers
            const badgeRepos = document.getElementById("gh-badge-repos");
            if (badgeRepos) badgeRepos.textContent = `📦 ${m.public_repos ?? 0} Repos`;
            
            const badgeStars = document.getElementById("gh-badge-stars");
            if (badgeStars) badgeStars.textContent = `⭐ ${m.total_stars ?? 0} Stars`;
            
            const badgeFollowers = document.getElementById("gh-badge-followers");
            if (badgeFollowers) badgeFollowers.textContent = `👥 ${m.followers ?? 0} Followers`;
            
            const badgeFocus = document.getElementById("gh-badge-focus");
            if (badgeFocus) badgeFocus.textContent = `🎯 ${m.focus_area || "Web Development"}`;
            
            const badgeLangs = document.getElementById("gh-badge-langs");
            if (badgeLangs) badgeLangs.textContent = `💻 ${m.languages_used || "Languages"}`;
        }

        // Auto-fill all form inputs
        document.getElementById("gh-repos").value = m.public_repos ?? 0;
        document.getElementById("gh-followers").value = m.followers ?? 0;
        document.getElementById("gh-following").value = m.following ?? 0;
        document.getElementById("gh-stars").value = m.total_stars ?? 0;
        document.getElementById("gh-forks").value = m.total_forks ?? 0;
        document.getElementById("gh-contribs").value = m.contributions_last_year ?? 0;
        document.getElementById("gh-streak").value = m.longest_streak_days ?? 0;
        document.getElementById("gh-readme").value = m.readme_coverage_percentage ?? 0;
        document.getElementById("gh-pinned").value = m.pinned_repos_count ?? 0;
        document.getElementById("gh-top-stars").value = m.top_repo_stars ?? 0;
        document.getElementById("gh-bio").checked = Boolean(m.has_bio);
        document.getElementById("gh-os").value = m.open_source_contributions ?? 0;
        document.getElementById("gh-score").value = m.profile_score ?? 0;
        document.getElementById("gh-langs").value = m.languages_used || "";
        
        const focusSel = document.getElementById("github-focus");
        if (focusSel && m.focus_area) {
            focusSel.value = m.focus_area;
        }

        // Automatically render the ML model rating result card
        if (res.ml_evaluation) {
            renderGitHubResult(res.ml_evaluation);
        }
    });

    // 6. GitHub Manual Submit Form
    document.getElementById("form-github")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            public_repos: getNum("gh-repos", 10),
            followers: getNum("gh-followers", 20),
            following: getNum("gh-following", 20),
            total_stars: getNum("gh-stars", 30),
            total_forks: getNum("gh-forks", 10),
            contributions_last_year: getNum("gh-contribs", 200),
            longest_streak_days: getNum("gh-streak", 15),
            readme_coverage_percentage: getFloat("gh-readme", 70.0),
            pinned_repos_count: getNum("gh-pinned", 2),
            top_repo_stars: getNum("gh-top-stars", 15),
            has_bio: document.getElementById("gh-bio")?.checked || false,
            open_source_contributions: getNum("gh-os", 5),
            profile_score: getFloat("gh-score", 70.0),
            languages_used: getStr("gh-langs", "Python, JavaScript"),
            focus_area: getStr("github-focus", "Web Development"),
        };
        const res = await apiPost("/api/predict/github", payload);
        if (res) renderGitHubResult(res);
    });

    // 7. Salary
    document.getElementById("form-salary")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            years_experience: getFloat("sal-exp", 3.0),
            skills_count: getNum("sal-skills", 5),
            certifications_count: getNum("sal-certs", 1),
            industry: getStr("salary-industry", "Technology"),
            job_title: getStr("salary-job-title", "Data Scientist"),
            education_level: getStr("salary-edu", "Bachelor's Degree"),
            degree_field: getStr("salary-degree", "Computer Science"),
            country: getStr("salary-country", "India"),
            company_size: getStr("salary-company-size", "Medium"),
            work_type: getStr("salary-work-type", "Hybrid"),
        };
        const res = await apiPost("/api/predict/salary", payload);
        if (res) renderSalaryResult(res);
    });

    // 8. Career Rec
    document.getElementById("form-career")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const selectedChips = Array.from(document.querySelectorAll(".interest-chip.selected"))
            .map(c => c.getAttribute("data-value"));
        const payload = {
            years_experience: getFloat("career-exp", 0.0),
            current_skills: getStr("career-skills", "Python, Problem Solving, SQL"),
            work_style: getStr("career-work-style", "Hybrid"),
            recommended_industry: getStr("career-industry", "Information Technology"),
            education_level: getStr("career-edu", "Bachelor's Degree"),
            selected_interests: selectedChips.length > 0 ? selectedChips : ["Technology", "Problem Solving"],
            top_k: parseInt(document.querySelector('input[name="career-topk"]:checked')?.value || 5),
        };
        const res = await apiPost("/api/predict/career-recommendations", payload);
        if (res) renderCareerResult(res);
    });
}

// REST Helper
async function apiPost(url, payload) {
    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        alert("Error executing prediction: " + err.message);
        console.error(err);
        return null;
    }
}

// Result Renderers
function renderResumeResult(data) {
    const container = document.getElementById("resume-result");
    container.style.display = "block";
    document.getElementById("ats-score-display").textContent = data.ats_score;
    document.getElementById("ats-category-display").textContent = data.category;
    document.getElementById("ats-skills-cnt").textContent = data.skills_detected;
    document.getElementById("ats-certs-cnt").textContent = data.certifications_detected;
    document.getElementById("ats-missing-cnt").textContent = data.missing_keywords_count;
    container.scrollIntoView({ behavior: "smooth" });
}

function renderSkillgapResult(data, req) {
    const container = document.getElementById("skillgap-result");
    if (!container) return;
    container.style.display = "block";

    // Titles & Metrics
    document.getElementById("sg-header-title").textContent = `${data.target_role || "Target Role"} Career Readiness Report`;
    document.getElementById("sg-header-subtitle").textContent = `Transitioning from ${data.current_role || "Current Role"} to ${data.target_role || "Target Role"}.`;

    document.getElementById("skillgap-months-display").textContent = `${data.estimated_months} Months`;
    document.getElementById("skillgap-readiness-display").textContent = `${data.readiness_percentage}%`;
    document.getElementById("skillgap-gap-display").textContent = `${data.skill_gap_count} Skills Missing`;

    // 1. Verified Strengths (Matched Skills)
    const matchedEl = document.getElementById("sg-matched-badges");
    if (matchedEl) {
        matchedEl.innerHTML = "";
        const matched = data.matched_skills || [];
        if (matched.length > 0) {
            matched.forEach(sk => {
                const badge = document.createElement("span");
                badge.className = "badge-pill badge-green";
                badge.innerHTML = `✓ ${sk}`;
                matchedEl.appendChild(badge);
            });
        } else {
            matchedEl.innerHTML = `<span style="font-size:0.85rem; color:#94a3b8;">No direct overlap with target role benchmarks yet. Start with core fundamentals!</span>`;
        }
    }

    // 2. Target Benchmark Skills
    const benchEl = document.getElementById("sg-benchmark-badges");
    if (benchEl) {
        benchEl.innerHTML = "";
        const bench = data.benchmark_required_skills || [];
        bench.forEach(sk => {
            const isMatched = (data.matched_skills || []).includes(sk);
            const badge = document.createElement("span");
            badge.className = isMatched ? "badge-pill badge-green" : "badge-pill badge-blue";
            badge.textContent = sk;
            benchEl.appendChild(badge);
        });
    }

    // 3. Missing Skills with Recommended Learning Resources
    const resEl = document.getElementById("sg-resources-list");
    if (resEl) {
        resEl.innerHTML = "";
        const missingRes = data.missing_skills_with_resources || [];
        if (missingRes.length > 0) {
            missingRes.forEach((item, idx) => {
                const itemCard = document.createElement("div");
                itemCard.style.padding = "0.85rem 1rem";
                itemCard.style.background = "rgba(255,255,255,0.03)";
                itemCard.style.border = "1px solid rgba(255,255,255,0.08)";
                itemCard.style.borderRadius = "var(--radius-sm)";
                itemCard.style.display = "flex";
                itemCard.style.justifyContent = "space-between";
                itemCard.style.alignItems = "center";
                itemCard.style.flexWrap = "wrap";
                itemCard.style.gap = "0.5rem";

                itemCard.innerHTML = `
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span class="badge-pill badge-purple" style="font-size:0.75rem;">Priority ${idx + 1}</span>
                        <strong style="color:#fff; font-size:0.95rem;">${item.skill}</strong>
                    </div>
                    <div style="font-size:0.85rem; color:var(--glow-cyan); display:flex; align-items:center; gap:0.3rem;">
                        <span class="material-symbols-outlined" style="font-size:1rem;">auto_stories</span>
                        ${item.resource}
                    </div>
                `;
                resEl.appendChild(itemCard);
            });
        } else {
            resEl.innerHTML = `<div style="color:var(--glow-emerald); font-size:0.9rem; padding:0.5rem 0;">🎉 Outstanding! You already possess all standard core competencies for this role!</div>`;
        }
    }

    container.scrollIntoView({ behavior: "smooth" });
}

function renderRoadmapResult(data, req) {
    const container = document.getElementById("roadmap-result");
    container.style.display = "block";
    document.getElementById("roadmap-months-display").textContent = data.total_duration_months + " Months";
    document.getElementById("roadmap-weeks-display").textContent = data.total_weeks + " Weeks";
    document.getElementById("roadmap-hours-display").textContent = data.total_study_hours + " Hours";

    // 1. Populate Multi-Phase Curriculum Timeline
    const timelineEl = document.getElementById("roadmap-phases-timeline");
    if (timelineEl) {
        timelineEl.innerHTML = "";
        const phases = data.phases || [];
        if (phases.length > 0) {
            phases.forEach(p => {
                const phaseCard = document.createElement("div");
                phaseCard.className = "m3-card";
                phaseCard.style.padding = "1.25rem";
                phaseCard.style.borderLeft = "4px solid var(--glow-cyan)";
                phaseCard.style.background = "linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.65) 100%)";
                
                const topicsTags = (p.key_topics || []).map(t => `<span class="badge-pill badge-blue" style="font-size:0.78rem;">${t}</span>`).join(" ");
                
                phaseCard.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.5rem;">
                        <div>
                            <span class="badge-pill badge-green" style="font-size:0.8rem; margin-bottom:0.35rem;">Phase ${p.phase_number}: ${p.level}</span>
                            <h4 style="color:#fff; margin:0.35rem 0; font-size:1.05rem;">${p.title}</h4>
                        </div>
                        <span class="badge-pill badge-purple">${p.badge}</span>
                    </div>
                    <p style="color:var(--md-sys-color-on-surface-variant); font-size:0.9rem; margin:0.4rem 0 0.75rem; line-height:1.45;">
                        ${p.details}
                    </p>
                    <div style="display:flex; gap:0.4rem; flex-wrap:wrap; align-items:center;">
                        <span style="font-size:0.8rem; color:#94a3b8; font-weight:600;">Key Topics:</span>
                        ${topicsTags}
                    </div>
                `;
                timelineEl.appendChild(phaseCard);
            });
        } else {
            timelineEl.innerHTML = `<p style="color:#94a3b8;">Custom structured roadmap active for ${data.target_role || "target role"}.</p>`;
        }
    }

    // 2. Populate Milestones Target
    const milestonesEl = document.getElementById("roadmap-milestones-list");
    if (milestonesEl) {
        milestonesEl.innerHTML = "";
        const ms = data.milestones || [];
        if (ms.length > 0) {
            ms.forEach(item => {
                const li = document.createElement("li");
                li.style.marginBottom = "0.5rem";
                li.innerHTML = `<strong>${item}</strong>`;
                milestonesEl.appendChild(li);
            });
        } else {
            milestonesEl.innerHTML = `<li>Month 3: Complete Foundation Fundamentals</li><li>Month 6: Build 1st Full Application</li><li>Month ${Math.round(data.total_duration_months)}: Ready for ${data.target_role || "target"} applications</li>`;
        }
    }

    // 3. Populate Recommended Projects
    const projectsEl = document.getElementById("roadmap-projects-list");
    if (projectsEl) {
        projectsEl.innerHTML = "";
        const projs = data.recommended_projects || [];
        if (projs.length > 0) {
            projs.forEach(item => {
                const li = document.createElement("li");
                li.style.marginBottom = "0.5rem";
                li.innerHTML = `🚀 ${item}`;
                projectsEl.appendChild(li);
            });
        } else {
            projectsEl.innerHTML = `<li>🚀 Full Stack Portfolio Application</li><li>🚀 Cloud Deployed Microservice</li><li>🚀 Capstone Showcase Project</li>`;
        }
    }

    // 4. Populate Certifications
    const certsEl = document.getElementById("roadmap-certs-badges");
    if (certsEl) {
        certsEl.innerHTML = "";
        const certs = data.top_certifications || [];
        if (certs.length > 0) {
            certs.forEach(c => {
                const span = document.createElement("span");
                span.className = "badge-pill badge-amber";
                span.textContent = `📜 ${c}`;
                certsEl.appendChild(span);
            });
        } else {
            certsEl.innerHTML = `<span class="badge-pill badge-amber">📜 Professional Industry Certificate</span>`;
        }
    }

    container.scrollIntoView({ behavior: "smooth" });
}

function renderInterviewQuestionBank(data) {
    const container = document.getElementById("interview-bank-container");
    if (!container) return;
    container.style.display = "block";

    // Set titles and badges
    document.getElementById("interview-bank-title").textContent = `${data.job_title} Comprehensive Question Bank`;
    document.getElementById("interview-bank-subtitle").textContent = `Displaying ${data.display_count} of ${data.total_questions_in_bank} available interview questions for ${data.job_title} in ${data.industry}.`;
    
    document.getElementById("q-total-badge").textContent = `📦 ${data.display_count} Questions`;
    
    const types = data.type_breakdown || {};
    document.getElementById("q-tech-badge").textContent = `💻 ${types.Technical || 0} Technical`;
    document.getElementById("q-behav-badge").textContent = `🧠 ${types.Behavioral || 0} Behavioral`;
    document.getElementById("q-sit-badge").textContent = `🎯 ${types.Situational || 0} Situational`;

    // Populate question cards
    const listEl = document.getElementById("interview-questions-list");
    listEl.innerHTML = "";

    const questions = data.questions || [];
    if (questions.length === 0) {
        listEl.innerHTML = `<div class="m3-card" style="padding:1.5rem; text-align:center; color:#94a3b8;">No interview questions matched your specific filter. Try selecting 'All' categories or clearing search keywords.</div>`;
        container.scrollIntoView({ behavior: "smooth" });
        return;
    }

    questions.forEach((q, idx) => {
        const card = document.createElement("div");
        card.className = "m3-card";
        card.style.padding = "1.5rem";
        card.style.borderLeft = "4px solid " + (q.difficulty_level === "Hard" ? "var(--glow-purple)" : (q.difficulty_level === "Medium" ? "var(--glow-cyan)" : "var(--glow-emerald)"));
        card.style.background = "linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.65) 100%)";

        // Badges
        const diffClass = q.difficulty_level === "Hard" ? "badge-purple" : (q.difficulty_level === "Medium" ? "badge-blue" : "badge-green");
        const typeClass = q.question_type === "Technical" ? "badge-blue" : (q.question_type === "Behavioral" ? "badge-purple" : "badge-amber");

        const evalBullets = (q.key_evaluation_points || []).map(pt => `<li>${pt}</li>`).join("");

        card.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.75rem;">
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
                    <span class="badge-pill ${typeClass}">${q.question_type}</span>
                    <span class="badge-pill ${diffClass}">${q.difficulty_level} Difficulty</span>
                    <span style="font-size:0.8rem; color:#94a3b8;">⏱️ ~${q.ideal_answer_length_words || 200} words ideal answer</span>
                </div>
                <span style="font-weight:700; color:#64748b; font-size:0.9rem;">#${idx + 1}</span>
            </div>

            <h3 style="color:#fff; font-size:1.1rem; margin:0.4rem 0 1rem; line-height:1.45;">
                ${q.question_text}
            </h3>

            <!-- Evaluation Criteria Box -->
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:var(--radius-sm); padding:0.85rem 1rem; margin-bottom:0.75rem;">
                <div style="font-size:0.82rem; font-weight:700; color:var(--glow-cyan); margin-bottom:0.35rem; display:flex; align-items:center; gap:0.3rem;">
                    <span class="material-symbols-outlined" style="font-size:1rem;">checklist</span> What the Hiring Manager Evaluates:
                </div>
                <ul style="margin:0; padding-left:1.2rem; font-size:0.85rem; color:#cbd5e1; line-height:1.4;">
                    ${evalBullets || "<li>Demonstrates technical understanding and clear structure.</li>"}
                </ul>
            </div>
        `;
        listEl.appendChild(card);
    });

    // Handle Load More Button
    const loadMoreContainer = document.getElementById("load-more-container");
    const loadMoreBtn = document.getElementById("btn-load-more-q");
    if (loadMoreContainer && loadMoreBtn) {
        if (data.display_count < data.total_questions_in_bank) {
            loadMoreContainer.style.display = "block";
            loadMoreBtn.onclick = async () => {
                const currentCount = data.display_count;
                const nextLimit = currentCount + 20;
                const payload = {
                    industry: data.industry,
                    job_title: data.job_title,
                    question_type: getStr("interview-type-filter", "All"),
                    difficulty_level: getStr("interview-diff-filter", "All"),
                    limit: nextLimit,
                    search_query: getStr("interview-search-input", "").trim(),
                };
                const nextRes = await apiPost("/api/interview/questions", payload);
                if (nextRes) renderInterviewQuestionBank(nextRes);
            };
        } else {
            loadMoreContainer.style.display = "none";
        }
    }

    container.scrollIntoView({ behavior: "smooth" });
}

function renderLinkedInResult(data) {
    const container = document.getElementById("linkedin-result");
    container.style.display = "block";
    document.getElementById("linkedin-rating-display").textContent = data.predicted_rating;
    container.scrollIntoView({ behavior: "smooth" });
}

function renderGitHubResult(data) {
    const container = document.getElementById("github-result");
    container.style.display = "block";
    document.getElementById("github-rating-display").textContent = data.predicted_rating;
    document.getElementById("github-stars-display").textContent = data.total_stars;
    document.getElementById("github-contribs-display").textContent = data.contributions_last_year;
    container.scrollIntoView({ behavior: "smooth" });
}

function renderSalaryResult(data) {
    const container = document.getElementById("salary-result");
    container.style.display = "block";
    document.getElementById("salary-local-display").textContent = data.formatted_local_salary;
    document.getElementById("salary-usd-display").textContent = "$" + Math.round(data.predicted_salary_usd).toLocaleString() + " USD";
    document.getElementById("salary-monthly-local").textContent = data.currency_symbol + Math.round(data.monthly_salary_local).toLocaleString() + " /mo";
    document.getElementById("salary-col-display").textContent = data.cost_of_living_index;
    document.getElementById("salary-ppp-display").textContent = "$" + Math.round(data.ppp_adjusted_salary_usd).toLocaleString() + " USD";
    container.scrollIntoView({ behavior: "smooth" });
}

function renderCareerResult(data) {
    const container = document.getElementById("career-result");
    if (!container) return;
    container.style.display = "block";

    const badge = document.getElementById("career-match-badge");
    if (badge) {
        badge.textContent = `🎯 Match Score: ${data.match_score || 85}%`;
    }

    const list = document.getElementById("career-recs-list");
    list.innerHTML = "";

    const recs = data.top_recommendations || [];
    if (recs.length === 0) {
        list.innerHTML = `<div class="m3-card" style="padding:1.5rem; text-align:center; color:#94a3b8;">No direct career recommendations matched. Try selecting more interest chips!</div>`;
        container.scrollIntoView({ behavior: "smooth" });
        return;
    }

    recs.forEach(rec => {
        const card = document.createElement("div");
        card.className = "m3-card";
        card.style.padding = "1.25rem 1.5rem";
        card.style.borderLeft = "4px solid " + (rec.rank === 1 ? "var(--glow-emerald)" : (rec.rank <= 3 ? "var(--glow-cyan)" : "var(--glow-purple)"));
        card.style.background = "linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.65) 100%)";

        const badgeClass = rec.rank === 1 ? "badge-green" : (rec.rank <= 3 ? "badge-blue" : "badge-purple");

        card.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.5rem;">
                <div style="display:flex; align-items:center; gap:0.6rem;">
                    <span class="badge-pill ${badgeClass}" style="font-weight:700;">#${rec.rank} Match</span>
                    <h3 style="color:#fff; font-size:1.15rem; margin:0;">${rec.career_title}</h3>
                </div>
                <span class="badge-pill badge-green" style="font-size:0.9rem; font-weight:700;">${rec.confidence_percentage}% Confidence</span>
            </div>
            
            <p style="color:#cbd5e1; font-size:0.88rem; margin:0.4rem 0 0; line-height:1.45;">
                💡 ${rec.reasoning || "Strong alignment with your skills and chosen industry interest domains."}
            </p>
        `;
        list.appendChild(card);
    });

    container.scrollIntoView({ behavior: "smooth" });
}
