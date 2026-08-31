# AI Career Mentor — Trained Models (README)

Yeh package AI Career Mentor app ke liye 8 trained ML models + data + notebook contain karta hai.
Sab 8 models **200,000-row** datasets par train hue hain aur test set par verify kiye gaye hain.

---

## Files in this package

| File | Kya hai |
|---|---|
| `01_resume_ats_model.pkl` | Resume ATS score predictor (regression) |
| `02_skillgap_model.pkl` | Skill-gap ke liye estimated months predictor (regression) |
| `03_roadmap_model.pkl` | Career roadmap ki total duration predictor (regression) |
| `04_interview_model.pkl` | Interview question difficulty classifier (⚠️ neeche note dekho) |
| `05_linkedin_model.pkl` | LinkedIn profile review-rating classifier |
| `06_github_model.pkl` | GitHub profile review-rating classifier |
| `07_salary_prediction_model.pkl` | Salary predictor — **real internet data se enrich kiya hua** |
| `08_career_recommendation_model.pkl` | Career recommendation (63 classes) — top-3/top-5 shortlist ke liye |
| `career_utils.py` | Feature-engineering helper functions (sab models isse use karte hain) |
| `country_data.py` | **Real** country economic + currency data (Numbeo + live FX rates) |
| `07_salary_prediction_dataset_enriched.csv` | Salary dataset + real country data columns |
| `*_metrics.json`, `all_metrics.json` | Har model ke test-set results |
| `AI_Career_Mentor_Model_Training.ipynb` | Colab notebook — sab kuch from-scratch reproduce karta hai |

---

## Model results (test set par, 20% held-out)

| Model | Task | Metric | Result |
|---|---|---|---|
| Resume ATS Score | Regression | R² | **0.935** |
| Skill Gap (months) | Regression | R² | **0.771** |
| Roadmap Duration | Regression | R² | **0.900** |
| Interview Difficulty | Classification | Accuracy | 0.399 (⚠️ baseline ke barabar) |
| LinkedIn Rating | Classification | Accuracy | **0.994** |
| GitHub Rating | Classification | Accuracy | **0.997** |
| Salary Prediction | Regression | R² | **0.984** |
| Career Recommendation | Multiclass (63) | Top-1 / Top-3 / Top-5 | 0.163 / **0.502** / **0.830** |

### ⚠️ Interview Difficulty model — limitation
Is model ki accuracy (0.399) exactly majority-class baseline ke barabar hai. Iska matlab: is dataset mein
`difficulty_level` column baaki sab features (industry, job_title, question_type, answer length) se
**independent/random** generate hua hai — koi real learnable pattern nahi hai. Model sahi kaam kar raha hai,
lekin data mein signal hi nahi hai. Isko fix karne ke liye source dataset ko is tarah regenerate karna hoga
ki difficulty question ki characteristics par depend kare.

### Career Recommendation model — kaise use karein
63 classes ke saath top-1 accuracy naturally kam hoti hai. App mein iska use **shortlist recommender**
ki tarah karo — `model.predict_proba()` se top-3 ya top-5 careers dikhao (jinki combined accuracy 50%
aur 83% hai), na ki sirf ek single best guess.

---

## Real internet data — kya add kiya gaya (salary dataset)

Baaki 7 datasets fully synthetic hain (koi "real internet data" ka genuine signal nahi tha unme),
isliye sirf **salary prediction dataset** ko real data se enrich kiya:

- **Cost of Living Index** aur **Local Purchasing Power Index** — [Numbeo](https://www.numbeo.com/cost-of-living/rankings_by_country.jsp) (2026 mid-year data), sabhi 15 countries ke liye.
- **Live currency exchange rates** (USD base) — x-rates.com / OFX se, Aug 2026 snapshot.
- Naye columns: `cost_of_living_index`, `purchasing_power_index`, `currency_code`, `exchange_rate_usd`,
  `predicted_salary_local_currency`, `ppp_adjusted_salary_usd`.

### India ke liye local currency (jaisa specifically maanga gaya tha)
Model salary ko **USD mein predict karta hai** (consistent training target ke liye), phir
`country_data.py` ka `to_local_currency()` function use karke real exchange rate se local currency
mein convert karta hai:

```python
import joblib
from country_data import to_local_currency

bundle = joblib.load('07_salary_prediction_model.pkl')
model = bundle['model']

usd_salary = model.predict(row)[0]          # e.g. 23,647 USD
inr_salary = to_local_currency(usd_salary, 'India')   # -> ₹ ~22,55,000 INR (rate: 1 USD ≈ 95.4 INR)
```

Sabhi 15 countries supported hain: Australia, Brazil, Canada, Germany, India, Japan, Nigeria, Pakistan,
Philippines, Poland, Singapore, South Africa, UAE, United Kingdom, United States.

**Zaroori baat:** exchange rates roz badalte hain. Yeh Aug 2026 ka snapshot hai — production app mein
`country_data.py` ke `EXCHANGE_RATE_TO_USD` dict ko live FX API (Xe, Wise, exchangerate.host) se refresh
karna better hoga.

---

## Notebook (`.ipynb`) — Colab mein khud train karna ho to

1. `AI_Career_Mentor_Model_Training.ipynb` ko Google Colab mein upload karo.
2. **Runtime → Run all.**
3. Jab prompt aaye, sabhi 8 original CSV datasets upload karo (multi-select).
4. Notebook automatically: real data se salary dataset enrich karega → sab 8 models train karega →
   metrics print karega → sab `.pkl` + report zip karke download kara dega.
5. Poora chalne mein Colab ke free CPU par ~5–10 minute lagte hain.

Notebook maine khud end-to-end run karke verify kiya hai — bina kisi error ke chalta hai, aur exactly
wahi metrics deta hai jo upar table mein diye hain.

---

## Feature engineering approach (sab models mein common)

- Categorical columns → label-encoded (`career_utils.encode_categorical`)
- Multi-value text columns (skills, interests, etc.) → count-based features (`career_utils.count_items`)
- Ordinal columns (education level, difficulty, rating) → manually ordered encoding
- Model: `HistGradientBoostingRegressor` / `Classifier` (scikit-learn) — 200k rows ke liye fast aur accurate
- Text/leakage-prone columns (jaise `reasoning`, `strengths`, `improvement_suggestions` jo target ka
  explanation hote hain) **jaan-boojh kar drop** kiye gaye taaki model asli signal se sikhe, na ki
  target ke text-based hints se.
