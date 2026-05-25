# KODO Student Dropout Prediction

> **Prototype of the ML system I built during my internship at KODO (Nov 2025 – Feb 2026)**
> This repository contains the full end-to-end pipeline — from data generation and model training to a live Flask API and instructor dashboard.

---

## What is this?

KODO is a study platform where students take courses, submit assignments, watch videos, and interact with instructors through forums. The problem KODO faced was simple — **instructors had no way to know which students were silently disengaging before it was too late.**

During my internship I was tasked with building a machine learning system that could look at a student's engagement behaviour on the platform and predict whether they were at risk of dropping out. This repository is the prototype of that system.

The model looks at 9 engagement signals and outputs a risk score — which gets displayed on the instructor dashboard so they can reach out to at-risk students before they churn.

---

## Live Demo

> **Dashboard:** *(add your Render URL here after deployment)*
> **API:** *(add your Render API URL here)*

---

## The Problem

Instructors on KODO were checking student progress manually which doesn't scale. A single instructor managing 100+ students can't individually monitor who is becoming inactive, who is struggling with quizzes, or who hasn't submitted assignments in weeks.

The goal was to automate this — flag students who show early signs of dropout so instructors can intervene at the right time.

---

## What I Built

### 1. Data Pipeline
Since real student data wasn't available at the prototype stage, I generated synthetic student data designed to mimic real platform behaviour. The dataset has 1,200 students with 9 features each.

The 9 features used:

| Feature | What it captures |
|---|---|
| `logins_per_week` | How often the student visits the platform |
| `avg_time_per_session` | How long they stay when they do visit |
| `quiz_score_avg` | Their average score across all quizzes |
| `modules_completed` | How many course modules they finished |
| `days_since_login` | How many days since they last logged in |
| `revisit_count` | How often they go back and review old content |
| `assignment_submissions` | How many assignments they submitted |
| `forum_activity` | How many times they posted or replied in forums |
| `video_watch_pct` | What percentage of course videos they watched |

A student is flagged **at risk** if they show any of these patterns:
- Quiz average below 40% (serious academic failure)
- Inactive for 15+ days AND completed fewer than 8 modules
- Logging in less than 2 times a week AND submitting fewer than 3 assignments AND watching less than 30% of videos

### 2. Model Training
Trained and compared 3 models — Logistic Regression, Random Forest, and XGBoost.

**Key steps in the pipeline:**
- Applied **SMOTE** to fix class imbalance (only on training data, never on test)
- Used **StandardScaler** for feature scaling
- Used **GridSearchCV** with 5-fold **StratifiedKFold** cross validation to tune XGBoost
- Tried 108 hyperparameter combinations to find the best settings

**Final model results (Tuned XGBoost):**

| Metric | Score |
|---|---|
| F1 Score | **0.93** |
| ROC-AUC | **0.978** |
| CV Mean F1 | **0.9273** |
| CV Std Dev | **0.0081** |

**Feature importance (what the model cares about most):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | video_watch_pct | 33.4% |
| 2 | assignment_submissions | 19.9% |
| 3 | forum_activity | 14.5% |
| 4 | revisit_count | 11.4% |
| 5 | avg_time_per_session | 5.6% |
| 6 | quiz_score_avg | 4.6% |
| 7 | modules_completed | 3.9% |
| 8 | logins_per_week | 3.4% |
| 9 | days_since_login | 3.3% |

### 3. Flask REST API
Wrapped the model in a Flask API with two endpoints so the KODO platform can call it programmatically.

**`POST /predict`** — single student prediction
```json
// request
{
  "logins_per_week": 2,
  "avg_time_per_session": 20,
  "quiz_score_avg": 35,
  "modules_completed": 3,
  "days_since_login": 18,
  "revisit_count": 1,
  "assignment_submissions": 1,
  "forum_activity": 0,
  "video_watch_pct": 12
}

// response
{
  "risk_label": "AT RISK",
  "risk_probability": 0.9944,
  "risk_percent": "99.4%",
  "message": "High risk — this student needs immediate attention"
}
```

**`POST /predict-batch`** — check an entire class at once

### 4. Instructor Dashboard
Built a clean web dashboard (`index.html`) so instructors can use the system without touching any code.

**Features:**
- Single student check with name and ID fields
- 3-level color coded risk — 🔴 High (≥75%) / 🟡 Medium (50–74%) / 🟢 Low (<50%)
- Prediction history with bar chart showing session summary
- Batch CSV upload to check an entire class at once
- Donut chart showing risk distribution across the class
- Filter results by risk level (All / High / Medium / Low only)
- Export results as CSV
- Sample CSV download so instructors know the required format

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, NumPy, Pandas, Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| API | Flask, Flask-CORS, Joblib |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Dev Tools | Jupyter Notebook, Spyder |

---

## Project Structure

```
kodo-dropout-prediction/
├── app.py                           # Flask REST API
├── index.html                       # Instructor dashboard
├── kodo_dropout_prediction.ipynb    # Full ML notebook (data → model → save)
├── kodo_xgboost_model.pkl           # Trained XGBoost model
├── kodo_scaler.pkl                  # StandardScaler (must be used with model)
├── kodo_students_v2.csv             # Synthetic training dataset (1200 students)
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/kodo-dropout-prediction.git
cd kodo-dropout-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the Flask API**
```bash
python app.py
# server starts at http://127.0.0.1:5000
```

**4. Open the dashboard**

Open `index.html` in your browser. That's it.

**5. Test the API directly**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"logins_per_week":2,"avg_time_per_session":10,"quiz_score_avg":30,"modules_completed":1,"days_since_login":20,"revisit_count":0,"assignment_submissions":1,"forum_activity":0,"video_watch_pct":10}'
```

---

## Important Notes for Connecting Real Data

This is a **prototype built on synthetic data**. When real KODO student data becomes available:

1. Replace Cell 1 in the notebook with real data loading
2. Make sure the real dataset has the same 9 feature column names
3. Retrain from Cell 4 onwards — the pipeline stays exactly the same
4. Save new `.pkl` files and replace the existing ones

The model's F1 score on real data will likely settle around **0.85–0.92** since real students behave less predictably than synthetic ones — which is actually a more honest result.

---

## What I Learned

- How to handle **class imbalance** in real ML pipelines (SMOTE only on train, never test)
- How **data leakage** silently inflates model scores and how to catch it
- Why **feature importance** matters — video watch time and assignment submissions turned out to be stronger dropout signals than quiz scores alone
- End-to-end ML deployment — from Jupyter notebook to a live REST API

---

## About

Built by **Nishant Choudhary** during ML internship at **KODO** (Nov 2025 – Feb 2026)

B.Tech CSE — Data Science Specialization | Manipal University Jaipur

[GitHub](https://github.com/ni-saantt) · [LinkedIn](https://linkedin.com/in/ni-santt)
