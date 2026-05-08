# KODO Student Dropout Prediction System

Machine Learning based student dropout prediction system developed during my internship at KODO.

## Overview

This project predicts students who are at risk of dropping out using behavioral and engagement data collected from the learning platform.

The system analyzes:
- Login frequency
- Quiz scores
- Session duration
- Module completion rate
- Activity consistency

The goal is to help educators identify at-risk students early and improve student retention.

---

## Features

- Student dropout prediction using Machine Learning.
- Random Forest and XGBoost models.
- Real-time prediction through Flask REST API.
- Class imbalance handling using SMOTE.
- Exploratory Data Analysis (EDA) and visualization.
- Risk scoring integration-ready backend.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- Matplotlib
- Seaborn
- SMOTE (imbalanced-learn)

---

## Model Performance

| Model | F1-Score |
|-------|-----------|
| Random Forest | 87% |
| XGBoost | 87% |

Key findings:
- Quiz inactivity strongly correlated with dropout risk.
- Session drop-off patterns were major indicators of disengagement.

---

## Project Workflow

1. Data Collection & Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Handling Class Imbalance using SMOTE
5. Model Training & Evaluation
6. Flask API Deployment
7. Dashboard Integration

---

## Folder Structure

```bash
├── data/
├── notebooks/
├── models/
├── api/
├── static/
├── templates/
├── requirements.txt
├── app.py
└── README.md
