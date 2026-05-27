import pandas as pd
import numpy as np
import json

df = pd.read_csv('kodo_students_v2.csv')

total_students = len(df)
at_risk_rate = float(df['at_risk'].mean())
avg_quiz = float(df['quiz_score_avg'].mean())
avg_video = float(df['video_watch_pct'].mean())

print(f"Total: {total_students}")
print(f"At-risk rate: {at_risk_rate:.4f}")
print(f"Avg Quiz Score: {avg_quiz:.4f}")
print(f"Avg Video Watch: {avg_video:.4f}")

# Engagement groups: logins_per_week (0-2, 3-4, 5-7, 8-10, 11-13)
bins = [-1, 2, 4, 7, 10, 13]
labels = ['0–2', '3–4', '5–7', '8–10', '11–13']
df['login_grp'] = pd.cut(df['logins_per_week'], bins=bins, labels=labels)
risk_by_login = df.groupby('login_grp', observed=False)['at_risk'].mean().to_dict()
print("Risk by login per week:")
for k, v in risk_by_login.items():
    print(f"  {k}: {v*100:.2f}%")

# Cohort Risk Breakdown (Low: logins <= 3, Mid: logins 4-7, High: logins >= 8)
def get_cohort(logins):
    if logins <= 3: return 'Low Engagement'
    elif logins <= 7: return 'Mid Engagement'
    else: return 'High Engagement'

df['cohort'] = df['logins_per_week'].apply(get_cohort)
cohort_risk = df.groupby(['cohort', 'at_risk']).size().unstack(fill_value=0)
print("\nCohort Risk:")
print(cohort_risk)
