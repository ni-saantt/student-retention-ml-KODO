# KODO DROPOUT PREDICTION — FLASK API
# Connects the trained model with the KODO dashboard


from flask import Flask, request, jsonify
import joblib
import numpy as np
import csv

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# load once at startup to avoid reloading on every request
model  = joblib.load('kodo_xgboost_model.pkl')
scaler = joblib.load('kodo_scaler.pkl')

print("model loaded successfully")
print("scaler loaded successfully")

TRAINING_STATS = None

def calculate_training_stats():
    global TRAINING_STATS
    try:
        students_data = []
        with open('kodo_students_v2.csv', mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                students_data.append({
                    'logins_per_week': float(row['logins_per_week']),
                    'avg_time_per_session': float(row['avg_time_per_session']),
                    'quiz_score_avg': float(row['quiz_score_avg']),
                    'modules_completed': float(row['modules_completed']),
                    'days_since_login': float(row['days_since_login']),
                    'revisit_count': float(row['revisit_count']),
                    'assignment_submissions': float(row['assignment_submissions']),
                    'forum_activity': float(row['forum_activity']),
                    'video_watch_pct': float(row['video_watch_pct']),
                    'at_risk': int(row['at_risk'])
                })
        
        # Prepare for predictions
        features_list = []
        for s in students_data:
            features_list.append([s[f] for f in FEATURES])
        
        X = np.array(features_list)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        for i, s in enumerate(students_data):
            s['probability'] = float(probs[i])
            
        total = len(students_data)
        at_risk_gt_count = sum(1 for s in students_data if s['at_risk'] == 1)
        at_risk_rate = at_risk_gt_count / total
        
        avg_quiz = sum(s['quiz_score_avg'] for s in students_data) / total
        avg_video = sum(s['video_watch_pct'] for s in students_data) / total
        
        # Bins for logins_per_week
        eng_bins = {
            '0–2 logins': {'total': 0, 'risk': 0},
            '3–4': {'total': 0, 'risk': 0},
            '5–7': {'total': 0, 'risk': 0},
            '8–10': {'total': 0, 'risk': 0},
            '11–13': {'total': 0, 'risk': 0}
        }
        for s in students_data:
            logins = s['logins_per_week']
            if logins <= 2:
                bin_name = '0–2 logins'
            elif logins <= 4:
                bin_name = '3–4'
            elif logins <= 7:
                bin_name = '5–7'
            elif logins <= 10:
                bin_name = '8–10'
            else:
                bin_name = '11–13'
                
            eng_bins[bin_name]['total'] += 1
            if s['at_risk'] == 1:
                eng_bins[bin_name]['risk'] += 1
                
        eng_chart_data = []
        for label in ['0–2 logins', '3–4', '5–7', '8–10', '11–13']:
            b = eng_bins[label]
            pct = (b['risk'] / b['total'] * 100) if b['total'] > 0 else 0
            eng_chart_data.append(round(pct, 1))
            
        # Cohorts
        cohorts = {
            'Low Engagement': {'high': 0, 'medium': 0, 'low': 0},
            'Mid Engagement': {'high': 0, 'medium': 0, 'low': 0},
            'High Engagement': {'high': 0, 'medium': 0, 'low': 0}
        }
        for s in students_data:
            logins = s['logins_per_week']
            if logins <= 3:
                c_name = 'Low Engagement'
            elif logins <= 7:
                c_name = 'Mid Engagement'
            else:
                c_name = 'High Engagement'
                
            prob = s['probability']
            if prob >= 0.75:
                cohorts[c_name]['high'] += 1
            elif prob >= 0.50:
                cohorts[c_name]['medium'] += 1
            else:
                cohorts[c_name]['low'] += 1
                
        cohort_chart_data = {
            'High Risk': [cohorts['Low Engagement']['high'], cohorts['Mid Engagement']['high'], cohorts['High Engagement']['high']],
            'Medium Risk': [cohorts['Low Engagement']['medium'], cohorts['Mid Engagement']['medium'], cohorts['High Engagement']['medium']],
            'Low Risk': [cohorts['Low Engagement']['low'], cohorts['Mid Engagement']['low'], cohorts['High Engagement']['low']]
        }
        
        TRAINING_STATS = {
            'total_students': total,
            'at_risk_rate': round(at_risk_rate * 100, 1),
            'avg_quiz_score': round(avg_quiz, 1),
            'avg_video_watch': round(avg_video, 1),
            'eng_chart_data': eng_chart_data,
            'cohort_chart_data': cohort_chart_data
        }
    except Exception as e:
        print(f"Error calculating training stats: {str(e)}")

# health check — open in browser to confirm server is alive
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status'  : 'kodo dropout prediction api is running',
        'version' : '1.0',
        'endpoint': 'POST /predict'
    })

@app.route('/training-stats', methods=['GET'])
def training_stats():
    global TRAINING_STATS
    if TRAINING_STATS is None:
        calculate_training_stats()
    if TRAINING_STATS is None:
        return jsonify({'error': 'could not calculate training stats'}), 500
    return jsonify(TRAINING_STATS)


# feature order must match training data exactly
FEATURES = [
    'logins_per_week',
    'avg_time_per_session',
    'quiz_score_avg',
    'modules_completed',
    'days_since_login',
    'revisit_count',
    'assignment_submissions',
    'forum_activity',
    'video_watch_pct'
]


# predict dropout risk for a single student
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({'error': 'no data received — send json body'}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({'error': 'missing features', 'missing': missing}), 400

    try:
        features = np.array([[data[f] for f in FEATURES]])  # shape (1, 9)
    except Exception as e:
        return jsonify({'error': f'invalid feature values : {str(e)}'}), 400

    features_scaled = scaler.transform(features)  # must match training scaling
    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0][1]  # prob of at-risk

    risk_label = 'AT RISK' if prediction == 1 else 'SAFE'

    if probability >= 0.75:
        message = 'high risk — this student needs immediate attention'
    elif probability >= 0.50:
        message = 'moderate risk — keep an eye on this student'
    else:
        message = 'low risk — student is doing fine'

    return jsonify({
        'risk_label'       : risk_label,
        'risk_probability' : round(float(probability), 4),
        'risk_percent'     : f"{round(probability * 100, 1)}%",
        'message'          : message
    })


# predict for multiple students in one call
@app.route('/predict-batch', methods=['POST'])
def predict_batch():

    data = request.get_json()

    if not data or 'students' not in data:
        return jsonify({'error': 'send json with students list'}), 400

    students = data['students']

    if len(students) == 0:
        return jsonify({'error': 'students list is empty'}), 400

    results = []

    for i, student in enumerate(students):

        missing = [f for f in FEATURES if f not in student]
        if missing:
            results.append({'student_index': i, 'error': f'missing features : {missing}'})
            continue

        features        = np.array([[student[f] for f in FEATURES]])
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        probability     = model.predict_proba(features_scaled)[0][1]
        risk_label      = 'AT RISK' if prediction == 1 else 'SAFE'

        results.append({
            'student_index'    : i,
            'risk_label'       : risk_label,
            'risk_probability' : round(float(probability), 4),
            'risk_percent'     : f"{round(probability * 100, 1)}%"
        })

    at_risk_count = sum(1 for r in results if r.get('risk_label') == 'AT RISK')

    return jsonify({
        'total_students' : len(students),
        'at_risk_count'  : at_risk_count,
        'safe_count'     : len(students) - at_risk_count,
        'predictions'    : results
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    # set debug=False in production



