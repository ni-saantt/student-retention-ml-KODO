# KODO DROPOUT PREDICTION — FLASK API
# Connects the trained model with the KODO dashboard


from flask import Flask, request, jsonify
import joblib
import numpy as np

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# load once at startup to avoid reloading on every request
model  = joblib.load('kodo_xgboost_model.pkl')
scaler = joblib.load('kodo_scaler.pkl')

print("model loaded successfully")
print("scaler loaded successfully")


# health check — open in browser to confirm server is alive
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status'  : 'kodo dropout prediction api is running',
        'version' : '1.0',
        'endpoint': 'POST /predict'
    })


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



