from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and feature names
with open('heart_failure_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        anaemia = int(request.form['anaemia'])
        creatinine_phosphokinase = float(request.form['creatinine_phosphokinase'])
        diabetes = int(request.form['diabetes'])
        ejection_fraction = float(request.form['ejection_fraction'])
        high_blood_pressure = int(request.form['high_blood_pressure'])
        platelets = float(request.form['platelets'])
        serum_creatinine = float(request.form['serum_creatinine'])
        serum_sodium = float(request.form['serum_sodium'])
        sex = int(request.form['sex'])
        smoking = int(request.form['smoking'])
        time = float(request.form['time'])
        
        # Create feature array in the correct order
        features = np.array([[
            age, anaemia, creatinine_phosphokinase, diabetes,
            ejection_fraction, high_blood_pressure, platelets,
            serum_creatinine, serum_sodium, sex, smoking, time
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Format result
        if prediction == 0:
            result = "LOW RISK - Patient likely to survive"
            confidence = probability[0] * 100
            result_class = "low-risk"
        else:
            result = "HIGH RISK - Patient needs immediate attention"
            confidence = probability[1] * 100
            result_class = "high-risk"
        
        return render_template('index.html', 
                             prediction_text=result,
                             confidence=f"{confidence:.1f}%",
                             result_class=result_class)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text="Error in prediction. Please check your inputs.",
                             result_class="error")

if __name__ == "__main__":
    app.run(debug=True)