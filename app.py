from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        
        # Scale data
        scaled_data = scaler.transform([data])
        
        # Prediction
        prediction = model.predict(scaled_data)[0]
        
        # # Result message
        # if prediction == 1:
        #     result = "⚠️ High risk of Diabetes!"
        # else:
        #     result = "✅ Low risk of Diabetes."
        
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
