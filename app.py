from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

# -------------------- Flask App --------------------
app = Flask(__name__)
MODEL_PATH = "best_model.pkl"

# -------------------- Load Model --------------------
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

model = load_model()

# -------------------- Home Route --------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------- Prediction Route --------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        contract = request.form.get('Contract')
        tenure = float(request.form.get('tenure', 0))
        sim_operator = request.form.get('SIM_Operator')
        online_security = request.form.get('OnlineSecurity')

        # Convert categorical to numeric if needed
        # For simplicity, using dummy encoding. In production, map exactly as during training
        data = pd.DataFrame([{
            'Contract': contract,
            'tenure': tenure,
            'SIM_Operator': sim_operator,
            'OnlineSecurity': 1 if online_security=="Yes" else 0
        }])

        # Dummy encoding for categorical features
        data = pd.get_dummies(data)
        # Ensure columns match training columns
        model_columns = model.get_booster().feature_names if hasattr(model, "get_booster") else model.feature_names_in_
        for col in model_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[model_columns]

        # Prediction
        prediction = model.predict(data)[0]
        prediction_proba = model.predict_proba(data)[:,1][0]

        result = {
            "prediction": int(prediction),
            "probability": round(float(prediction_proba),4)
        }

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', error=str(e))

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)

