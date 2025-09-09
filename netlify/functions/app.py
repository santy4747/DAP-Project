import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the entire data bundle from the file
model_data = joblib.load('student_performance_model.pkl')
model = model_data['model']
visualizations = model_data['visualizations'] # Extract the EDA plots

# --- Endpoint to serve EDA plots ---
@app.route('/api/eda', methods=['GET'])
def get_eda_data():
    """Returns the pre-generated EDA visualization strings."""
    return jsonify(visualizations)

# --- Prediction Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])
        
        # Ensure numeric types for columns used in prediction
        for col in ['G1', 'G2', 'absences', 'failures', 'Medu', 'studytime', 'Fedu', 'goout', 'Dalc', 'Walc', 'health']:
             if col in input_data.columns:
                 input_data[col] = pd.to_numeric(input_data[col])
        
        prediction = model.predict(input_data)[0]
        # Clamp prediction to be within the valid grade range (0-20)
        final_prediction = max(0.0, min(round(prediction, 2), 20.0))
        return jsonify({'prediction': final_prediction})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# The handler for Netlify
def handler(event, context):
    from serverless_wsgi import handle
    return handle(app, event, context)

