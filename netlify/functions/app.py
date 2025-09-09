import flask
from flask import request, jsonify
import joblib
import pandas as pd
import os

app = flask.Flask(__name__)

model_data = None

# --- Function to load the model ---
def load_model():
    global model_data
    if model_data is None:
        # The path needs to be relative to this app.py file
        model_path = os.path.join(os.path.dirname(__file__), 'student_performance_model.pkl')
        
        # Check if the file exists before trying to load it
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model_data = joblib.load(model_path)

# --- EDA Endpoint ---
@app.route('/api/eda', methods=['GET'])
def get_eda_data():
    try:
        load_model()
        visualizations = model_data.get('visualizations')
        if not visualizations:
            return jsonify({"error": "Visualizations not found in model file."}), 500
        return jsonify(visualizations)
        
    except FileNotFoundError as e:
        return jsonify({"error": "Model file is missing from the deployment package.", "details": str(e)}), 500
    except Exception as e:
        # Catch any other unexpected errors during loading or processing
        return jsonify({"error": "An unexpected error occurred on the server.", "details": str(e)}), 500

# --- Prediction Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        load_model()
        model = model_data.get('model')
        if model is None:
            return jsonify({"error": "Model object not found in pkl file."}), 500

        # Get data from the request
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON input"}), 400

        # Convert to DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Ensure all required columns are present for the model
        required_cols = model.feature_names_in_
        for col in required_cols:
             if col not in df.columns:
                 # Add missing columns with a default value if not provided by the form
                 # This is a simple fix; a more robust solution would handle this more gracefully
                 df[col] = 0

        prediction = model.predict(df)[0]
        
        return jsonify({'prediction': f'{prediction:.2f}'})

    except FileNotFoundError as e:
        return jsonify({"error": "Model file is missing from the deployment package.", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500

# This is required for Netlify to run the Flask app
# The file is executed as a script, not through a WSGI server
def handler(event, context):
    from werkzeug.wrappers import Request, Response
    from werkzeug.serving import run_simple

    # We need to wrap the Flask app for the serverless environment
    return run_simple(
        'localhost', 5000, app, environ=event, start_response=Response
    )

