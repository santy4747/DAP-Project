import joblib
import pandas as pd
import json
from flask import Flask, request, jsonify

# We need to define the app in the global scope for Netlify
app = Flask(__name__)

# Load the trained model and data from the file
model_data = joblib.load('life_expectancy_model.pkl')
model = model_data['model']
top_features = model_data['top_features']

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # The frontend will send a dictionary of feature values
        # We need to convert this into a DataFrame for the model
        input_data = pd.DataFrame([data])
        
        # Ensure all required columns are present, even if with default values (e.g., NaN)
        # The pipeline's imputer will handle missing values
        for col in model.named_steps['preprocessor'].transformers_[0][2]: # numerical features
             if col not in input_data.columns:
                 input_data[col] = 0 # or np.nan
        
        for col in model.named_steps['preprocessor'].transformers_[1][2]: # categorical features
             if col not in input_data.columns:
                 input_data[col] = "Developing" # a default category


        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Return the prediction as JSON
        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# This is the handler that Netlify will use to run your function
def handler(event, context):
    # This is a wrapper to integrate Flask with AWS Lambda (which Netlify uses)
    from serverless_wsgi import handle
    return handle(app, event, context)
