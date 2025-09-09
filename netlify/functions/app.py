import json
import joblib
import pandas as pd
import os
from urllib.parse import parse_qs

# Global variable to cache the model
model_data = None

def load_model():
    """Load the model once and cache it"""
    global model_data
    if model_data is None:
        # Get the path relative to this function file
        model_path = os.path.join(os.path.dirname(__file__), 'student_performance_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model_data = joblib.load(model_path)
    return model_data

def handler(event, context):
    """Main Netlify function handler"""
    try:
        # Get HTTP method and path
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '')
        
        # Set CORS headers
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Content-Type': 'application/json'
        }
        
        # Handle preflight OPTIONS request
        if http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': ''
            }
        
        # Route handling
        if path.endswith('/eda') and http_method == 'GET':
            return handle_eda(headers)
        elif path.endswith('/predict') and http_method == 'POST':
            return handle_predict(event, headers)
        else:
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps({'error': 'Endpoint not found'})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': 'Internal server error', 'details': str(e)})
        }

def handle_eda(headers):
    """Handle EDA endpoint"""
    try:
        data = load_model()
        visualizations = data.get('visualizations')
        
        if not visualizations:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': 'Visualizations not found in model file'})
            }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(visualizations)
        }
        
    except FileNotFoundError as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': 'Model file missing', 'details': str(e)})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': 'Error loading EDA data', 'details': str(e)})
        }

def handle_predict(event, headers):
    """Handle prediction endpoint"""
    try:
        data = load_model()
        model = data.get('model')
        
        if model is None:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': 'Model object not found in pkl file'})
            }
        
        # Parse request body
        body = event.get('body', '{}')
        if isinstance(body, str):
            request_data = json.loads(body)
        else:
            request_data = body
            
        if not request_data:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Invalid JSON input'})
            }
        
        # Convert to DataFrame for prediction
        df = pd.DataFrame([request_data])
        
        # Ensure all required columns are present
        if hasattr(model, 'feature_names_in_'):
            required_cols = model.feature_names_in_
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({'prediction': f'{prediction:.2f}'})
        }
        
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'headers': headers,
            'body': json.dumps({'error': 'Invalid JSON format'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': 'Prediction error', 'details': str(e)})
        }