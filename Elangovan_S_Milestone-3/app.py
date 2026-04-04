from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained models
with open('ridge.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Feature names used in the model (day, month, year, Region, and FWI were excluded during training)
FEATURE_NAMES = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_fire_danger_level(fwi):
    """Classify FWI value into danger level"""
    if fwi < 5:
        return 'Low'
    elif fwi < 15:
        return 'Moderate'
    elif fwi < 30:
        return 'High'
    else:
        return 'Extreme'

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    Expects JSON data with feature values
    Returns predicted FWI value
    """
    try:
        # Get input data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Extract features in the correct order
        features = []
        for feature_name in FEATURE_NAMES:
            value = float(data.get(feature_name, 0))
            features.append(value)
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(features_array)
        
        # Make prediction using the Ridge model
        prediction = ridge_model.predict(scaled_features)
        
        # Return the result
        result = {
            'success': True,
            'predicted_fwi': float(prediction[0]),
            'input_features': dict(zip(FEATURE_NAMES, features))
        }
        
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('result.html', 
                                   prediction=float(prediction[0]),
                                   features=dict(zip(FEATURE_NAMES, features)))
    
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        
        if request.is_json:
            return jsonify(error_result), 400
        else:
            return render_template('error.html', error=str(e)), 400

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    API endpoint for prediction
    Accepts JSON with feature values
    Returns JSON with prediction
    """
    try:
        data = request.get_json()
        
        # Extract features
        features = [float(data.get(name, 0)) for name in FEATURE_NAMES]
        features_array = np.array(features).reshape(1, -1)
        
        # Scale and predict
        scaled_features = scaler.transform(features_array)
        prediction = ridge_model.predict(scaled_features)
        
        return jsonify({
            'success': True,
            'predicted_fwi': float(prediction[0])
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    """Handle batch predictions from uploaded CSV file"""
    if request.method == 'GET':
        return render_template('batch_predict.html')
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('error.html', error='No file uploaded'), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return render_template('error.html', error='No file selected'), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return render_template('error.html', error='Only CSV files are allowed'), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Validate that all required features are present
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            return render_template('error.html', 
                                   error=f'Missing required columns: {", ".join(missing_features)}'), 400
        
        # Extract features in correct order
        X = df[FEATURE_NAMES].values
        
        # Check for missing values
        if np.isnan(X).any():
            return render_template('error.html', 
                                   error='Dataset contains missing values. Please clean your data.'), 400
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = ridge_model.predict(X_scaled)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['Predicted_FWI'] = predictions
        result_df['Fire_Danger_Level'] = result_df['Predicted_FWI'].apply(get_fire_danger_level)
        
        # Generate statistics
        stats = {
            'total_records': len(result_df),
            'avg_fwi': float(np.mean(predictions)),
            'min_fwi': float(np.min(predictions)),
            'max_fwi': float(np.max(predictions)),
            'low_danger': int(sum(predictions < 5)),
            'moderate_danger': int(sum((predictions >= 5) & (predictions < 15))),
            'high_danger': int(sum((predictions >= 15) & (predictions < 30))),
            'extreme_danger': int(sum(predictions >= 30))
        }
        
        # Save result to session or temporary file
        output_filename = f"predictions_{secure_filename(file.filename)}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        result_df.to_csv(output_path, index=False)
        
        # Render results page
        return render_template('batch_result.html', 
                               stats=stats,
                               preview=result_df.head(10).to_html(classes='table table-striped', index=False),
                               download_file=output_filename)
    
    except pd.errors.EmptyDataError:
        return render_template('error.html', error='The uploaded file is empty'), 400
    except Exception as e:
        return render_template('error.html', error=f'Error processing file: {str(e)}'), 400

@app.route('/download/<filename>')
def download_file(filename):
    """Download the prediction results CSV file"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return render_template('error.html', error='File not found'), 404
    except Exception as e:
        return render_template('error.html', error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
