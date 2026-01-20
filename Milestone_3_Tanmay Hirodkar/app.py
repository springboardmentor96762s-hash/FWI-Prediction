from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os

app = Flask(__name__)

class FWIPredictor:
    def __init__(self):
        self.fwi_model = None
        self.fire_model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset for training"""
        try:
            # Get the absolute path to the CSV file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, 'ff2.csv')
            
            # Read CSV with error handling for malformed data
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            
            # Clean the data - remove rows with any non-numeric data in numeric columns
            numeric_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
            
            # Convert to numeric, setting errors='coerce' to convert invalid values to NaN
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values in critical columns
            df = df.dropna(subset=numeric_columns + ['Classes'])
            
            # Convert Classes to binary (1 for fire, 0 for not fire)
            df['fire_binary'] = (df['Classes'] == 'fire').astype(int)
            
            # Select features for prediction
            # Using weather parameters to predict FWI components and fire risk
            self.feature_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']
            
            # Prepare features and targets
            X = df[self.feature_columns]
            y_fwi = df['FWI']
            y_fire = df['fire_binary']
            
            print(f"Loaded {len(df)} valid records from CSV")
            
            return X, y_fwi, y_fire
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def train_models(self):
        """Train both FWI regression and fire classification models"""
        X, y_fwi, y_fire = self.load_and_prepare_data()
        
        if X is None:
            return False
            
        # Split data
        X_train, X_test, y_fwi_train, y_fwi_test, y_fire_train, y_fire_test = train_test_split(
            X, y_fwi, y_fire, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train FWI regression model using Ridge Regression
        self.fwi_model = Ridge(alpha=1.0, random_state=42)
        self.fwi_model.fit(X_train_scaled, y_fwi_train)
        
        # Train fire classification model using Logistic Regression
        self.fire_model = LogisticRegression(random_state=42, max_iter=1000)
        self.fire_model.fit(X_train_scaled, y_fire_train)
        
        # Evaluate models
        fwi_pred = self.fwi_model.predict(X_test_scaled)
        fire_pred = self.fire_model.predict(X_test_scaled)
        
        fwi_rmse = np.sqrt(mean_squared_error(y_fwi_test, fwi_pred))
        fire_accuracy = accuracy_score(y_fire_test, fire_pred)
        
        print(f"Ridge Regression FWI Model RMSE: {fwi_rmse:.2f}")
        print(f"Logistic Regression Fire Classification Accuracy: {fire_accuracy:.2f}")
        
        self.is_trained = True
        return True
    
    def predict(self, temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui):
        """Make predictions for FWI and fire risk"""
        if not self.is_trained:
            return None, None, None
        
        # Prepare input data
        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui]])
        input_scaled = self.scaler.transform(input_data)
        
        # Predict FWI
        fwi_prediction = self.fwi_model.predict(input_scaled)[0]
        
        # Predict fire probability and classification
        fire_probability = self.fire_model.predict_proba(input_scaled)[0][1]  # Probability of fire
        fire_prediction = "Fire Risk" if fire_probability > 0.5 else "No Fire Risk"
        
        return fwi_prediction, fire_probability, fire_prediction

# Initialize predictor
predictor = FWIPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_models():
    """Train the models"""
    try:
        success = predictor.train_models()
        if success:
            return jsonify({'status': 'success', 'message': 'Models trained successfully!'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to train models. Check if ff2.csv exists.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    """Make FWI and fire risk predictions"""
    try:
        if not predictor.is_trained:
            return jsonify({'status': 'error', 'message': 'Models not trained yet. Please train first.'})
        
        # Get input data from form
        data = request.get_json()
        
        temperature = float(data['temperature'])
        rh = float(data['rh'])
        ws = float(data['ws'])
        rain = float(data['rain'])
        ffmc = float(data['ffmc'])
        dmc = float(data['dmc'])
        dc = float(data['dc'])
        isi = float(data['isi'])
        bui = float(data['bui'])
        
        # Make predictions
        fwi_pred, fire_prob, fire_class = predictor.predict(
            temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui
        )
        
        if fwi_pred is None:
            return jsonify({'status': 'error', 'message': 'Prediction failed'})
        
        return jsonify({
            'status': 'success',
            'fwi_index': round(fwi_pred, 2),
            'fire_probability': round(fire_prob * 100, 1),
            'fire_classification': fire_class,
            'risk_level': get_risk_level(fwi_pred)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})

def get_risk_level(fwi):
    """Convert FWI to risk level"""
    if fwi < 1:
        return "Very Low"
    elif fwi < 3:
        return "Low"
    elif fwi < 7:
        return "Moderate"
    elif fwi < 17:
        return "High"
    else:
        return "Very High"

if __name__ == '__main__':
    app.run(debug=True)
