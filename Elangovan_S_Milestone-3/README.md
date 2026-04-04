# Fire Weather Index (FWI) Prediction - Milestone 3

## 🔥 Project Overview

This is **Milestone 3** of the Fire Weather Index (FWI) Prediction project - a comprehensive **Flask-based web application** that deploys a machine learning model for real-time prediction of Fire Weather Index values. The application provides both a user-friendly web interface and a RESTful API for predicting fire danger levels based on meteorological and fire weather system components.

### 🎯 Project Objectives

1. **Deploy ML Model**: Create a production-ready web application for the trained Ridge Regression model
2. **Web Interface**: Develop intuitive user interfaces for single and batch predictions
3. **API Development**: Provide RESTful API endpoints for programmatic access
4. **Scalability**: Handle both individual predictions and bulk CSV file processing
5. **User Experience**: Deliver real-time predictions with visual fire danger classifications

### 📊 About Fire Weather Index (FWI)

The Fire Weather Index (FWI) is a comprehensive numerical rating system used to estimate fire intensity and behavior. It combines various weather conditions and fuel moisture codes to assess fire danger. This system is widely used by forest fire management agencies worldwide for:

- Fire danger assessment and communication
- Resource allocation and firefighting planning
- Public warnings and fire prevention strategies
- Research and historical fire behavior analysis

## ✨ Key Features

### Core Functionality

- **🌐 Interactive Web Interface**: User-friendly form with real-time validation for single predictions
- **📁 Batch Prediction System**: Upload CSV files for bulk predictions on entire datasets (up to 16MB)
- **⚡ Real-time Predictions**: Instant FWI predictions using the pre-trained Ridge Regression model
- **📊 Comprehensive Statistics**: Detailed analytics for batch predictions including distribution analysis
- **🎨 Fire Danger Classification**: Visual color-coded indicators for danger levels (Low, Moderate, High, Extreme)
- **💾 Download Results**: Export prediction results as CSV files with danger level classifications
- **🔌 REST API**: JSON API endpoint for seamless integration with other applications
- **📱 Responsive Design**: Mobile-friendly interface that works across all devices
- **🔒 Input Validation**: Robust error handling and data validation
- **📈 Preview Mode**: View first 10 predictions before downloading complete results

### Technical Features

- **Feature Standardization**: Automatic scaling using pre-trained StandardScaler
- **File Upload Management**: Secure file handling with size limits and validation
- **Error Handling**: Comprehensive error messages and user-friendly error pages
- **Data Integrity**: Checks for missing values and invalid data formats
- **Memory Efficient**: Streaming file processing for large datasets

## 🧠 Model Information

### Machine Learning Pipeline

**Algorithm**: Ridge Regression (L2 Regularization)

- **Alpha Parameter**: 1.0
- **Regularization**: L2 penalty to prevent overfitting
- **Training Dataset**: Algerian Forest Fires Dataset (2012)
- **Regions**: Bejaia (northeast) and Sidi Bel-abbes (northwest) Algeria
- **Total Samples**: 244 instances
- **Data Period**: June to September 2012

### Preprocessing Pipeline

**Feature Scaling**: StandardScaler

- **Method**: Z-score normalization (mean=0, std=1)
- **Purpose**: Normalize features to same scale for optimal Ridge performance
- **Fitted On**: Training data from Milestone 1
- **Applied To**: All predictions (real-time and batch)

### Model Features

**Input Features** (9 features used for prediction):

- Temperature, RH (Relative Humidity), Ws (Wind Speed), Rain
- FFMC (Fine Fuel Moisture Code), DMC (Duff Moisture Code), DC (Drought Code)
- ISI (Initial Spread Index), BUI (Buildup Index)

**Excluded Features** (from original dataset):

- day, month, year (temporal features)
- Region (categorical location)
- Classes (fire/no fire binary classification)

**Excluded Features** (from original dataset):

- day, month, year (temporal features)
- Region (categorical location)
- Classes (fire/no fire binary classification)

**Target Variable**: FWI (Fire Weather Index)

- **Range**: Typically 0-50+ (higher values indicate greater fire danger)
- **Interpretation**: Numerical rating combining ISI and BUI
- **Usage**: Primary indicator for fire intensity and danger assessment

## 🚀 Installation & Setup

### Prerequisites

- **Python**: Version 3.8 or higher
- **pip**: Python package installer
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 1GB RAM recommended for batch processing
- **Storage**: ~50MB for dependencies and model files

### Step-by-Step Installation

1. **Navigate to the project directory**:

   ```bash
   cd /home/elango/Documents/projects/infosys/FWI-Prediction/Elangovan_S_Milestone-2
   ```

2. **Create a virtual environment** (strongly recommended):

   ```bash
   # Create virtual environment
   python3 -m venv venv

   # Activate virtual environment
   # On Linux/Mac:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The application requires the following Python packages:

```
Flask==3.0.0          # Web framework
numpy==1.24.3         # Numerical computing
pandas==2.0.3         # Data manipulation
scikit-learn==1.3.0   # Machine learning (includes Ridge, StandardScaler)
```

### Required Files Verification

Ensure the following files are present in the project directory:

**Essential Files**:

- ✅ `ridge.pkl` - Trained Ridge regression model (serialized)
- ✅ `scaler.pkl` - Fitted StandardScaler for feature normalization (serialized)
- ✅ `app.py` - Flask application main file
- ✅ `requirements.txt` - Python dependencies list
- ✅ `sample_dataset.csv` - Sample data for testing batch predictions

**Template Files** (in `templates/` directory):

- ✅ `index.html` - Home page with input form for single prediction
- ✅ `result.html` - Results display page for single predictions
- ✅ `batch_predict.html` - Batch prediction upload page
- ✅ `batch_result.html` - Batch prediction results with statistics
- ✅ `error.html` - Error handling page

**Optional Files**:

- 📊 `Algerian_forest_fires_cleaned_dataset.csv` - Full training dataset
- 📓 `Elangovan_milestone-1.ipynb` - Model training notebook
- 📁 `uploads/` - Directory for uploaded CSV files and results

## 💻 Usage Guide

### Starting the Application

1. **Start the Flask development server**:

   ```bash
   python app.py
   ```

   Expected output:

   ```
    * Serving Flask app 'app'
    * Debug mode: on
    * Running on all addresses (0.0.0.0)
    * Running on http://127.0.0.1:5000
    * Running on http://192.168.x.x:5000
   Press CTRL+C to quit
   ```

2. **Access the application**:
   - **Local access**: `http://localhost:5000` or `http://127.0.0.1:5000`
   - **Network access**: `http://<your-ip>:5000` (from other devices on same network)

### Method 1: Single Prediction (Web Form)

**Perfect for**: Testing individual scenarios, quick predictions, manual data entry

1. Navigate to the home page: `http://localhost:5000`
2. Fill in all 9 required input fields:
   - **Temperature** (°C): Air temperature (e.g., 29)
   - **RH** (%): Relative Humidity 0-100 (e.g., 57)
   - **Ws** (km/h): Wind Speed (e.g., 18)
   - **Rain** (mm): Rainfall amount, ≥0 (e.g., 0)
   - **FFMC**: Fine Fuel Moisture Code 0-101 (e.g., 65.7)
   - **DMC**: Duff Moisture Code (e.g., 3.4)
   - **DC**: Drought Code (e.g., 7.6)
   - **ISI**: Initial Spread Index (e.g., 1.3)
   - **BUI**: Buildup Index (e.g., 3.4)
3. Click **"Predict FWI"** button
4. View results:
   - Predicted FWI value
   - Fire danger level (color-coded)
   - Input summary for verification

**Example Input Values**:

```
Temperature: 35°C, RH: 30%, Ws: 25 km/h, Rain: 0 mm
FFMC: 90.5, DMC: 50.2, DC: 200.3, ISI: 15.5, BUI: 80.4
→ Expected: High to Extreme fire danger
```

### Method 2: Batch Prediction (CSV Upload)

**Perfect for**: Processing multiple records, historical data analysis, bulk predictions

1. **Navigate to batch prediction page**:

   - Click "Batch Prediction" link from home page, or
   - Go directly to `http://localhost:5000/batch_predict`

2. **Prepare your CSV file** with these columns (exact names required):

   ```
   Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI
   ```

3. **Upload the CSV file**:

   - Click "Choose File" and select your CSV
   - File size limit: 16MB
   - Supported format: .csv only

4. **View comprehensive results**:

   **Overall Statistics**:

   - Total records processed
   - Average FWI across all predictions
   - Minimum and maximum FWI values

   **Fire Danger Distribution**:

   - Low danger count (FWI < 5)
   - Moderate danger count (5 ≤ FWI < 15)
   - High danger count (15 ≤ FWI < 30)
   - Extreme danger count (FWI ≥ 30)

   **Data Preview**:

   - First 10 rows with predictions
   - Predicted_FWI column added
   - Fire_Danger_Level column added

5. **Download results**:
   - Click "Download Results" button
   - CSV file includes all original columns plus predictions

**Sample CSV Format** (`sample_dataset.csv` included):

```csv
Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI
29,57,18,0,65.7,3.4,7.6,1.3,3.4
35,30,25,0,90.5,50.2,200.3,15.5,80.4
25,80,10,5.5,45.0,10.0,50.0,2.0,20.0
22,65,12,0.0,55.3,25.6,100.8,3.2,35.7
```

**Output Format**:

```csv
Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Predicted_FWI,Fire_Danger_Level
29,57,18,0,65.7,3.4,7.6,1.3,3.4,2.45,Low
35,30,25,0,90.5,50.2,200.3,15.5,80.4,28.76,High
25,80,10,5.5,45.0,10.0,50.0,2.0,20.0,4.23,Low
22,65,12,0.0,55.3,25.6,100.8,3.2,35.7,12.89,Moderate
```

### Method 3: REST API (Programmatic Access)

**Perfect for**: System integration, automated workflows, mobile apps, data pipelines

### Method 3: REST API (Programmatic Access)

**Perfect for**: System integration, automated workflows, mobile apps, data pipelines

#### API Endpoint

**URL**: `POST /predict_api`  
**Content-Type**: `application/json`  
**Port**: 5000 (default)

#### Request Format

**JSON Body** (all fields required):

```json
{
  "Temperature": 29.0,
  "RH": 57.0,
  "Ws": 18.0,
  "Rain": 0.0,
  "FFMC": 65.7,
  "DMC": 3.4,
  "DC": 7.6,
  "ISI": 1.3,
  "BUI": 3.4
}
```

#### Response Format

**Success Response** (HTTP 200):

```json
{
  "success": true,
  "predicted_fwi": 2.5346
}
```

**Error Response** (HTTP 400):

```json
{
  "success": false,
  "error": "Error message describing the issue"
}
```

#### Example Implementations

**Using cURL** (Command Line):

```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "Temperature": 29,
    "RH": 57,
    "Ws": 18,
    "Rain": 0,
    "FFMC": 65.7,
    "DMC": 3.4,
    "DC": 7.6,
    "ISI": 1.3,
    "BUI": 3.4
  }'
```

**Using Python** (with requests library):

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/predict_api"

# Prepare data
data = {
    "Temperature": 29,
    "RH": 57,
    "Ws": 18,
    "Rain": 0,
    "FFMC": 65.7,
    "DMC": 3.4,
    "DC": 7.6,
    "ISI": 1.3,
    "BUI": 3.4
}

# Make request
response = requests.post(url, json=data)

# Parse response
if response.status_code == 200:
    result = response.json()
    if result['success']:
        print(f"Predicted FWI: {result['predicted_fwi']}")
    else:
        print(f"Error: {result['error']}")
else:
    print(f"HTTP Error: {response.status_code}")
```

**Using JavaScript** (Node.js/Browser):

```javascript
const url = "http://localhost:5000/predict_api";

const data = {
  Temperature: 29,
  RH: 57,
  Ws: 18,
  Rain: 0,
  FFMC: 65.7,
  DMC: 3.4,
  DC: 7.6,
  ISI: 1.3,
  BUI: 3.4,
};

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(data),
})
  .then((response) => response.json())
  .then((result) => {
    if (result.success) {
      console.log("Predicted FWI:", result.predicted_fwi);
    } else {
      console.error("Error:", result.error);
    }
  })
  .catch((error) => console.error("Request failed:", error));
```

## 📋 Input Features Reference

### Detailed Feature Descriptions

| Feature     | Description             | Unit | Typical Range | Data Type | Notes                          |
| ----------- | ----------------------- | ---- | ------------- | --------- | ------------------------------ |
| Temperature | Air temperature         | °C   | -5 to 45      | Float     | Daily maximum temperature      |
| RH          | Relative Humidity       | %    | 0 to 100      | Float     | Percentage of moisture in air  |
| Ws          | Wind Speed              | km/h | 0 to 50       | Float     | Wind speed at standard height  |
| Rain        | Rainfall amount         | mm   | 0 to 50+      | Float     | Daily rainfall (0 = no rain)   |
| FFMC        | Fine Fuel Moisture Code | -    | 0 to 101      | Float     | Fine fuel dryness indicator    |
| DMC         | Duff Moisture Code      | -    | 0 to 500+     | Float     | Medium-depth organic layer     |
| DC          | Drought Code            | -    | 0 to 1000+    | Float     | Deep organic layer dryness     |
| ISI         | Initial Spread Index    | -    | 0 to 50+      | Float     | Fire spread rate indicator     |
| BUI         | Buildup Index           | -    | 0 to 300+     | Float     | Total available fuel indicator |

### FWI System Components Explained

#### Moisture Codes

**1. FFMC (Fine Fuel Moisture Code)** [0-101]

- **Measures**: Moisture in fine dead fuels (litter, grass, needles)
- **Response Time**: Rapid (hours)
- **Influenced by**: Temperature, humidity, wind, rain
- **Interpretation**:
  - < 70: High moisture, low ignition risk
  - 70-85: Moderate moisture
  - 85-92: Low moisture, increasing risk
  - > 92: Very dry, high ignition potential

**2. DMC (Duff Moisture Code)** [0-500+]

- **Measures**: Moisture in loosely compacted organic layers
- **Response Time**: Moderate (days)
- **Depth**: ~7cm of decomposed organic matter
- **Interpretation**:
  - < 30: Wet conditions
  - 30-60: Moderate drying
  - > 60: Significant drying

**3. DC (Drought Code)** [0-1000+]

- **Measures**: Moisture in deep, compact organic layers
- **Response Time**: Slow (weeks/months)
- **Depth**: ~18cm of deep duff
- **Interpretation**:
  - < 200: Low seasonal drought
  - 200-400: Moderate drought
  - > 400: Severe drought conditions

#### Fire Behavior Indices

**4. ISI (Initial Spread Index)** [0-50+]

- **Formula**: Combines FFMC and wind speed
- **Indicates**: Expected rate of fire spread
- **Purpose**: Predicts how fast fire will spread initially
- **Interpretation**:
  - < 5: Slow spread
  - 5-10: Moderate spread
  - > 10: Fast spread potential

**5. BUI (Buildup Index)** [0-300+]

- **Formula**: Combines DMC and DC
- **Indicates**: Total fuel available for combustion
- **Purpose**: Represents fuel accumulation and drying
- **Interpretation**:
  - < 40: Low fuel availability
  - 40-80: Moderate fuel buildup
  - > 80: High fuel availability

### Target Variable: FWI (Fire Weather Index)

**Calculation**: Combines ISI and BUI using empirical relationships  
**Range**: 0 to 50+ (no upper limit, but values >50 are rare)  
**Purpose**: Comprehensive fire danger rating

## 🔥 Fire Danger Classification System

The application classifies predicted FWI values into four danger levels:

| Danger Level | FWI Range  | Color Code | Meaning              | Recommended Actions                        |
| ------------ | ---------- | ---------- | -------------------- | ------------------------------------------ |
| **Low**      | < 5        | 🟢 Green   | Minimal fire danger  | Normal conditions, routine monitoring      |
| **Moderate** | 5 to < 15  | 🟡 Yellow  | Moderate fire danger | Increase awareness, monitor conditions     |
| **High**     | 15 to < 30 | 🟠 Orange  | High fire danger     | Heightened alert, restrict activities      |
| **Extreme**  | ≥ 30       | 🔴 Red     | Extreme fire danger  | Critical alert, fire bans, emergency ready |

### Danger Level Interpretations

**Low (FWI < 5)**:

- Fuels are moist, ignition is unlikely
- Fires that start will spread slowly
- Minimal resource requirements if fires occur

**Moderate (5 ≤ FWI < 15)**:

- Average fire danger conditions
- Fires can start and spread at moderate rates
- Standard fire suppression resources adequate

**High (15 ≤ FWI < 30)**:

- High probability of fire starts
- Rapid fire spread expected
- Increased resources and vigilance needed
- Consider activity restrictions

**Extreme (FWI ≥ 30)**:

- Very high fire danger, critical conditions
- Explosive fire behavior possible
- Crown fires and spotting likely
- Full fire ban, emergency readiness essential

## 🏗️ Project Architecture

### Application Structure

```
Elangovan_S_Milestone-2/
│
├── app.py                                    # Flask application (main entry point)
├── requirements.txt                          # Python package dependencies
├── README.md                                 # This documentation file
├── documentation.docx                        # Detailed project documentation
│
├── Models (Serialized)
│   ├── ridge.pkl                            # Trained Ridge Regression model
│   └── scaler.pkl                           # Fitted StandardScaler transformer
│
├── Data Files
│   ├── Algerian_forest_fires_cleaned_dataset.csv  # Full training dataset
│   ├── sample_dataset.csv                   # Sample data for testing
│   └── Elangovan_milestone-1.ipynb         # Model training notebook
│
├── templates/                               # HTML templates (Flask Jinja2)
│   ├── index.html                          # Home page with input form
│   ├── result.html                         # Single prediction results
│   ├── batch_predict.html                  # Batch upload page
│   ├── batch_result.html                   # Batch results with statistics
│   └── error.html                          # Error handling page
│
├── uploads/                                 # User uploaded files & results
│   └── predictions_*.csv                   # Generated prediction files
│
└── Visualizations (from Milestone 1)
    ├── boxplot_features.png                # Feature distribution boxplots
    ├── correlation_heatmap.png             # Feature correlation matrix
    ├── distribution_plot.png               # FWI distribution plot
    └── pairplot.png                        # Pairwise feature relationships
```

### Application Flow

```
User Request
    │
    ├─→ Web Browser (/)
    │       ├─→ GET  /              → index.html (input form)
    │       ├─→ POST /predict       → result.html (prediction display)
    │       ├─→ GET  /batch_predict → batch_predict.html
    │       └─→ POST /batch_predict → batch_result.html (with stats)
    │
    └─→ API Client (/predict_api)
            └─→ POST /predict_api   → JSON response
│
Flask Application (app.py)
    │
    ├─→ Load Models
    │      ├─→ ridge.pkl    (Ridge Regression)
    │      └─→ scaler.pkl   (StandardScaler)
    │
    ├─→ Route Handlers
    │      ├─→ home()              # Render input form
    │      ├─→ predict()           # Process single prediction
    │      ├─→ predict_api()       # API endpoint
    │      ├─→ batch_predict()     # Handle CSV upload
    │      └─→ download_file()     # Serve result files
    │
    └─→ Helper Functions
           ├─→ allowed_file()              # Validate uploads
           └─→ get_fire_danger_level()     # Classify FWI
│
Prediction Pipeline
    │
    User Input (9 features)
        ↓
    Validation (data types, missing values)
        ↓
    Feature Extraction (FEATURE_NAMES order)
        ↓
    StandardScaler Transform (scaler.pkl)
        ↓
    Ridge Model Prediction (ridge.pkl)
        ↓
    Fire Danger Classification
        ↓
    Output (FWI value + danger level)
```

### Technology Stack

**Backend**:

- **Flask 3.0.0**: Lightweight WSGI web framework
- **Python 3.8+**: Core programming language

**Machine Learning**:

- **scikit-learn 1.3.0**: Ridge Regression, StandardScaler
- **NumPy 1.24.3**: Numerical computing
- **Pandas 2.0.3**: Data manipulation and CSV processing

**Frontend**:

- **HTML5**: Structure and semantics
- **CSS3**: Styling (included in templates)
- **JavaScript**: Client-side interactions (if applicable)

**File Handling**:

- **Werkzeug**: Secure filename handling
- **Python Pickle**: Model serialization

## 🔧 Configuration & Customization

### Application Settings

**File Upload Configuration** (in `app.py`):

```python
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit
```

**Server Configuration**:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Customization Options

#### 1. Change Port Number

```python
# In app.py, modify the last line:
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # Change port to 8080
```

Or run from command line:

```bash
python -c "from app import app; app.run(port=8080)"
```

#### 2. Update Fire Danger Thresholds

```python
# In app.py, modify get_fire_danger_level() function:
def get_fire_danger_level(fwi):
    if fwi < 5:
        return 'Low'
    elif fwi < 15:   # Adjust these thresholds
        return 'Moderate'
    elif fwi < 30:   # Adjust these thresholds
        return 'High'
    else:
        return 'Extreme'
```

#### 3. Add New Features to Model

If you retrain the model with additional features:

```python
# In app.py, update FEATURE_NAMES list:
FEATURE_NAMES = ['Temperature', 'RH', 'Ws', 'Rain',
                 'FFMC', 'DMC', 'DC', 'ISI', 'BUI',
                 'NewFeature1', 'NewFeature2']  # Add new features

# Then replace ridge.pkl and scaler.pkl with retrained models
```

#### 4. Increase Upload File Size Limit

```python
# In app.py:
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
```

#### 5. Production Deployment Settings

```python
# For production, disable debug mode:
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

## 🐛 Troubleshooting & Common Issues

### Issue 1: Port Already in Use

**Error**: `Address already in use` or `Port 5000 is already in use`

**Solutions**:

```bash
# Option 1: Kill process using port 5000
lsof -ti:5000 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :5000   # Windows (find PID, then taskkill)

# Option 2: Use different port
python -c "from app import app; app.run(port=5001)"

# Option 3: Change port in app.py
# Edit app.run(port=5001) in app.py
```

### Issue 2: Missing Model Files

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'ridge.pkl'`

**Solution**:

```bash
# Ensure you're in the correct directory
cd /home/elango/Documents/projects/infosys/FWI-Prediction/Elangovan_S_Milestone-2

# Verify files exist
ls -la *.pkl

# If missing, train model using Milestone-1 notebook
# Run: Elangovan_milestone-1.ipynb
```

### Issue 3: Import/Module Errors

**Error**: `ModuleNotFoundError: No module named 'flask'` or similar

**Solutions**:

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify installations
pip list
```

### Issue 4: CSV Upload Errors

**Error**: `Missing required columns: ...`

**Solution**:

- Ensure CSV has exact column names (case-sensitive):
  `Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI`
- Check for extra spaces in headers
- Verify CSV format (comma-separated, not semicolon)
- Use provided `sample_dataset.csv` as template

### Issue 5: Prediction Values Seem Wrong

**Possible Causes**:

1. **Incorrect feature scaling**: Scaler trained on different data
2. **Feature order mismatch**: Features not in expected order
3. **Model-data mismatch**: Model trained on different preprocessing

**Verification**:

```python
# Test with known sample from training data
# Expected: Similar FWI values as original dataset
```

### Issue 6: Large CSV Files Timeout

**Error**: Request timeout or memory error

**Solutions**:

```python
# Increase max file size (in app.py):
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

# Or process in chunks (requires code modification)
```

### Issue 7: Permission Denied (uploads folder)

**Error**: `Permission denied: 'uploads/'`

**Solution**:

```bash
# Create uploads directory with proper permissions
mkdir -p uploads
chmod 755 uploads

# Or run in app.py (already implemented):
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
```

## 🚀 Deployment Options

### Local Development (Current Setup)

```bash
# Already configured for local development
python app.py
# Access at http://localhost:5000
```

### Production Deployment with Gunicorn (Linux)

1. **Install Gunicorn**:

```bash
pip install gunicorn
```

2. **Run with Gunicorn**:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
# -w 4: 4 worker processes
# -b: bind to all interfaces
```

3. **Run as background service**:

```bash
nohup gunicorn -w 4 -b 0.0.0.0:5000 app:app > app.log 2>&1 &
```

### Docker Deployment

**Create `Dockerfile`**:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

**Build and Run**:

```bash
docker build -t fwi-prediction .
docker run -p 5000:5000 fwi-prediction
```

### Cloud Deployment (AWS/Azure/GCP)

**For Heroku**:

1. Add `Procfile`:

```
web: gunicorn app:app
```

2. Deploy:

```bash
heroku create fwi-prediction-app
git push heroku main
```

**For AWS EC2**:

1. Launch EC2 instance
2. Install Python, dependencies
3. Use Gunicorn + Nginx as reverse proxy
4. Configure security groups for port 5000

## 📊 Testing the Application

### Manual Testing Checklist

**Single Prediction**:

- [ ] Access home page successfully
- [ ] Submit form with valid data
- [ ] Receive prediction with danger level
- [ ] Test edge cases (extreme values)
- [ ] Verify error handling (invalid inputs)

**Batch Prediction**:

- [ ] Access batch prediction page
- [ ] Upload sample_dataset.csv
- [ ] View statistics correctly
- [ ] Download results successfully
- [ ] Test with custom CSV
- [ ] Verify error handling (wrong format, missing columns)

**API Testing**:

- [ ] Send POST request to /predict_api
- [ ] Receive JSON response
- [ ] Test error responses
- [ ] Verify response format

### Sample Test Cases

**Test Case 1: Low Danger Scenario**

```json
Input: {
  "Temperature": 22, "RH": 80, "Ws": 10, "Rain": 2.5,
  "FFMC": 50, "DMC": 15, "DC": 80, "ISI": 2, "BUI": 25
}
Expected: FWI < 5 (Low danger)
```

**Test Case 2: Extreme Danger Scenario**

```json
Input: {
  "Temperature": 38, "RH": 20, "Ws": 30, "Rain": 0,
  "FFMC": 95, "DMC": 80, "DC": 400, "ISI": 20, "BUI": 150
}
Expected: FWI ≥ 30 (Extreme danger)
```

## 📈 Performance Considerations

**Application Performance**:

- **Single Prediction**: ~10-50ms response time
- **Batch Prediction**: ~100-500ms for 100 rows
- **Memory Usage**: ~150-200MB (with models loaded)
- **Concurrent Users**: Supports 10-50 concurrent requests (development server)

**Optimization Tips**:

1. Use production WSGI server (Gunicorn) instead of Flask dev server
2. Implement caching for frequently accessed data
3. Add request rate limiting for API endpoints
4. Optimize large CSV processing with chunking
5. Use async processing for batch predictions

## 📚 API Documentation Summary

### Endpoints Overview

| Endpoint           | Method | Purpose                 | Input                | Output      |
| ------------------ | ------ | ----------------------- | -------------------- | ----------- |
| `/`                | GET    | Home page               | -                    | HTML        |
| `/predict`         | POST   | Single prediction (web) | Form data            | HTML        |
| `/predict_api`     | POST   | Single prediction (API) | JSON                 | JSON        |
| `/batch_predict`   | GET    | Batch upload page       | -                    | HTML        |
| `/batch_predict`   | POST   | Batch processing        | CSV file (multipart) | HTML + file |
| `/download/<file>` | GET    | Download results        | Filename in URL      | CSV file    |

## 🎓 Learning Outcomes (Milestone 2)

Through this milestone, the following skills were developed:

1. **Web Application Development**: Built full-stack Flask application
2. **Model Deployment**: Deployed ML model as web service
3. **API Design**: Created RESTful API endpoints
4. **File Handling**: Implemented secure file upload/download
5. **Data Processing**: Batch processing with pandas
6. **Error Handling**: Comprehensive validation and error management
7. **User Interface**: Designed intuitive web forms and result displays
8. **Documentation**: Created comprehensive technical documentation

## 📝 Future Enhancements

Potential improvements for future milestones:

1. **User Authentication**: Add login system for personalized experience
2. **Database Integration**: Store prediction history in PostgreSQL/MongoDB
3. **Visualization Dashboard**: Interactive charts with Plotly/D3.js
4. **Model Versioning**: Support multiple models with comparison
5. **Real-time Updates**: WebSocket for live predictions
6. **Mobile App**: React Native or Flutter mobile application
7. **Advanced Analytics**: Trend analysis and forecasting
8. **Email Notifications**: Alert system for extreme danger predictions
9. **Multi-language Support**: Internationalization (i18n)
10. **Cloud Storage**: S3/Azure Blob for uploaded files

## 👨‍💻 Development Team

**Developer**: Elangovan S  
**Project**: Fire Weather Index Prediction System  
**Institution**: Infosys Springboard  
**Milestone**: 2 - Model Deployment & Web Application

## 📄 License

This project is developed for educational purposes as part of the Infosys Springboard program.

## 🙏 Acknowledgments

- **Dataset Source**: Algerian Forest Fires Dataset (UCI Machine Learning Repository)
- **Research**: Forest fire prediction methodologies
- **Regions**: Bejaia and Sidi Bel-abbes regions, Algeria
- **Framework**: Flask web framework and scikit-learn library
- **Institution**: Infosys Springboard for project guidance

## 📞 Support & Contact

For issues, questions, or contributions:

- Review this README documentation
- Check troubleshooting section
- Refer to inline code comments in `app.py`
- Consult `documentation.docx` for detailed specifications

---

**Last Updated**: December 2024  
**Version**: 2.0  
**Status**: ✅ Production Ready
