**Project Name: Tempest FWI Predictor – A Machine Learning Model to Predict Fire Weather Index**

**Forest Fire Weather Index (FWI) Prediction – Flask App**

**1.Flask Application Setup**
Created Flask app using app.py

Defined routes for:
         input page
         prediction result

  Loaded trained model and scaler

**2.Input Handling**

Accepted 8 input parameters:

Temperature, Humidity, Rain, FFMC, DMC, DC, ISI, BUI

Validated inputs for missing and numeric values

**3.Prediction Logic**

Scaled input data using saved scaler

Predicted FWI value

Rounded and classified FWI into danger levels

**4.Danger Classification**

Low, Moderate, High, Very High, Extreme

Assigned CSS class, GIF, sound, and voice message for each level

**5.Index Page (index.html)**

Created input form with all required fields

Linked form action to Flask /ict route

Clean and user-friendly UI

**6.Result Page (home.html)**
Displayed:

Predicted FWI value

Fire danger level

Animated meter bar

Dynamic background based on danger level

**7.Animations & GIFs**

Added fire-related GIF animations

Different GIF shown for each danger level

Stored in static/gifs/ folder

**8.Sound & Voice Alerts**

Added danger-level sound alerts

Implemented Text-to-Speech voice warning

Auto-plays message after prediction

**9.Error Handling**

Handled missing input errors

Handled invalid numeric values

Displayed user-friendly error messages

**10.Project Structure**

Organized files into:

(html and index files)
templates/

(gifs of fire level)
static/gifs/
