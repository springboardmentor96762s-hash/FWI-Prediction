import pickle

# Load your model
ridge = pickle.load(open("ridge.pkl", "rb"))

# Print model information
print("Number of features expected:", ridge.n_features_in_)

# For models trained with pandas DataFrame
try:
    print("Feature names:", ridge.feature_names_in_)
except:
    print("No feature names stored in model.")
