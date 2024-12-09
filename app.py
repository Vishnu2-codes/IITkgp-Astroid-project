import streamlit as st
import pandas as pd
import lightgbm as lgb
import joblib
import numpy as np

# Load your trained model
model = joblib.load('C:\\Users\\vishn\\OneDrive\\Desktop\\IITkgp Astroid project\\asteroid_hazard_model.pkl') 
# Streamlit app title
st.title("Asteroid Hazard Classification")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Function to create user input fields
def user_input_features():
    approach_date = st.sidebar.date_input("Approach Date")
    relative_velocity = st.sidebar.number_input("Relative Velocity (km/s)", min_value=0.0)
    miss_distance = st.sidebar.number_input("Miss Distance (km)", min_value=0.0)
    semi_major_axis = st.sidebar.number_input("Semi-major Axis (AU)", min_value=0.0)
    eccentricity = st.sidebar.number_input("Eccentricity", min_value=0.0, max_value=1.0)
    # Add more input fields as per your features, following the same format

    # Create a DataFrame from the input data
    data = {
        'approach_date': approach_date,
        'relative_velocity': relative_velocity,
        'miss_distance': miss_distance,
        'semi_major_axis': semi_major_axis,
        'eccentricity': eccentricity,
        # Include all other input features here, ensuring they match your model's input
    }
    return pd.DataFrame(data, index=[0])

input_data = user_input_features()

# Display the user inputs
st.subheader("User Input Data")
st.write(input_data)

# Prediction
if st.sidebar.button("Predict"):
    # Ensure your model accepts the input shape
    input_data['approach_date'] = pd.to_datetime(input_data['approach_date']).astype(np.int64) // 10**9  # Convert date to timestamp
    prediction = model.predict(input_data)
    st.write(f"Prediction: {'Hazardous' if prediction[0] == 1 else 'Not Hazardous'}")

# Optional: Add visualizations or any additional output
st.subheader("Data Overview")
# Load the original dataset for reference
data = pd.read_csv('C:/Users/vishn/OneDrive/Desktop/IITkgp Astroid project/dataset.csv')
st.write(data.describe())  # Show statistics of the dataset

# Optionally, show plots
# You can create plots based on your original dataset or the predictions made
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model):
    # Assuming you want to show feature importance
    importance = model.feature_importance()
    feature_names = ['relative_velocity', 'miss_distance', 'semi_major_axis', 'eccentricity']  # Adjust to your feature names
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.title("Feature Importance")
    st.pyplot(plt)

# Show feature importance when button clicked
if st.sidebar.button("Show Feature Importance"):
    plot_feature_importance(model)
