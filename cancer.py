import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('rfc_model.pkl')

# Function to perform prediction
def predict_cancer(radius_mean, perimeter_mean, area_mean, symmetry_mean, compactness_mean, concave_points_mean):
    # Preprocess input data
    # (you might need to scale the input features using the provided min-max ranges)
    # (for simplicity, I'm assuming no scaling in this example)
    
    # Make prediction
    prediction = model.predict([[radius_mean, perimeter_mean, area_mean, symmetry_mean, compactness_mean, concave_points_mean]])
    
    return prediction[0]

# Streamlit app
st.title('Breast Cancer Prediction')

# Information about the input fields
st.sidebar.write("""
## Input Parameters:
- **Radius Mean:** The mean of distances from center to points on the perimeter.
- **Perimeter Mean:** The mean size of the core tumor.
- **Area Mean:** The mean size of the core tumor.
- **Symmetry Mean:** Symmetry of the core tumor.
- **Compactness Mean:** Compactness of the core tumor.
- **Concave Points Mean:** The mean number of concave portions of the contour.
""")

# Add input fields in sidebar
radius_mean = st.sidebar.slider('Radius Mean (7.0 - 29.0)', min_value=7.0, max_value=29.0, step=0.1)
perimeter_mean = st.sidebar.slider('Perimeter Mean (43.79 - 188.5)', min_value=43.79, max_value=188.5, step=0.1)
area_mean = st.sidebar.slider('Area Mean (143.5 - 2501.0)', min_value=143.5, max_value=2501.0, step=1.0)
symmetry_mean = st.sidebar.slider('Symmetry Mean (0.106 - 0.304)', min_value=0.106, max_value=0.304, step=0.001)
compactness_mean = st.sidebar.slider('Compactness Mean (0.01938 - 0.3454)', min_value=0.01938, max_value=0.3454, step=0.0001)
concave_points_mean = st.sidebar.slider('Concave Points Mean (0.0 - 0.2012)', min_value=0.0, max_value=0.2012, step=0.0001)

# Predict button
if st.sidebar.button('Predict'):
    diagnosis = predict_cancer(radius_mean, perimeter_mean, area_mean, symmetry_mean, compactness_mean, concave_points_mean)
    if diagnosis == 1:
        st.write("## Diagnosis: Malignant")
        st.error("Please consult a healthcare professional for further evaluation.")
    else:
        st.write("## Diagnosis: Benign")
        st.success("No evidence of malignant tumor detected. However, regular check-ups are recommended.")
