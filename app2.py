import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Placeholder for loading your trained model
loaded_model = pickle.load(open('classifier.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Set up the app
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

# Section navigation
section = st.radio("Navigate to:", ["Introduction to Project", "About Early Detection", "Simulation"])

# Introduction to Project
if section == "Introduction to Project":
    st.title("Welcome to the Breast Cancer Prediction Project")
    
    st.write("""
    This project aims to support the early detection of breast cancer through a machine learning model that predicts the likelihood of a tumor being malignant (cancerous) or benign (non-cancerous).
    Our model is trained on data from breast tissue images, helping to provide a preliminary assessment that can encourage individuals to seek further medical evaluation.
    By providing this tool, we hope to raise awareness about the importance of regular health screenings and proactive health management.
    """)

    # Add an inspiring image
    st.image("image2.jpg", use_column_width=True)

# About Early Detection
elif section == "About Early Detection":
    st.title("The Importance of Early Detection")
    
    # Information about survival rates and early detection
    st.write("""
    According to the American Cancer Society, when breast cancer is detected early, and is in the localized stage, 
    the 5-year relative survival rate is **99%**. Early detection includes doing monthly breast self-exams, 
    and scheduling regular clinical breast exams and mammograms.
    """)

    st.subheader("Inspiring Words on Early Detection")
    
    # List of quotes and corresponding images
    early_detection_quotes = [
        "“I can’t stress enough how important it is to get screened and checked for all cancers — and to do self breast-exams.” —Robin Roberts",
        "“I am a 36-year-old person with breast cancer, and not many people know that that happens to women my age or women in their 20s. This is my opportunity now to go out and fight as hard as I can for early detection.” —Christina Applegate",
        "“But it is possible to take control and tackle head-on any health issue. You can seek advice, learn about the options and make choices that are right for you. Knowledge is power.” —Angelina Jolie"
    ]
    
    image_files = ["image3.jpg", "image4.jpg", "image5.jpg"]

    # Display quotes with images
    for quote, image in zip(early_detection_quotes, image_files):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, use_column_width=True)
        with col2:
            st.markdown(f"> {quote}")

# Simulation
elif section == "Simulation":
    st.title("Breast Cancer Prediction Simulation")

    # Input fields for selected features
    st.subheader("Enter the following parameters:")
    data_input = {
        'radius_mean': st.number_input('Radius Mean', min_value=0.0, step=0.1),
        'perimeter_mean': st.number_input('Perimeter Mean', min_value=0.0, step=0.1),
        'area_mean': st.number_input('Area Mean', min_value=0.0, step=0.1),
        'concavity_mean': st.number_input('Concavity Mean', min_value=0.0, step=0.01),
        'concave points_mean': st.number_input('Concave Points Mean', min_value=0.0, step=0.01),
        'radius_worst': st.number_input('Radius Worst', min_value=0.0, step=0.1),
        'perimeter_worst': st.number_input('Perimeter Worst', min_value=0.0, step=0.1),
        'area_worst': st.number_input('Area Worst', min_value=0.0, step=0.1),
        'concavity_worst': st.number_input('Concavity Worst', min_value=0.0, step=0.01),
        'concave points_worst': st.number_input('Concave Points Worst', min_value=0.0, step=0.01)
    }

    if st.button('Predict'):
        # Convert user input to DataFrame
        input_df = pd.DataFrame([data_input])

        # Scale the input using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make a prediction
        prediction = loaded_model.predict(scaled_input)

        # Display the result
        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        st.write(f"Prediction: **{result}**")

        # Display prediction probability (placeholder example)
        st.bar_chart(pd.DataFrame({
            'Class': ['Benign', 'Malignant'],
            'Probability': [0.7, 0.3] if result == 'Benign' else [0.3, 0.7]
        }))

        # Downloadable report
        report = f"Prediction Report\n{'-'*20}\n{data_input}\nPrediction: {result}"
        st.download_button("Download Report", data=report, file_name="prediction_report.pdf")
