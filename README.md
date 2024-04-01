# Network Intrusion Detection System (NIDS) with Streamlit

## Overview

This project is a Network Intrusion Detection System (NIDS) implemented using Python and Streamlit. The NIDS is based on a Random Forest classifier trained on a dataset of network traffic data. The Streamlit app provides an interface for users to input sample data or upload CSV files for prediction. Additionally, the app visualizes the distribution of predicted labels and saves predictions to separate CSV files.

## Features

- Input sample data for prediction using interactive fields.
- Upload CSV files containing network traffic data for bulk prediction.
- Predict network intrusion labels (e.g., benign, malicious, outlier).
- Visualize the distribution of predicted labels using bar charts.
- Save predictions to separate CSV files based on predicted labels.
- Pre-trained model is loaded from disk to avoid retraining.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Kevinsholim06/IDS-machine-learning.git


2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt

3. Run the Streamlit app:

    ```bash
    streamlit run app.py

4. Access the app in your web browser at http://localhost:8501.

## Usage
- Input sample data in the provided fields and click "Predict" to see the prediction.
- Upload a CSV file containing network traffic data for bulk prediction. 
- The app will predict -labels for each row in the file and display the predictions.
- The app visualizes the distribution of predicted labels using bar charts.
- Predictions are saved to separate CSV files based on predicted labels.

## File Structure
- app.py: Main Streamlit application script.
- requirements.txt: List of Python dependencies.
- trained_model.pkl: Pre-trained Random Forest classifier serialized using joblib.

## Contributors
- https://github.com/Kevinsholim06
