import streamlit as st
import pandas as pd
import os
import pickle
from model import train_model, predict_sample, evaluate_model

# Title of the Streamlit app
st.title('Network Intrusion Detection System')

# Load trained models and scaler
if 'intrusion_detection_model.pkl' not in os.listdir():
    intrusion_forest, rf_clf, scaler, X_train, X_test, y_train, y_test = train_model('/home/sholim/ml/archive/2020/06/2020.06.20/2020.06.20.csv')
    with open('intrusion_detection_model.pkl', 'wb') as f:
        pickle.dump((intrusion_forest, rf_clf, scaler, X_train, X_test, y_train, y_test), f)
else:
    with open('intrusion_detection_model.pkl', 'rb') as f:
        intrusion_forest, rf_clf, scaler, X_train, X_test, y_train, y_test = pickle.load(f)

# Sidebar for user input
st.sidebar.header('Enter Sample Data:')
avg_ipt = st.sidebar.number_input('Average Inter-Packet Time:', value=22.0)
bytes_in = st.sidebar.number_input('Bytes In:', value=5603)
bytes_out = st.sidebar.number_input('Bytes Out:', value=5109)
dest_ip = st.sidebar.number_input('Destination IP:', value=786)
dest_port = st.sidebar.number_input('Destination Port:', value=9200.0)
entropy = st.sidebar.number_input('Entropy:', value=6.189827)
num_pkts_out = st.sidebar.number_input('Number of Packets Out:', value=61)
num_pkts_in = st.sidebar.number_input('Number of Packets In:', value=86)
proto = st.sidebar.number_input('Protocol:', value=6)
src_ip = st.sidebar.number_input('Source IP:', value=14061)
src_port = st.sidebar.number_input('Source Port:', value=52441.0)
time_end = st.sidebar.number_input('Time End:', value=159261117061976)
time_start = st.sidebar.number_input('Time Start:', value=1592611158753504)
total_entropy = st.sidebar.number_input('Total Entropy:', value=66305.43)
duration = st.sidebar.number_input('Duration:', value=0.866256)

if st.sidebar.button('Make Prediction'):
    # Predictions
    sample_data = pd.DataFrame({
        'avg_ipt': [avg_ipt],
        'bytes_in': [bytes_in],
        'bytes_out': [bytes_out],
        'dest_ip': [dest_ip],
        'dest_port': [dest_port],
        'entropy': [entropy],
        'num_pkts_out': [num_pkts_out],
        'num_pkts_in': [num_pkts_in],
        'proto': [proto],
        'src_ip': [src_ip],
        'src_port': [src_port],
        'time_end': [time_end],
        'time_start': [time_start],
        'total_entropy': [total_entropy],
        'duration': [duration]
    })

    out_prediction, class_prediction = predict_sample(sample_data, intrusion_forest, rf_clf, scaler)

    # Display predictions
    st.subheader('Prediction:')
    st.write("Anomaly Prediction:", out_prediction)
    if out_prediction == "not an outlier":
        st.write("Classification Prediction:", class_prediction)

# Evaluation metrics
if st.checkbox('Show Model Performance'):
    precision_iso, recall_iso, f1_iso, accuracy_rf = evaluate_model(intrusion_forest, rf_clf, X_test, y_test)
    st.write('Isolation Forest Metrics:')
    st.write('Precision Score:', precision_iso)
    st.write('Recall Score:', recall_iso)
    st.write('F1 Score:', f1_iso)

    st.write('Random Forest Classifier Metrics:')
    st.write('Accuracy Score:', accuracy_rf)
