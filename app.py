import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Define function to train the model
def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Function to load or train the model
def load_or_train_model(X_train, y_train):
    model_filename = "trained_model.pkl"
    if os.path.exists(model_filename):
        clf = joblib.load(model_filename)
    else:
        clf = train_model(X_train, y_train)
        joblib.dump(clf, model_filename)
    return clf

# Define function to make predictions
def predict_label(sample_data, clf):
    return clf.predict(sample_data)

# Create Streamlit app
def main():
    st.title("Network Intrusion Detection System")

    # Load the dataset
    data = pd.read_csv('/home/sholim/ml/IDS/dataset/2020.06.19.csv')
    # Drop missing values
    data.dropna(inplace=True)

    # Split features and labels
    X = data.drop(columns=['label'])
    y = data['label']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    clf = load_or_train_model(X_train, y_train)

    st.write("### Sample Data Input")
    st.write("Enter sample data for prediction:")
  
    avg_ipt = st.number_input("avg_ipt", value=7.5)
    bytes_in = st.number_input("bytes_in", value=342)
    bytes_out = st.number_input("bytes_out", value=3679)
    dest_ip = st.number_input("dest_ip", value=786)
    dest_port = st.number_input("dest_port", value=9200)
    entropy = st.number_input("entropy", value=5.436687)
    num_pkts_out = st.number_input("num_pkts_out", value=2)
    num_pkts_in = st.number_input("num_pkts_in", value=2)
    proto = st.number_input("proto", value=6)
    src_ip = st.number_input("src_ip", value=786)
    src_port = st.number_input("src_port", value=57392)
    time_end = st.number_input("time_end", value=1592533725648144)
    time_start = st.number_input("time_start", value=1592533725632946)
    total_entropy = st.number_input("total_entropy", value=21860.918)
    duration = st.number_input("duration", value=0.01519)

    # Create DataFrame for sample data
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

    if st.button("Predict"):
        prediction = predict_label(sample_data, clf)
        st.write("### Prediction")
        st.write("Predicted label for sample data:", prediction)

    st.write("### Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Predict labels for each row in the uploaded CSV file
        predictions = predict_label(data.drop(columns=['label']), clf)

        st.write("### Predictions for Uploaded Data")
        st.write(predictions)

        # Save predictions to separate CSV files based on predicted labels
        prediction_df = pd.DataFrame({'Predicted_Label': predictions})
        for label in prediction_df['Predicted_Label'].unique():
            label_data = data[prediction_df['Predicted_Label'] == label]
            label_data.to_csv(f"{label}.csv", index=False)

        # Visualize the distribution of predicted labels
        st.write("### Distribution of Predicted Labels")
        prediction_counts = pd.Series(predictions).value_counts()
        st.bar_chart(prediction_counts)

if __name__ == "__main__":
    main()

# import streamlit as st
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib

# # Load the dataset
# data = pd.read_csv('/dataset/2020.06.19.csv')
# # Drop missing values

# data.dropna(inplace=True)

# # Split features and labels
# X = data.drop(columns=['label'])
# y = data['label']

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Save the trained model
# model_filename = "trained_model.pkl"
# joblib.dump(clf, model_filename)

# # Define function to make predictions
# def predict_label(sample_data):
#     return clf.predict(sample_data)

# # Create Streamlit app
# def main():
#     st.title("Network Intrusion Detection System")

#     st.write("### Sample Data Input")
#     st.write("Enter sample data for prediction:")
  

#     # Define sample data input fields
#     avg_ipt = st.number_input("avg_ipt", value=7.5)
#     bytes_in = st.number_input("bytes_in", value=342)
#     bytes_out = st.number_input("bytes_out", value=3679)
#     dest_ip = st.number_input("dest_ip", value=786)
#     dest_port = st.number_input("dest_port", value=9200)
#     entropy = st.number_input("entropy", value=5.436687)
#     num_pkts_out = st.number_input("num_pkts_out", value=2)
#     num_pkts_in = st.number_input("num_pkts_in", value=2)
#     proto = st.number_input("proto", value=6)
#     src_ip = st.number_input("src_ip", value=786)
#     src_port = st.number_input("src_port", value=57392)
#     time_end = st.number_input("time_end", value=1592533725648144)
#     time_start = st.number_input("time_start", value=1592533725632946)
#     total_entropy = st.number_input("total_entropy", value=21860.918)
#     duration = st.number_input("duration", value=0.01519)

#     # Create DataFrame for sample data
#     sample_data = pd.DataFrame({
#         'avg_ipt': [avg_ipt],
#         'bytes_in': [bytes_in],
#         'bytes_out': [bytes_out],
#         'dest_ip': [dest_ip],
#         'dest_port': [dest_port],
#         'entropy': [entropy],
#         'num_pkts_out': [num_pkts_out],
#         'num_pkts_in': [num_pkts_in],
#         'proto': [proto],
#         'src_ip': [src_ip],
#         'src_port': [src_port],
#         'time_end': [time_end],
#         'time_start': [time_start],
#         'total_entropy': [total_entropy],
#         'duration': [duration]
#     })

#     if st.button("Predict"):
#         prediction = predict_label(sample_data)
#         st.write("### Prediction")
#         st.write("Predicted label for sample data:", prediction)

#     st.write("### Upload CSV File")
#     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)

#         # Predict labels for each row in the uploaded CSV file
#         predictions = predict_label(data.drop(columns=['label']))

#         st.write("### Predictions for Uploaded Data")
#         st.write(predictions)

#         # Save predictions to separate CSV files based on predicted labels
#         prediction_df = pd.DataFrame({'Predicted_Label': predictions})
#         for label in prediction_df['Predicted_Label'].unique():
#             label_data = data[prediction_df['Predicted_Label'] == label]
#             label_data.to_csv(f"{label}.csv", index=False)

#         # Visualize the distribution of predicted labels
#         st.write("### Distribution of Predicted Labels")
#         prediction_counts = pd.Series(predictions).value_counts()
#         st.bar_chart(prediction_counts)

# if __name__ == "__main__":
#     main()
