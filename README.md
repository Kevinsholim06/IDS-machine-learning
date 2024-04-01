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
