# IoT-Anomaly-Detector
Unsupervised Anomaly Detection system for Industrial IoT. Uses a Deep Learning Autoencoder (Keras/TensorFlow) to monitor engine health and predict failures via reconstruction error analysis. Features a live Streamlit dashboard.
#  Smart Factory: IoT Edge AI Anomaly Detector
### Predicting Industrial Downtime before it Happens

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](http://localhost:8502/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
In industrial settings, unplanned equipment downtime can cost thousands of dollars per minute. This project demonstrates a **Predictive Maintenance** solution that "listens" to machine heartbeats (sensor data) and identifies mechanical degradation without requiring human-labeled "failure" data.

Using an **Unsupervised Deep Learning** approach, the system learns the inherent patterns of a healthy machine and triggers an alarm the moment it encounters "confused" states that deviate from the norm.

## 🚀 Key Features
* **Real-Time Telemetry:** Live visualization of Temperature and Vibration streams using Plotly.
* **Deep Learning Inference:** Uses a trained **Autoencoder Neural Network** to calculate reconstruction error.
* **Smart Alerting:** Automated threshold logic that triggers a "CRITICAL" status when anomalies exceed safety limits.
* **Interactive Digital Twin:** Built with Streamlit to allow factory managers to monitor asset health via a web interface.

## 🧠 The AI Logic: Autoencoder Anomaly Detection
Traditional ML requires thousands of examples of "Broken" machines to learn. In the real world, we rarely have that data. 

**This project uses an Unsupervised approach:**
1. **Training:** The model is trained only on **Healthy Data**. It learns to compress and reconstruct normal sensor patterns perfectly.
2. **Detection:** When the machine starts to fail (Anomaly), the model gets "confused." 
3. **Reconstruction Error:** We measure this confusion (Mean Squared Error). If the error spikes above a specific **Threshold**, an alarm is triggered.



## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow / Keras
* **Data Science:** Pandas, NumPy, Scikit-Learn (Min-Max Scaling)
* **Dashboard:** Streamlit
* **Visualization:** Plotly (Interactive Charts)

