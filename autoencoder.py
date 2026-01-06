import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Richa's AI IoT Monitor", 
    layout="wide", 
    page_icon="🤖"
)

# ---THEME CSS---
st.markdown("""
    <style>
    /* 1. Metric Value: Deep Cobalt Blue */
    [data-testid="stMetricValue"] > div {
        color: #1f4287 !important;
        font-weight: 700 !important;
    }
    
    /* 2. Metric Label: Slate Grey */
    [data-testid="stMetricLabel"] > div > p {
        color: #6c757d !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* 3. The Card: Subtle Border instead of solid white */
    [data-testid="stMetric"] {
        border: 1px solid #1f4287;
        padding: 15px !important;
        border-radius: 12px !important;
        background-color: rgba(31, 66, 135, 0.05) !important; /* Very faint blue tint */
    }
    
    /* 4. Footer Styling */
    .footer {
        position: fixed;
        left: 0; bottom: 0; width: 100%;
        text-align: center; padding: 8px;
        color: #1f4287; font-style: italic;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: MISSION CONTROL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2082/2082211.png", width=80)
    st.title("Mission Control")
    st.markdown("---")
    st.info("**Developer:** Richa Muchhal")
    st.write("**Target:** Predictive Maintenance (IIoT)")
    st.markdown("---")
    st.write("### Project Motive")
    st.caption("""
    In industrial settings, a single minute of downtime can cost thousands of dollars. 
    This project demonstrates how **Unsupervised Deep Learning** can 'listen' to machine 
    heartbeats and predict failure BEFORE it happens.
    """)
    if st.button("Reset Simulation Data"):
        st.rerun()

# --- HEADER SECTION ---
st.title("🏭 Smart Factory: Edge AI Anomaly Detector")
st.markdown("""
    **What is happening here?** We are simulating a Jet Engine's sensor data. 
    A Deep Learning **Autoencoder** is monitoring these signals to find hidden patterns of failure.
""")

# --- STEP 1: DATA GENERATION ENGINE ---
@st.cache_data
def generate_industrial_data():
    np.random.seed(42)
    time_steps = np.arange(0, 1000)
    # Simulate normal operational noise
    temp = 200 + np.random.normal(0, 1.5, 1000)
    vib = 50 + np.random.normal(0, 0.8, 1000)
    # Inject an 'Industrial Fault' (Anomaly) starting at index 950
    temp[950:] += (np.linspace(0, 25, 50) + np.random.normal(0, 2, 50))
    vib[950:] += (np.linspace(0, 15, 50) + np.random.normal(0, 1.5, 50))
    return pd.DataFrame({'temp': temp, 'vib': vib})

df = generate_industrial_data()

# --- UPDATED REAL-TIME METRICS ---
# Using standard Streamlit columns to ensure text contrast
st.subheader("Current Machine State")
m_col1, m_col2, m_col3 = st.columns(3)

curr_temp = df['temp'].iloc[-1]
curr_vib = df['vib'].iloc[-1]

# We use the 'delta' parameter to show the High/Normal status clearly
m_col1.metric(
    label="Engine Temperature", 
    value=f"{curr_temp:.2f} K", 
    delta="HIGH TEMP" if curr_temp > 215 else "Normal",
    delta_color="inverse" if curr_temp > 215 else "normal"
)

m_col2.metric(
    label="Vibration Level", 
    value=f"{curr_vib:.2f} mm/s", 
    delta="HIGH VIB" if curr_vib > 55 else "Normal",
    delta_color="inverse" if curr_vib > 55 else "normal"
)

m_col3.metric(
    label="System Status", 
    value="CRITICAL" if curr_temp > 215 else "STABLE",
    delta="Check Required" if curr_temp > 215 else "Optimal",
    delta_color="inverse" if curr_temp > 215 else "normal"
)



# --- STEP 3: INTERACTIVE DATA EXPLORATION ---
st.subheader(" Multi-Sensor Telemetry")
tab1, tab2, tab3 = st.tabs([" Temperature", " Vibration", "How it works"])

with tab1:
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(y=df['temp'], name="Temperature", line=dict(color='#FF4B4B', width=2)))
    fig_t.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Time (s)", yaxis_title="Kelvin")
    st.plotly_chart(fig_t, use_container_width=True)

with tab2:
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(y=df['vib'], name="Vibration", line=dict(color='#0068C9', width=2)))
    fig_v.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Time (s)", yaxis_title="mm/s")
    st.plotly_chart(fig_v, use_container_width=True)

with tab3:
    st.markdown("""
    ### The Science of Anomaly Detection
    1. **Data Ingestion:** Sensors stream Temperature and Vibration data every second.
    2. **Feature Scaling:** We shrink numbers to a range of 0 to 1 so the AI doesn't get overwhelmed.
    3. **The Bottleneck:** Our **Autoencoder** squeezes the data into a tiny summary. 
    4. **The Reconstruction:** The AI tries to 'redraw' the original sensors from that summary. 
    5. **The Verdict:** If the AI fails to redraw accurately, we know the machine has entered a state it never saw during training!
    """)

# --- STEP 4: AI DIAGNOSTICS SECTION ---
st.divider()
st.subheader(" Deep Learning Inference")

if st.button("🚀 Run AI Health Check"):
    try:
        # Load the model
        model = load_model("engine_model.keras")
        
        # Prepare data for prediction
        data_values = df[['temp', 'vib']].values
        
        # Inference Step
        with st.spinner('AI is analyzing sensor patterns...'):
            reconstructions = model.predict(data_values)
            mse = np.mean(np.power(data_values - reconstructions, 2), axis=1)
        
        # Visualization of AI "Confusion"
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(y=mse, name="Reconstruction Error", fill='tozeroy', line=dict(color='orange')))
        
        # Define a dynamic threshold based on the data
        threshold = 0.08
        fig_ai.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Safety Limit")
        
        fig_ai.update_layout(title="AI Reconstruction Error (Confusion Score)", height=450)
        st.plotly_chart(fig_ai, use_container_width=True)
        
        # Final Verdict Logic
        if mse[-1] > threshold:
            st.error(f"⚠️ **Anomaly Detected!** Final Error Score: {mse[-1]:.4f}. Current patterns suggest mechanical degradation.")
        else:
            st.success(f"✅ **System Healthy.** Final Error Score: {mse[-1]:.4f}. Sensor patterns match trained 'Normal' parameters.")
            st.balloons()
            
    except Exception as e:
        st.warning("⚠️ **System Error:** Could not find `engine_model.keras`. Please ensure your trained model is in the root directory.")
        st.info("Tip: You can build and save this model using the provided Jupyter Notebook.")

# --- FOOTER ---
st.markdown(f'<div class="footer">Built with 💙 by Richa Muchhal | ML & IoT Enthusiast</div>', unsafe_allow_html=True)