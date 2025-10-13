import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import requests
from streamlit_lottie import st_lottie
import base64 # Required for encoding the video

# --- 1. AESTHETICS & HELPERS ---

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_video_background(file_path):
    """
    Sets a video as the background of the Streamlit app.
    """
    bin_str = get_base64_of_bin_file(file_path)
    
    video_html = f"""
    <style>
    .stApp {{
        background: none; /* Required to see the video */
    }}
    #bg-video {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        z-index: -1; /* Pushes video to the background */
    }}
    </style>
    <video id="bg-video" autoplay loop muted>
        <source src="data:video/mp4;base64,{bin_str}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(
    page_title="Drug Recommendation DSS",
    page_icon="💊",
    layout="wide"
)

# --- APPLY AESTHETICS ---

# Use the new MP4 file for the background
set_video_background("assets/comr.mp4")

local_css("assets/style.css")

# --- 2. DATA LOADING AND MODEL TRAINING ---
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('syndata.csv')
    features = ['Condition', 'Symptom']
    target = 'Assigned_Drug'
    X = df[features]
    y = df[target]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)
    return model, encoder, list(df['Symptom'].unique()), list(df['Condition'].unique())

model, encoder, symptom_list, condition_list = load_and_train_model()

# --- 3. USER INTERFACE ---
st.title('Drug Recommendation DSS')

# --- SIDEBAR ---
with st.sidebar:
    lottie_url = "https://raw.githubusercontent.com/Xeon-1001/xeon.github.io/refs/heads/main/assets/animation.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=150, key="lottie_animation")

    st.header("👤 Your Details")
    
    selected_symptom = st.selectbox(
        "Diagnosed Disease:",
        options=symptom_list
    )

    selected_condition = st.radio(
        "Personal Profile:",
        options=condition_list,
        horizontal=True
    )

    st.markdown("---")
    recommend_button = st.button("Get Recommendation", type="primary")

# --- 4. PREDICTION AND OUTPUT ---
if recommend_button:
    user_input = pd.DataFrame({'Condition': [selected_condition], 'Symptom': [selected_symptom]})
    user_input_encoded = encoder.transform(user_input)
    prediction = model.predict(user_input_encoded)
    
    st.subheader('📝 Recommendation Result')

    # Create a centered layout using three columns
    left_col, mid_col, right_col = st.columns([1, 3, 1])
    
    with mid_col:
        # Use st.markdown to create a custom "card"
        st.markdown(
            f"""
            <div style="
                border: 1px solid #262730;
                border-radius: 10px;
                padding: 25px;
                text-align: center;
                background-color: #0E1117;
            ">
                <p style="font-size: 1rem; color: #FAFAFA;">Suggested Medication:</p>
                <h2 style="color: #33FFB8; margin-top: -10px;">{prediction[0]}</h2>
                <hr style="border-color: #262730;">
                <p style="font-size: 0.9rem; color: #A0A0A0;">
                    <b>Profile:</b> {selected_condition.capitalize()} | <b>Diagnosis:</b> {selected_symptom}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Disclaimer remains full-width below the centered card
    with st.expander("⚠️ Important Disclaimer"):
        st.warning("This is a prototype DSS. Dont let em docs run outta jobs.")
else:
    st.info("Please enter your details in the sidebar and click 'Get Recommendation'.")












