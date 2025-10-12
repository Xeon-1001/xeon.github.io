import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import requests
from streamlit_lottie import st_lottie


# 1. Aesthetics
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Drug Recommendation DSS",
    page_icon="💊",
    layout="centered"
)

# THEN, call local_css
local_css("assets/style.css")

# --- 2. DATA LOADING AND MODEL TRAINING (No changes here) ---
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


# --- 3. USER INTERFACE (This is the updated section) ---
st.title('Symptom-Based Drug Recommendation DSS')
st.markdown("This system suggests medication based on your profile and diagnosis.")

# Move all the controls into a sidebar
with st.sidebar:
    lottie_url = "https://lottie.host/e1664778-ef1e-47f2-a5c2-77689af60fb1/Lbw3J2Y5j2.json"
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

    st.markdown("---") # Visual separator
    recommend_button = st.button("Get Recommendation", type="primary")


# --- 4. PREDICTION AND OUTPUT ---
if recommend_button:
    # --- (Your prediction logic stays the same) ---
    user_input = pd.DataFrame({'Condition': [selected_condition], 'Symptom': [selected_symptom]})
    user_input_encoded = encoder.transform(user_input)
    prediction = model.predict(user_input_encoded)
    
    st.subheader('📝 Recommendation Result')

    # Create a card-like layout
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("<h1 style='text-align: center; font-size: 5rem;'>💊</h1>", unsafe_allow_html=True)
    
    with col2:
        st.success(f"**Suggested Medication:**")
        st.markdown(f"<h3 style='text-align: left; color: #33FFB8;'>{prediction[0]}</h3>", unsafe_allow_html=True)
        st.info(f"**Profile:** {selected_condition.capitalize()} | **Diagnosis:** {selected_symptom}")

    with st.expander("⚠️ Important Disclaimer"):
        st.warning("""
        This is only a DSS. Dont let em docs run out of job.
        """)
else:
    st.info("Please enter your details in the sidebar and click 'Get Recommendation'.")




