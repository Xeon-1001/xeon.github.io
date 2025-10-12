import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# --- 1. PAGE CONFIG (Add this at the top) ---
# This MUST be the first Streamlit command in your app
st.set_page_config(
    page_title="Drug Recommendation DSS",
    page_icon="💊",
    layout="centered"
)

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
    user_input = pd.DataFrame({
        'Condition': [selected_condition],
        'Symptom': [selected_symptom]
    })
    user_input_encoded = encoder.transform(user_input)
    prediction = model.predict(user_input_encoded)
    
    st.subheader('📝 Recommendation Result')
    st.success(f"**Based on your inputs, the suggested medication is:** `{prediction[0]}`")

    with st.expander("⚠️ Important Disclaimer"):
        st.warning("""
        This is a prototype DSS. Always consult a qualified healthcare professional 
        for any medical advice, diagnosis, or treatment.
        """)
else:
    # This message will show on the main page before the button is clicked
    st.info("Please enter your details in the sidebar and click 'Get Recommendation'.")
