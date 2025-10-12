import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# --- 1. DATA LOADING AND PREPARATION (No changes here) ---
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


# --- 2. USER INTERFACE (UPDATED SECTION) ---

st.title('Symptom-Based Drug Recommendation DSS')
st.markdown("### Prototype model")

selected_symptom = st.selectbox(
    "Select the diagnosed disease:",
    options=symptom_list
)

selected_condition = st.radio(
    "Select your personal profile:",
    options=condition_list,
    horizontal=True
)

recommend_button = st.button("Get Recommendation")


# --- 3. PREDICTION AND OUTPUT

if recommend_button:
    user_input = pd.DataFrame({
        'Condition': [selected_condition],
        'Symptom': [selected_symptom]
    })

    user_input_encoded = encoder.transform(user_input)
    prediction = model.predict(user_input_encoded)
    
    st.markdown("---") 
    st.subheader('Recommended Medication:')
    
    st.success(f"**Based on your inputs, the suggested medication is:  `{prediction[0]}`**")

    st.warning("**Disclaimer:** This is a just a DSS,dont let the doctors run out of job")

else:
   
    st.info("Please select your condition and profile, then click 'Get Recommendation'.")
