import streamlit as st

# --- 1. Knowledge Base (A simple substitute for a database or ML model) ---
# This dictionary holds our medical data and decision rules.
# Keys are diseases. Values are details about the standard treatment and alternatives.
drug_database = {
    "Common Cold": {
        "normal_drug": "Ibuprofen",
        "pregnancy_safe": False,
        "alcohol_interaction": True,
        "alternative_drug": "Paracetamol"
    },
    "Bacterial Infection": {
        "normal_drug": "Doxycycline",
        "pregnancy_safe": False,
        "alcohol_interaction": True,
        "alternative_drug": "Amoxicillin"
    },
    "Acid Reflux": {
        "normal_drug": "Omeprazole",
        "pregnancy_safe": True,
        "alcohol_interaction": False,
        "alternative_drug": None # No common alternative needed
    },
    "Migraine": {
        "normal_drug": "Sumatriptan",
        "pregnancy_safe": False,
        "alcohol_interaction": False,
        "alternative_drug": "Paracetamol"
    },
    "Insomnia": {
        "normal_drug": "Zolpidem",
        "pregnancy_safe": False,
        "alcohol_interaction": True,
        "alternative_drug": "Melatonin (Consult Doctor)"
    }
}

# --- 2. User Interface (The part you see and interact with) ---
st.title("Symptom-Based Drug Recommendation DSS")
st.markdown("### A Prototype for MAT5024")

# Create a list of diseases from our database keys for the dropdown menu
disease_list = list(drug_database.keys())

# User inputs
selected_disease = st.selectbox("Step 1: Select the diagnosed disease", options=disease_list)
user_condition = st.radio(
    "Step 2: Select your personal profile",
    options=["Normal", "Pregnant", "Alcohol Consumer"],
    horizontal=True
)

# A button to trigger the decision logic
recommend_button = st.button("Get Recommendation")


# --- 3. Decision Logic (The "Brain" of the DSS) ---
if recommend_button:
    # Retrieve all information for the selected disease
    drug_info = drug_database[selected_disease]
    
    # Default recommendation is the standard drug
    final_recommendation = drug_info["normal_drug"]
    reason = f"The standard recommended medication for **{selected_disease}** is **{final_recommendation}**."
    
    # Rule 1: Check for pregnancy contraindication
    if user_condition == "Pregnant" and not drug_info["pregnancy_safe"]:
        final_recommendation = drug_info["alternative_drug"]
        reason = (f"**Warning:** The standard drug ({drug_info['normal_drug']}) is not recommended during pregnancy. "
                  f"A safer alternative is **{final_recommendation}**.")

    # Rule 2: Check for alcohol interaction
    elif user_condition == "Alcohol Consumer" and drug_info["alcohol_interaction"]:
        final_recommendation = drug_info["alternative_drug"]
        reason = (f"**Warning:** The standard drug ({drug_info['normal_drug']}) has known interactions with alcohol. "
                  f"A safer alternative is **{final_recommendation}**.")

    # --- 4. Output (Display the result) ---
    st.markdown("---")
    st.subheader("Your Recommendation:")
    
    if "Warning" in reason:
        st.warning(reason)
    else:
        st.success(reason)

    if final_recommendation is None:
        st.error("No suitable alternative medication found in the database. Please consult a doctor.")
    
    st.info("Disclaimer: This is a prototype DSS for academic purposes only. Always consult a healthcare professional before taking any medication.")