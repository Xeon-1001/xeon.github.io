import streamlit as st
import pandas as pd
import warnings
import joblib
from datetime import datetime

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="ML-Powered Drug DSS",
    page_icon="🤖",
    layout="wide"
)

# --- 2. Load Drug Data (Cached) ---
@st.cache_data
def load_drug_data():
    """
    Loads ALL datasets and merges them.
    Guarantees a return of (DataFrame, bool) or (None, False).
    """
    try:
        # Load the main drug database
        df_drugs = pd.read_csv('Data sets/drugs_side_effects_drugs_com.csv')
        df_drugs.columns = df_drugs.columns.str.strip() # Clean columns
        
        # --- Prepare main filter columns ---
        df_drugs['pregnancy_category'] = df_drugs['pregnancy_category'].fillna('N')
        df_drugs['alcohol'] = df_drugs['alcohol'].fillna('N')
        df_drugs['csa'] = df_drugs['csa'].fillna('N') 
        df_drugs['rx_otc'] = df_drugs['rx_otc'].fillna('Rx')
        df_drugs['rating'] = pd.to_numeric(df_drugs['rating'], errors='coerce').fillna(5.0)
        df_drugs['drug_name'] = df_drugs['drug_name'].astype(str).str.strip()

        # --- Load and merge NEW contraindications ---
        adv_filters_loaded = False
        try:
            df_contra = pd.read_csv('Data sets/new_contraindications.csv')
            df_contra.columns = df_contra.columns.str.strip() # Clean columns
            df_contra['drug_name'] = df_contra['drug_name'].astype(str).str.strip()
            
            df_drugs = pd.merge(df_drugs, df_contra, on="drug_name", how="left")
            adv_filters_loaded = True
            
            new_cols = ['is_safe_pediatric', 'is_safe_geriatric', 'is_safe_breastfeeding', 
                        'kidney_disease', 'liver_disease', 'smoker_interaction', 
                        'diabetic_interaction', 'asthmatic_interaction', 'heart_condition']
            for col in new_cols:
                if col in df_drugs.columns:
                    df_drugs[col] = df_drugs[col].fillna('safe')

        except FileNotFoundError:
            st.warning("`new_contraindications.csv` not found. Advanced filters will not be available.", icon="⚠️")
            # Note: We continue without this file, adv_filters_loaded remains False.

        df_drugs['medical_condition'] = df_drugs['medical_condition'].astype(str).str.strip()
        
        # --- Success ---
        return df_drugs, adv_filters_loaded
    
    except FileNotFoundError:
        st.error("Error: 'drugs_side_effects_drugs_com.csv' not found. App cannot load.")
        # --- MUST RETURN TWO VALUES ---
        return None, False
    
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        # --- MUST RETURN TWO VALUES ---
        return None, False


# --- 3. Load ML Model (Cached) ---
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('disease_predictor_model.pkl')
        encoder = joblib.load('disease_encoder.pkl')
        return model, encoder
    except FileNotFoundError:
        st.error("ML Model files not found. Please run 'model.py' first.", icon="🔥")
        return None, None
df_drugs, adv_filters_loaded = load_drug_data()
ml_model, disease_encoder = load_ml_model()


# --- FIXED: The "Rosetta Stone" Disease Map ---
# Translates (ML Prediction) -> (Drug Database Condition)
DISEASE_MAP = {
    # ML Prediction (from your list) : Drug DB Condition (from your images)
    'Acne': 'Acne',
    'Allergic Rhinitis': 'Allergies',
    "Alzheimer's Disease": "Alzheimer's",
    'Anemia': 'Anemia', # Assuming this is in your drug list
    'Anxiety Disorders': 'Anxiety',
    'Appendicitis': 'Appendicitis', # Not in images, but mapping 1-to-1
    'Asthma': 'Asthma',
    'Atherosclerosis': 'Atherosclerosis', # Not in images
    'Autism Spectrum Disorder (ASD)': 'Autism', # Guessing
    'Bipolar Disorder': 'Bipolar Disorder',
    'Bladder Cancer': 'Cancer',
    'Brain Tumor': 'Cancer', # Guessing
    'Breast Cancer': 'Cancer',
    'Bronchitis': 'Bronchitis',
    'Cataracts': 'Cataracts', # Not in images
    'Cerebral Palsy': 'Cerebral Palsy', # Not in images
    'Chickenpox': 'Chickenpox', # Not in images
    'Cholecystitis': 'Gastrointestinal', # Guessing
    'Cholera': 'Cholera', # Not in images
    'Chronic Kidney Disease': 'Kidney Disease',
    'Chronic Obstructive Pulmonary Disease (COPD)': 'COPD',
    'Chronic Obstructive Pulmonary...': 'COPD',
    'Cirrhosis': 'Liver Disease',
    'Colorectal Cancer': 'Cancer',
    'Common Cold': 'Colds & Flu',
    'Conjunctivitis (Pink Eye)': 'Conjunctivitis', # Not in images
    'Coronary Artery Disease': 'Heart Condition', # Guessing
    "Crohn's Disease": 'IBD (Bowel)',
    'Cystic Fibrosis': 'Cystic Fibrosis', # Not in images
    'Dementia': 'Dementia', # Not in images
    'Dengue Fever': 'Dengue Fever', # Not in images
    'Depression': 'Depression',
    'Diabetes': 'Diabetes (Type 2)', # Defaulting to Type 2
    'Diverticulitis': 'Gastrointestinal', # Guessing
    'Down Syndrome': 'Down Syndrome', # Not in images
    'Eating Disorders (Anorexia,...)': 'Eating Disorders', # Guessing
    'Ebola Virus': 'Ebola Virus', # Not in images
    'Eczema': 'Eczema',
    'Endometriosis': 'Endometriosis', # Not in images
    'Epilepsy': 'Seizures',
    'Esophageal Cancer': 'Cancer',
    'Fibromyalgia': 'Fibromyalgia', # Not in images
    'Gastroenteritis': 'Gastrointestinal',
    'Glaucoma': 'Glaucoma', # Not in images
    'Gout': 'Gout',
    'HIV/AIDS': 'AIDS/HIV',
    'Hemophilia': 'Hemophilia', # Not in images
    'Hemorrhoids': 'Hemorrhoids', # Not in images
    'Hepatitis': 'Hepatitis',
    'Hepatitis B': 'Hepatitis',
    'Hyperglycemia': 'Diabetes (Type 2)', # Guessing
    'Hypertension': 'Hypertension',
    'Hypertensive Heart Disease': 'Heart Condition', # Guessing
    'Hyperthyroidism': 'Hyperthyroidism',
    'Hypoglycemia': 'Hypoglycemia', # Not in images
    'Hypothyroidism': 'Hypothyroidism',
    'Influenza': 'Colds & Flu',
    'Kidney Cancer': 'Cancer',
    'Kidney Disease': 'Kidney Disease',
    'Klinefelter Syndrome': 'Klinefelter Syndrome', # Not in images
    'Liver Cancer': 'Cancer',
    'Liver Disease': 'Liver Disease',
    'Lung Cancer': 'Cancer',
    'Lyme Disease': 'Lyme Disease', # Not in images
    'Lymphoma': 'Cancer',
    'Malaria': 'Malaria', # Not in images
    'Marfan Syndrome': 'Marfan Syndrome', # Not in images
    'Measles': 'Measles', # Not in images
    'Melanoma': 'Cancer',
    'Migraine': 'Migraine',
    'Multiple Sclerosis': 'Multiple Sclerosis', # Not in images
    'Mumps': 'Mumps', # Not in images
    'Muscular Dystrophy': 'Muscular Dystrophy', # Not in images
    'Myocardial Infarction (Heart...)': 'Heart Condition',
    'Obsessive-Compulsive Disorde...': 'OCD', # Guessing
    'Osteoarthritis': 'Osteoarthritis',
    'Osteomyelitis': 'Osteomyelitis', # Not in images
    'Osteoporosis': 'Osteoporosis',
    'Otitis Media (Ear Infection)': 'Ear Infection', # Guessing
    'Ovarian Cancer': 'Cancer',
    'Pancreatic Cancer': 'Cancer',
    'Pancreatitis': 'Pancreatitis', # Not in images
    "Parkinson's Disease": "Parkinson's Disease", # Guessing
    'Pneumocystis Pneumonia (PCP)': 'Pneumonia',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax', # Not in images
    'Polio': 'Polio', # Not in images
    'Polycystic Ovary Syndrome (PCOS)': 'PCOS', # Guessing
    'Prader-Willi Syndrome': 'Prader-Willi Syndrome', # Not in images
    'Prostate Cancer': 'Cancer',
    'Psoriasis': 'Psoriasis',
    'Rabies': 'Rabies', # Not in images
    'Rheumatoid Arthritis': 'Rheumatoid Arthritis',
    'Rubella': 'Rubella', # Not in images
    'Schizophrenia': 'Schizophrenia',
    'Scoliosis': 'Scoliosis', # Not in images
    'Sepsis': 'Sepsis', # Not in images
    'Sickle Cell Anemia': 'Anemia',
    'Sinusitis': 'Sinusitis', # Not in images
    'Sleep Apnea': 'Sleep Apnea', # Not in images
    'Spina Bifida': 'Spina Bifida', # Not in images
    'Stroke': 'Stroke',
    'Systemic Lupus Erythematosus...': 'Lupus', # Guessing
    'Testicular Cancer': 'Cancer',
    'Tetanus': 'Tetanus', # Not in images
    'Thyroid Cancer': 'Cancer',
    'Tonsillitis': 'Tonsillitis', # Not in images
    'Tourette Syndrome': 'Tourette Syndrome', # Not in images
    'Tuberculosis': 'Tuberculosis', # Not in images
    'Turner Syndrome': 'Turner Syndrome', # Not in images
    'Typhoid Fever': 'Typhoid Fever', # Not in images
    'Ulcerative Colitis': 'IBD (Bowel)',
    'Urinary Tract Infection': 'UTI',
    'Urinary Tract Infection (UTI)': 'UTI',
    'Williams Syndrome': 'Williams Syndrome', # Not in images
    'Zika Virus': 'Zika Virus' # Not in images
}
# --- END NEW MAP ---


# --- 4. User Interface (Sidebar) ---
st.title("🤖 Smart ML-Powered Drug DSS")
st.markdown("Enter your symptoms to get a condition prediction, then see safe drug options based on your profile.")
st.markdown("---")

with st.sidebar:
    st.image("https://i.imgur.com/v80pD1O.png", width=50) # A little icon
    st.header("Step 1: Patient Profile")
    
    st.subheader("Symptom Checker")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Female", "Male"]) # Matching training data
    blood_pres = st.selectbox("Blood Pressure", ["Normal", "High", "Low"])
    cholesterol = st.selectbox("Cholesterol", ["Normal", "High"])
    
    col1, col2 = st.columns(2)
    with col1:
        fever = st.checkbox("Fever", value=False)
        cough = st.checkbox("Cough", value=False)
    with col2:
        fatigue = st.checkbox("Fatigue", value=False)
        diff_e = st.checkbox("Difficulty Breathing", value=False)
    
    st.markdown("---")
    st.header("Step 2: Drug Safety Filter")
    st.caption("Apply these filters to the drug recommendations.")

    is_pregnant = st.checkbox("Pregnant", value=False)
    is_breastfeeding = st.checkbox("Breastfeeding", value=False)
    is_alcohol_consumer = st.checkbox("Frequent Alcohol Consumer", value=False)
    is_smoker = st.checkbox("Smoker", value=False)
    is_pediatric = st.checkbox("Patient is a Child", value=False)
    is_geriatric = st.checkbox("Patient is Elderly", value=False)
    
    st.subheader("Medical History")
    has_kidney_disease = st.checkbox("Kidney Disease", value=False)
    has_liver_disease = st.checkbox("Liver Disease", value=False)
    is_diabetic = st.checkbox("Diabetic", value=False)
    is_asthmatic = st.checkbox("Asthmatic", value=False)
    has_heart_condition = st.checkbox("Heart Condition", value=False)

    st.subheader("Drug Preferences")
    avoid_csa_drugs = st.checkbox("Avoid controlled (addictive) drugs?", value=True)
    show_otc_only = st.checkbox("Show only Over-the-Counter (OTC)?", value=False)

    st.markdown("---")
    allergies_input = st.text_input("Known Allergies (comma-separated)")
    
    # The main button
    run_button = st.button("Get Diagnosis & Recommendations", type="primary", use_container_width=True)


# --- 5. Main Logic (Triggered by Button) ---
# --- FIXED: Check if DataFrames are 'is not None' ---
if run_button and ml_model is not None and df_drugs is not None:
    
    # --- PART 1: ML PREDICTION ---
    
    # --- FIXED: Use correct column names ---
    input_data = {
        'Fever': 1 if fever else 0,
        'Cough': 1 if cough else 0,
        'Fatigue': 1 if fatigue else 0,
        'Difficulty Breathing': 1 if diff_e else 0,
        'Age': age,
        'Gender': gender,
        'Blood Pressure': blood_pres,
        'Cholesterol Level': cholesterol
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Run the prediction
    try:
        prediction_encoded = ml_model.predict(input_df)
        predicted_disease = disease_encoder.inverse_transform(prediction_encoded)[0]
        
        
        # --- ADDED: Use the DISEASE_MAP to translate ---
        
        # 1. 'predicted_disease' is what the ML model said (e.g., 'Influenza')
        # 2. Use .get() to find it in the map. If not found, use the original prediction.
        mapped_condition = DISEASE_MAP.get(predicted_disease, predicted_disease)
        
        # 3. 'mapped_condition' is what the drug DB understands (e.g., 'Colds & Flu')
        st.header(f"🧑‍⚕️ ML-Predicted Condition: {predicted_disease}")
        
        if mapped_condition == predicted_disease:
             st.info(f"Showing drug options for: **{predicted_disease}**.")
        else:
             st.info(f"The ML model predicted **{predicted_disease}**. Showing drug options for the matching condition: **{mapped_condition}**.")
        
        # 4. Use the MAPPED condition for filtering
        selected_condition = mapped_condition 
        
        # --- END OF ADDED SECTION ---

        
        # --- PART 2: HEURISTIC DRUG FILTERING ---
        
        # Step 2.1: Find all drugs for the predicted condition
        results_df = df_drugs[df_drugs['medical_condition'] == selected_condition].copy()
        
        if results_df.empty:
            st.error(f"No drugs found in our database for '{selected_condition}'. Please consult a doctor.")
        
        else:
            warnings_list = [f"Showing drugs for '{selected_condition}'."]
            
            # --- Step 2.2: Apply HARD SAFETY Filters ---
            if is_pregnant:
                results_df = results_df[~results_df['pregnancy_category'].isin(['D', 'X'])]
                warnings_list.append("Filtered: Pregnancy (Category D, X).")
            if is_alcohol_consumer:
                results_df = results_df[results_df['alcohol'] != 'X']
                warnings_list.append("Filtered: Alcohol interactions.")
            if avoid_csa_drugs:
                results_df = results_df[results_df['csa'] == 'N']
                warnings_list.append("Filtered: Controlled substances (CSA).")
            if show_otc_only:
                results_df = results_df[results_df['rx_otc'].str.contains("OTC", case=False, na=False)]
                warnings_list.append("Filtered: OTC drugs only.")
            if allergies_input:
                allergy_list = [a.strip().lower() for a in allergies_input.split(',')]
                mask = ~results_df['drug_name'].str.lower().isin(allergy_list)
                results_df = results_df[mask]
                warnings_list.append(f"Filtered: Allergies ({allergies_input}).")

            if adv_filters_loaded:
                if is_breastfeeding and 'is_safe_breastfeeding' in results_df.columns:
                    results_df = results_df[results_df['is_safe_breastfeeding'] != 'unsafe']
                    warnings_list.append("Filtered: Breastfeeding safety.")
                if is_pediatric and 'is_safe_pediatric' in results_df.columns:
                    results_df = results_df[results_df['is_safe_pediatric'] != 'unsafe']
                    warnings_list.append("Filtered: Pediatric safety.")
                if is_geriatric and 'is_safe_geriatric' in results_df.columns:
                    results_df = results_df[results_df['is_safe_geriatric'] != 'unsafe']
                    warnings_list.append("Filtered: Geriatric safety.")
                if has_kidney_disease and 'kidney_disease' in results_df.columns:
                    results_df = results_df[results_df['kidney_disease'] != 'unsafe']
                    warnings_list.append("Filtered: Kidney Disease interactions.")
                if has_liver_disease and 'liver_disease' in results_df.columns:
                    results_df = results_df[results_df['liver_disease'] != 'unsafe']
                    warnings_list.append("Filtered: Liver Disease interactions.")
                if is_smoker and 'smoker_interaction' in results_df.columns:
                    results_df = results_df[results_df['smoker_interaction'] != 'unsafe']
                    warnings_list.append("Filtered: Smoker interactions.")
                if is_diabetic and 'diabetic_interaction' in results_df.columns:
                    results_df = results_df[results_df['diabetic_interaction'] != 'unsafe']
                    warnings_list.append("Filtered: Diabetic interactions.")
                if is_asthmatic and 'asthmatic_interaction' in results_df.columns:
                    results_df = results_df[results_df['asthmatic_interaction'] != 'unsafe']
                    warnings_list.append("Filtered: Asthmatic interactions.")
                if has_heart_condition and 'heart_condition' in results_df.columns:
                    results_df = results_df[results_df['heart_condition'] != 'unsafe']
                    warnings_list.append("Filtered: Heart Condition interactions.")
            
            # --- Step 2.3: HEURISTIC SCORING ---
            if not results_df.empty:
                results_df['suitability_score'] = results_df['rating']

                # Apply penalties for 'caution' items (the "smart" part)
                if has_kidney_disease and 'kidney_disease' in results_df.columns:
                    penalty = results_df['kidney_disease'].apply(lambda x: -1.5 if x == 'use_with_caution' else 0)
                    results_df['suitability_score'] += penalty
                if is_geriatric and 'is_safe_geriatric' in results_df.columns:
                    penalty = results_df['is_safe_geriatric'].apply(lambda x: -1.5 if x == 'use_with_caution' else 0)
                    results_df['suitability_score'] += penalty
                
                results_df['suitability_score'] = results_df['suitability_score'].clip(lower=0.1)
                
                final_recommendations = results_df.sort_values(by="suitability_score", ascending=False)
            
            else:
                final_recommendations = results_df

            # --- Step 2.4: Display Results ---
            st.subheader(f"Recommended Safe Medications for '{selected_condition}'")
            if final_recommendations.empty:
                st.warning("No medications found matching your strict safety profile. Please consult a doctor.")
            else:
                st.success(f"Found {len(final_recommendations)} suitable recommendations.")
                
                with st.expander("Filtering & Ranking Notes"):
                    for note in warnings_list: st.write(f"✓ {note}")
                
                for _, row in final_recommendations.head(10).iterrows():
                    rating_display = f"**Suitability: {row['suitability_score']:.1f}/10** (Generic: {row['rating']}/10)"
                    with st.expander(f"**{row['drug_name']}** ({rating_display})"):
                        st.markdown(f"**Generic Name:** {row['generic_name']}")
                        st.markdown(f"**Brand Names:** {row['brand_names']}")
                        st.markdown(f"**Type:** {row['rx_otc']} | **Pregnancy:** {row['pregnancy_category']} | **CSA:** {row['csa']}")
                        st.markdown(f"**Known Side Effects:**")
                        st.text(row['side_effects'])
                        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("This may be due to a mismatch in training data. Have you re-trained the model?")

# --- FIXED: Use 'is None' for robust checking ---
elif ml_model is None or df_drugs is None:
    st.warning("Data or models are still loading, or failed to load. Please check the terminal and refresh.")
else:
    st.info("Please fill out your profile in the sidebar and click the button to get recommendations.")
