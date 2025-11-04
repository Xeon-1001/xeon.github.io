import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("Starting model training...")

# --- 1. Load Data ---
try:
    df = pd.read_csv('Data sets/Disease_symptom_CLEANED.csv')
    
    # Clean column names (still good practice)
    df.columns = df.columns.str.strip()

except FileNotFoundError:
    print("Error: 'Data sets/Disease_symptom_CLEANED.csv' not found.")
    print("Please make sure your data is in the 'Data sets' folder.")
    exit()

# --- 2. Define Features (X) and Target (y) ---
target = 'Disease'

# --- THIS IS THE CORRECTED LIST ---
features = [
    'Fever', 
    'Cough', 
    'Fatigue', 
    'Difficulty Breathing', # <-- Fixed
    'Age', 
    'Gender', 
    'Blood Pressure',       # <-- Fixed
    'Cholesterol Level'     # <-- Fixed
]

# --- Check if all features exist ---
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    print(f"Error: The following columns are still missing: {missing_cols}")
    print("This should not happen. Please check for typos.")
    exit()

X = df[features]
y = df[target]

print(f"Loaded {len(df)} records.")

# --- 3. Encode the Target Variable (y) ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'disease_encoder.pkl')
print("Saved target encoder (disease_encoder.pkl).")


# --- 4. Define Preprocessing Pipeline ---
numeric_features = ['Age']

# --- CORRECTED LIST ---
categorical_features = ['Gender', 'Blood Pressure', 'Cholesterol Level']

# --- CORRECTED LIST ---
binary_features = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']

# Map Yes/No to 1/0
X.loc[:, binary_features] = X[binary_features].replace({'Yes': 1, 'No': 0})

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('binary', 'passthrough', binary_features),
        ('num', 'passthrough', numeric_features)
    ],
    remainder='drop' 
)

# --- 5. Create and Train the Model Pipeline ---
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Training Random Forest model on {len(X_train)} samples...")
model_pipeline.fit(X_train, y_train)

# --- 6. Evaluate and Save ---
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model training complete. Test Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model_pipeline, 'disease_predictor_model.pkl')
print("Model saved as 'disease_predictor_model.pkl'.")
