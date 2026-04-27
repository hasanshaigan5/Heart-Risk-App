import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIGURATION ---
# Switched to a universally supported stethoscope emoji
st.set_page_config(page_title="Heart Risk AI", page_icon="🩺", layout="centered")

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* 1. Add a beautiful soft gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 2. Reduce the empty white space at the top and between elements */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 3. Style the Title and Catchy Phrase */
    .main-title {
        text-align: center;
        color: #1E3A8A; /* Deep professional blue */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .catchy-phrase {
        text-align: center;
        color: #3B82F6; /* Lighter accent blue */
        font-size: 1.2rem;
        font-style: italic;
        font-weight: 500;
        margin-top: 5px;
        margin-bottom: 25px;
    }
    
    /* 4. Style the section headers to look like neat dividers */
    h3 {
        color: #1E3A8A;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 5px;
        margin-bottom: 10px;
        font-size: 1.5rem;
    }
    
    /* 5. Predict Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff1a1a 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 20px;
        font-weight: bold;
        width: 100%;
        border: none;
        transition: all 0.3s ease;
        margin-top: 20px;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0px 8px 20px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    
    /* 6. Custom Result Cards */
    .high-risk {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 6px solid #ff4b4b;
        padding: 20px;
        border-radius: 8px;
        color: #990000;
        margin-top: 20px;
    }
    .low-risk {
        background-color: rgba(0, 204, 102, 0.1);
        border-left: 6px solid #00cc66;
        padding: 20px;
        border-radius: 8px;
        color: #006633;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL AND SCALER ---
@st.cache_resource 
def load_components():
    model = joblib.load('decision_tree_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_components()

# --- APP HEADER ---
st.markdown("<h1 class='main-title'>❤️ CardioCare AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='catchy-phrase'>Empowering your heart's future with the precision of Artificial Intelligence.</p>", unsafe_allow_html=True)

# --- ORGANIZED INPUT SECTIONS (Compressed spacing) ---
st.markdown("### 👤 Patient Demographics")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
with col2:
    sex = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

st.markdown("### 🩺 Clinical Vitals")
col3, col4 = st.columns(2)
with col3:
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
with col4:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])

st.markdown("### 🏃‍♂️ Exercise & Cardiac Data")
col5, col6 = st.columns(2)
with col5:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    slope = st.selectbox("ST Segment Slope (0-2)", options=[0, 1, 2])
    thal = st.selectbox("Thalassemia Type (0-3)", options=[0, 1, 2, 3])
with col6:
    exang = st.selectbox("Exercise Induced Angina?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-4)", options=[0, 1, 2, 3, 4])


# --- PREDICTION LOGIC ---
if st.button("Analyze Patient Data"):
    
    # Structure inputs into dataframe
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
        'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    
    # List of dummy columns exactly as trained
    expected_columns = [
        'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
        'cp_1', 'cp_2', 'cp_3', 'restecg_1', 'restecg_2', 'slope_1', 'slope_2',
        'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_1', 'thal_2', 'thal_3'
    ]
    
    # Create dummies
    categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols)
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # Scale numericals
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    
    # Predict with a loading spinner for professional feel
    with st.spinner("Calculating risk profile..."):
        prediction = model.predict(input_encoded)
    
    # Display styled results
    if prediction[0] == 1:
        st.markdown("""
        <div class="high-risk">
            <h3 style="color:#990000; border-bottom: none; margin-bottom: 5px;">⚠️ High Risk Profile Detected</h3>
            <p style="font-size:16px; margin-top: 0px;">The predictive model indicates a <b>high likelihood</b> of heart disease based on the provided parameters. Medical consultation is recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="low-risk">
            <h3 style="color:#006633; border-bottom: none; margin-bottom: 5px;">✅ Low Risk Profile</h3>
            <p style="font-size:16px; margin-top: 0px;">The predictive model indicates a <b>low likelihood</b> of heart disease. Patient parameters appear to be within a safe baseline.</p>
        </div>
        """, unsafe_allow_html=True)
