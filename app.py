import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIGURATION ---
# This sets the browser tab title and widens the layout
st.set_page_config(page_title="Heart Risk AI", page_icon="🫀", layout="centered")

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Styling the main title */
    .main-title {
        text-align: center;
        color: #ff4b4b;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-text {
        text-align: center;
        color: #888888;
        font-size: 18px;
        margin-bottom: 30px;
    }
    /* Styling the Predict Button */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 20px;
        font-weight: bold;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff1a1a;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4);
        transform: scale(1.02);
    }
    /* Custom Result Cards */
    .high-risk {
        background-color: #ff4b4b15;
        border-left: 6px solid #ff4b4b;
        padding: 20px;
        border-radius: 8px;
        color: #ff4b4b;
    }
    .low-risk {
        background-color: #00cc6615;
        border-left: 6px solid #00cc66;
        padding: 20px;
        border-radius: 8px;
        color: #00cc66;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL AND SCALER ---
@st.cache_resource # This makes the app run faster by caching the model
def load_components():
    model = joblib.load('decision_tree_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_components()

# --- APP HEADER ---
st.markdown("<h1 class='main-title'>🫀 AI Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Enter patient parameters below for instant risk assessment.</p>", unsafe_allow_html=True)

# --- ORGANIZED INPUT SECTIONS ---
# Section 1: Demographics
with st.container():
    st.subheader("👤 Patient Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
    with col2:
        sex = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

st.divider()

# Section 2: Clinical Vitals
with st.container():
    st.subheader("🩺 Clinical Vitals")
    col3, col4 = st.columns(2)
    with col3:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with col4:
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
        restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])

st.divider()

# Section 3: Exercise & Heart Data
with st.container():
    st.subheader("🏃‍♂️ Exercise & Cardiac Data")
    col5, col6 = st.columns(2)
    with col5:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
        cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
        slope = st.selectbox("ST Segment Slope (0-2)", options=[0, 1, 2])
        thal = st.selectbox("Thalassemia Type (0-3)", options=[0, 1, 2, 3])
    with col6:
        exang = st.selectbox("Exercise Induced Angina?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-4)", options=[0, 1, 2, 3, 4])

st.write("") # Spacer

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
    
    # Predict
    with st.spinner("Analyzing parameters..."):
        prediction = model.predict(input_encoded)
    
    # Display styled results
    if prediction[0] == 1:
        st.markdown("""
        <div class="high-risk">
            <h2>⚠️ High Risk Detected</h2>
            <p style="font-size:18px;">The AI model indicates a <b>high likelihood</b> of heart disease based on the provided parameters. Immediate medical consultation is highly recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="low-risk">
            <h2>✅ Low Risk Detected</h2>
            <p style="font-size:18px;">The AI model indicates a <b>low likelihood</b> of heart disease. Patient parameters are within normal baseline ranges.</p>
        </div>
        """, unsafe_allow_html=True)
