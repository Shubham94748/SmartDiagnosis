import streamlit as st
import pickle
import numpy as np
import time

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="MediPredict AI", page_icon="🏥", layout="wide")

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# --- 2. DYNAMIC THEME ENGINE ---
# This ensures text is ALWAYS visible by using contrasting variables
def inject_theme():
    if st.session_state.theme == "dark":
        bg, card, text, subtext = "#0E1117", "#1E1E1E", "#FFFFFF", "#B0B0B0"
    else:
        bg, card, text, subtext = "#F0F2F6", "#FFFFFF", "#262730", "#555555"

    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        .main-card {{
            background-color: {card};
            color: {text};
            padding: 25px;
            border-radius: 15px;
            border: 1px solid rgba(128,128,128,0.2);
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1, h2, h3, h4, span, label, p {{ color: {text} !important; }}
        .stCheckbox label p {{ color: {text} !important; }}
        </style>
    """, unsafe_allow_html=True)

inject_theme()

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        return model, encoder
    except:
        return None, None

model, encoder = load_assets()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    is_dark = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"))
    if is_dark != (st.session_state.theme == "dark"):
        st.session_state.theme = "dark" if is_dark else "light"
        st.rerun()

# --- 5. MAIN UI ---
st.title("🏥 Disease Prediction System")

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("Step 1: Select Symptoms")
col1, col2 = st.columns(2)

symptoms_list = ['fever', 'headache', 'nausea', 'vomiting', 'fatigue', 
                 'joint_pain', 'skin_rash', 'cough', 'weight_loss', 'yellow_eyes']
selections = {}

for i, s in enumerate(symptoms_list):
    col = col1 if i % 2 == 0 else col2
    selections[s] = col.checkbox(s.replace('_', ' ').title())
st.markdown('</div>', unsafe_allow_html=True)

# --- 6. PREDICTION LOGIC ---
if st.button("Generate Result"):
    # Check if any option is selected
    if not any(selections.values()):
        st.success("✨ **You are fit!** No symptoms were selected.")
    else:
        if model:
            with st.spinner("Analyzing..."):
                time.sleep(0.8)
                input_data = np.array([list(selections.values())]).astype(int)
                pred = model.predict(input_data)
                disease = encoder.inverse_transform(pred)[0]
                
                st.markdown(f"""
                    <div class="main-card" style="border-left: 10px solid #2E7D32;">
                        <h3>Prediction Result</h3>
                        <p>Based on your input, the most likely condition is:</p>
                        <h2 style="color: #2E7D32 !important;">{disease}</h2>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Model files missing. Please upload model.pkl and encoder.pkl.")

st.caption("Disclaimer: This tool is for information only. Consult a doctor for medical advice.")
