# # ------------------ IMPORTS ------------------
# import torch
# try:
#     _ = list(torch.classes.__path__._path)
# except Exception:
#     pass  # ignore Streamlit watcher error

# import streamlit as st
# from PIL import Image
# import numpy as np
# import joblib
# from xgboost import XGBClassifier
# from ultralytics import YOLO

# # ------------------ LOAD MODELS ------------------
# try:
#     disease_model = YOLO("best.pt")
#     soil_model = joblib.load("soli_analysis.pkl")
# except Exception as e:
#     st.error(f"🔴 Error loading models: {e}")

# # ------------------ CLASS NAMES ------------------
# valid_classes = [
#     "Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf", "Bell_pepper leaf spot",
#     "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf",
#     "Peach leaf", "Potato leaf", "Potato leaf early blight", "Potato leaf late blight", "Raspberry leaf",
#     "Soyabean leaf", "Soybean leaf", "Squash Powdery mildew leaf", "Strawberry leaf",
#     "Tomato Early blight leaf", "Tomato Septoria leaf spot", "Tomato leaf", "Tomato leaf bacterial spot",
#     "Tomato leaf late blight", "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato mold leaf",
#     "Tomato two spotted spider mites leaf", "grape leaf", "grape leaf black rot"
# ]

# # ------------------ DISEASE DETECTION FUNCTION ------------------
# def detect_disease(image):
#     image.save("temp.jpg")
#     results = disease_model.predict("temp.jpg", save=True, stream=False, imgsz=640)
#     annotated = results[0].plot()
#     pred_image = Image.fromarray(annotated)
#     label = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "Unknown"
#     return label, pred_image

# # ------------------ SOIL FERTILITY FUNCTION ------------------
# def predict_soil_fertility_np(features_list):
#     arr = np.array(features_list, dtype=object).reshape(1, 12)
#     prediction = soil_model.predict(arr)[0]
#     soil_labels = ["Low Fertility", "Moderate Fertility", "High Fertility"]
#     recommendations = [
#         "Add organic manure and compost.",
#         "Apply balanced NPK fertilizers.",
#         "Great soil! Maintain it with cover crops."
#     ]
#     return soil_labels[prediction], recommendations[prediction]

# # ------------------ STREAMLIT UI ------------------
# st.set_page_config(page_title="🌿 AgriCare Tool", layout="centered")
# st.title("🌾 AgriCare AI: Plant Disease & Soil Analysis")

# # ------------- PLANT SECTION -------------
# st.header("🌿 Upload Plant Image for Disease Detection")
# st.info("📢 Note: This tool currently detects diseases in specific plants like tomato, potato, grape, corn, etc.")

# plant_image = st.file_uploader("Upload Plant Image (jpg/png)", type=["jpg", "jpeg", "png"])

# if plant_image and st.button("Analyze Plant"):
#     try:
#         img = Image.open(plant_image)
#         label, pred_img = detect_disease(img)
#         st.image(pred_img, caption=f"Detected: {label}")
#         st.success(f"Disease Detected: {label}")
#         if label not in valid_classes:
#             st.warning("⚠️ This disease is not yet supported by our system. Please try with a supported plant type.")
#     except Exception as e:
#         st.error(f"Error in plant analysis: {e}")

# # ------------- SOIL SECTION -------------
# st.header("🧪 Soil Nutrient Analysis")
# with st.form("soil_form"):
#     col1, col2 = st.columns(2)
#     with col1:
#         N = st.number_input("Nitrogen (N)", 0.0)
#         P = st.number_input("Phosphorus (P)", 0.0)
#         K = st.number_input("Potassium (K)", 0.0)
#         pH = st.number_input("pH Level", 0.0)
#         EC = st.number_input("Electrical Conductivity (EC)", 0.0)
#         OC = st.number_input("Organic Carbon", 0.0)
#     with col2:
#         S = st.number_input("Sulphur (S)", 0.0)
#         Zn = st.number_input("Zinc (Zn)", 0.0)
#         Fe = st.number_input("Iron (Fe)", 0.0)
#         Cu = st.number_input("Copper (Cu)", 0.0)
#         Mn = st.number_input("Manganese (Mn)", 0.0)
#         B = st.number_input("Boron (B)", 0.0)
#     submitted = st.form_submit_button("Analyze Soil")
# if submitted:
#     features = [N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]
#     try:
#         fertility, recommendation = predict_soil_fertility_np(features)
#         st.success(f"🌱 Soil Fertility: {fertility}")
#         st.info(f"✅ Recommendation: {recommendation}")
#     except Exception as e:
#         st.error(f"Error in soil analysis: {e}")

# ------------------ IMPORTS ------------------
# ==========-============================
# ========================================
# =====================================
try:
    _ = list(torch.classes.__path__._path)
except Exception:
    pass

import streamlit as st
from PIL import Image
import numpy as np
import joblib
from xgboost import XGBClassifier
from ultralytics import YOLO

# ------------------ CONFIG & STYLE ------------------
st.set_page_config(page_title="🌾 AgriCare AI", layout="centered", initial_sidebar_state="collapsed")

# Custom dark mode CSS
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-size: 16px;
        padding: 8px 24px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .css-1aumxhk {
        background-color: #0e1117 !important;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1e222a;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 AgriCare AI: Plant Disease & Soil Analysis")

# ------------------ LOAD MODELS ------------------
try:
    disease_model = YOLO("best.pt")
    soil_model = joblib.load("soli_analysis.pkl")
except Exception as e:
    st.error(f"🔴 Error loading models: {e}")

# ------------------ CLASS DEFINITIONS ------------------
valid_classes = [
    "Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf", "Bell_pepper leaf spot",
    "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf",
    "Peach leaf", "Potato leaf", "Potato leaf early blight", "Potato leaf late blight", "Raspberry leaf",
    "Soyabean leaf", "Soybean leaf", "Squash Powdery mildew leaf", "Strawberry leaf",
    "Tomato Early blight leaf", "Tomato Septoria leaf spot", "Tomato leaf", "Tomato leaf bacterial spot",
    "Tomato leaf late blight", "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato mold leaf",
    "Tomato two spotted spider mites leaf", "grape leaf", "grape leaf black rot"
]

# ------------------ FUNCTIONS ------------------
def detect_disease(image):
    image.save("temp.jpg")
    results = disease_model.predict("temp.jpg", save=True, stream=False, imgsz=640)
    annotated = results[0].plot()
    pred_image = Image.fromarray(annotated)
    label = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "Unknown"
    return label, pred_image

def predict_soil_fertility_np(features_list):
    arr = np.array(features_list, dtype=object).reshape(1, 12)
    prediction = soil_model.predict(arr)[0]
    soil_labels = ["Low Fertility", "Moderate Fertility", "High Fertility"]
    recommendations = [
        "Add organic manure and compost.",
        "Apply balanced NPK fertilizers.",
        "Great soil! Maintain it with cover crops."
    ]
    return soil_labels[prediction], recommendations[prediction]

# ------------------ TABS ------------------
tab1, tab2 = st.tabs(["🌿 Plant Disease Detection", "🧪 Soil Analysis"])

# ------------------ PLANT DISEASE TAB ------------------
with tab1:
    st.header("📷 Upload Plant Image")
    st.info("📢 This tool detects common diseases in tomato, potato, grape, corn, etc.")
    plant_image = st.file_uploader("Upload Plant Image (jpg/png)", type=["jpg", "jpeg", "png"])

    if plant_image and st.button("🔍 Analyze Plant"):
        try:
            img = Image.open(plant_image)
            label, pred_img = detect_disease(img)
            st.image(pred_img, caption=f"Detected: {label}")
            st.success(f"✅ Disease Detected: {label}")
            if label not in valid_classes:
                st.warning("⚠️ Disease not in our supported list. Try another common crop image.")
        except Exception as e:
            st.error(f"Error in plant analysis: {e}")

# ------------------ SOIL ANALYSIS TAB ------------------
with tab2:
    st.header("🌱 Soil Nutrient Inputs")
    with st.form("soil_form"):
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N)", 0.0)
            P = st.number_input("Phosphorus (P)", 0.0)
            K = st.number_input("Potassium (K)", 0.0)
            pH = st.number_input("pH Level", 0.0)
            EC = st.number_input("Electrical Conductivity (EC)", 0.0)
            OC = st.number_input("Organic Carbon", 0.0)
        with col2:
            S = st.number_input("Sulphur (S)", 0.0)
            Zn = st.number_input("Zinc (Zn)", 0.0)
            Fe = st.number_input("Iron (Fe)", 0.0)
            Cu = st.number_input("Copper (Cu)", 0.0)
            Mn = st.number_input("Manganese (Mn)", 0.0)
            B = st.number_input("Boron (B)", 0.0)

        submitted = st.form_submit_button("🧪 Analyze Soil")

    if submitted:
        features = [N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]
        try:
            fertility, recommendation = predict_soil_fertility_np(features)
            st.success(f"🌍 Soil Fertility Level: {fertility}")
            st.info(f"🌟 Recommendation: {recommendation}")
        except Exception as e:
            st.error(f"Error in soil analysis: {e}")
