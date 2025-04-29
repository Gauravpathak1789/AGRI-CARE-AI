import streamlit as st
from PIL import Image
import numpy as np
import xgboost 
import joblib
from ultralytics import YOLO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import torch
import warnings

# Suppress the specific XGBoost warning
warnings.filterwarnings("ignore", message=".*XGBoost.*")

torch.classes.__path__ = []
# ------------------ CONFIG & STYLE ------------------
st.set_page_config(page_title="🌾 AgriCare AI", layout="centered", initial_sidebar_state="collapsed")

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
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1e222a;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 AgriCare AI: Disease Detection, Soil & Smart Review Analyzer")

# ------------------ LOAD MODELS ------------------
bost = xgboost.Booster()
try:
    disease_model = YOLO("best.pt")
    soil_model = joblib.load("soli_analysis.pkl")
except Exception as e:
    st.error(f"🔴 Error loading models: {e}")

valid_classes = ["Tomato Early blight leaf", "Potato leaf early blight", "Tomato leaf", "Tomato mold leaf"]

# ------------------ FUNCTIONS ------------------


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def detect_disease(image):
    image.save("temp.jpg")
    results = disease_model.predict("temp.jpg", save=True, stream=False, imgsz=640)
    annotated = results[0].plot()
    pred_image = Image.fromarray(annotated)
    label = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "Unknown"
    return label, pred_image

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

def analyze_review_with_gemini_hindi(review_text):
    GOOGLE_API_KEY = "AIzaSyCaP65ZQSalRwjqUDFaJzo89JmYUcAyjmw"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
    prompt = f"""कृपया निम्नलिखित ग्राहक समीक्षा का विश्लेषण करें और इसे सरल हिंदी में संरचित तरीके से प्रस्तुत करें। मुख्य भावनाओं, सकारात्मक और नकारात्मक पहलुओं, और सुधार के सुझावों पर ध्यान केंद्रित करें।

    समीक्षा:
    {review_text}
    """
    result = model.invoke([HumanMessage(content=prompt)])
    return result.content

def generate_hindi_gpt_advice(plant_name, disease_label):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    GOOGLE_API_KEY = "AIzaSyCaP65ZQSalRwjqUDFaJzo89JmYUcAyjmw"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

    prompt = (
        f"मैं एक किसान हूँ। मेरे '{plant_name}' पौधे को '{disease_label}' नाम की बीमारी हो गई है। "
        f"कृपया मुझे कृषि विशेषज्ञ की तरह सलाह दें:\n"
        f"1. इस बीमारी के लक्षण क्या हैं?\n"
        f"2. यह बीमारी कैसे और क्यों होती है?\n"
        f"3. इसके इलाज के लिए कौन से जैविक (ऑर्गेनिक) उपाय या दवाइयाँ उपयोग की जा सकती हैं?\n"
        f"4. घर पर उपयोग किए जा सकने वाले घरेलू नुस्खे या उपाय भी बताएं।\n"
        f"5. इस बीमारी से बचाव के उपाय भी बताएं।\n"
        f"कृपया जवाब सरल हिंदी में दें, ताकि किसान आसानी से समझ सके।"
    )

    response = model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ------------------ TABS ------------------
tab1, tab2, tab3= st.tabs(["🌿 Plant Disease Detection", "🧪 Soil Analysis", "🧠 Smart Review Analysis"])

# ----------- TAB 1: PLANT DISEASE DETECTION ------------
with tab1:
    st.subheader("🎋😄:-'पत्तियों में छुपा राज़, हम बताएंगे आज!'")
    
    uploaded_img = st.file_uploader("📤 Upload  Plant Image", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        label, pred_img = detect_disease(img)

        st.image(pred_img, caption=f"Detected: {label}", use_container_width=True)
        st.caption(f"📷 Image shows: {label}")

        plant_name = st.text_input("Enter the plant name (e.g., Potato, tomato, etc.):")

        # If the user has provided a plant name
        if plant_name:
            st.write("🧠 GPT Output (Hindi):")
            try:
                response_text = generate_hindi_gpt_advice(plant_name, label)  # Pass both plant_name and disease label
                st.success(response_text)
            except Exception as e:
                st.error("❌ Error generating GPT response.")
                st.exception(e)
        else:
            st.info("Please enter the plant name to get advice.")

    else:
        st.info("👆 Upload an image above to detect disease and get recommendations.")





# ----------- TAB 2: SOIL ANALYSIS -------------
def predict_soil_fertility_np(features_list):
    arr = np.array(features_list, dtype=object).reshape(1, 12)
    prediction = soil_model.predict(arr)[0]
    soil_labels = ["Low Fertility", "Moderate Fertility", "High Fertility"]
    return soil_labels[prediction]

def get_soil_recommendations_with_gemini(inputs, fertility_result):
    GOOGLE_API_KEY = "AIzaSyCaP65ZQSalRwjqUDFaJzo89JmYUcAyjmw"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
    input_str = ", ".join(f"{key}: {value}" for key, value in inputs.items())
    prompt = f"""आप एक विशेषज्ञ कृषि सलाहकार हैं। निम्नलिखित मिट्टी विश्लेषण इनपुट और अनुमानित उर्वरता स्तर पर विचार करते हुए, उपयोगकर्ता को विस्तृत और विशिष्ट सिफारिशें प्रदान करें। अपनी प्रतिक्रिया सरल हिंदी में दें।

    इनपुट पैरामीटर: {input_str}
    अनुमानित उर्वरता स्तर: {fertility_result}

    सिफारिशें:"""
    result = model.invoke([HumanMessage(content=prompt)])
    return result.content

with tab2:
    st.header("🌱 मिट्टी से डेटा तक का सफर, अब होगा आसान हर डगर! ➡️ मिट्टी ➡️ 📈")
    with st.form("soil_form"):
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N)", 0.0)
            P = st.number_input("Phosphorus (P)", 0.0)
            K = st.number_input("Potassium (K)", 0.0)
            pH = st.number_input("pH Level", 0.0)
            EC = st.number_input("Electrical Conductivity", 0.0)
        with col2:
            OC = st.number_input("Organic Carbon", 0.0)
            S = st.number_input("Sulphur (S)", 0.0)
            Zn = st.number_input("Zinc (Zn)", 0.0)
            Fe = st.number_input("Iron (Fe)", 0.0)
            Cu = st.number_input("Copper (Cu)", 0.0)
        Mn = st.number_input("Manganese (Mn)", 0.0)
        B = st.number_input("Boron (B)", 0.0)
        submitted = st.form_submit_button("मिट्टी का विश्लेषण करें")

    if submitted:
        features = [N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]
        input_data = {
            "Nitrogen (N)": N,
            "Phosphorus (P)": P,
            "Potassium (K)": K,
            "pH Level": pH,
            "Electrical Conductivity": EC,
            "Organic Carbon": OC,
            "Sulphur (S)": S,
            "Zinc (Zn)": Zn,
            "Iron (Fe)": Fe,
            "Copper (Cu)": Cu,
            "Manganese (Mn)": Mn,
            "Boron (B)": B,
        }
        fertility = predict_soil_fertility_np(features)

        st.subheader("📊 मिट्टी विश्लेषण परिणाम:")
        st.write("आपने निम्नलिखित इनपुट प्रदान किए:")
        for key, value in input_data.items():
            st.write(f"- {key}: {value}")
        st.success(f"🌾 मिट्टी की उर्वरता: {fertility}")

        try:
            gemini_recommendation = get_soil_recommendations_with_gemini(input_data, fertility)
            st.subheader("🌱 सिफारिशें:")
            st.write(gemini_recommendation)
        except Exception as e:
            st.error("Gemini AI से सिफारिशें प्राप्त करने में त्रुटि हुई।")
            st.exception(e)
# ----------- TAB 3: SMART REVIEW ANALYSIS -------------
with tab3:
    st.header("🧠 फसल की हर कहानी, अब इस बॉट की जुबानी! 🌾🗣️🤖")
    review_input = st.text_area("अपनी ग्राहक समीक्षा नीचे पेस्ट करें:")
    if st.button("🧠 समीक्षा का विश्लेषण करें"):
        if review_input:
            try:
                hindi_result = analyze_review_with_gemini_hindi(review_input)
                st.subheader("📝 समीक्षा विश्लेषण (हिंदी में):")
                st.write(hindi_result)
            except Exception as e:
                st.error("समीक्षा का विश्लेषण करने में त्रुटि हुई।")
                st.exception(e)
        else:
            st.warning("कृपया समीक्षा टेक्स्ट दर्ज करें।")
