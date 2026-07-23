import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.set_page_config(page_title="Rock Classification", page_icon="🪨", layout="centered")

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .title-text {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0;
    }
    .subtitle-text {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f7f4;
        border-left: 5px solid #27ae60;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .stButton>button {
        border-radius: 8px;
    }
    .footer-text {
        text-align: center;
        color: #95a5a6;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ecf0f1;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">🪨 Rock Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload a rock image and let the model identify its type</p>', unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    return load_model("rock_classification.h5")

with st.spinner("Loading model..."):
    model = load_my_model()

class_names = ["class1", "class2", "class3"]

uploaded_file = st.file_uploader("📤 Choose a rock image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    with col2:
        st.markdown(f"""
            <div class="result-box">
                <h4 style="margin-top:0;">🔍 Prediction</h4>
                <p style="font-size:1.4rem; font-weight:600; color:#27ae60;">{predicted_class}</p>
            </div>
        """, unsafe_allow_html=True)

        st.write("**Confidence**")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}%")

        with st.expander("See all class probabilities"):
            for cls, prob in zip(class_names, prediction[0]):
                st.write(f"{cls}: {prob*100:.2f}%")
else:
    st.info("👆 Upload an image above to get started")

st.markdown('<p class="footer-text">Made with ❤️ by Satyam Kumar</p>', unsafe_allow_html=True)
