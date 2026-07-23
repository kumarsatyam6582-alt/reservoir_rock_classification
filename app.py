import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Rock Classification", page_icon="🪨", layout="centered")

# --- Custom styling ---
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
    .warning-box {
        background-color: #fef9e7;
        border-left: 5px solid #f39c12;
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

# --- Header ---
st.markdown('<p class="title-text">🪨 Rock Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload a rock image and let the model identify its type</p>', unsafe_allow_html=True)

# --- About section ---
with st.expander("ℹ️ About the rock classes"):
    st.markdown("""
    - **Sandstone** — A sedimentary rock made mostly of sand-sized mineral grains, usually quartz. Often light tan, yellow, or gray with a granular, gritty texture.
    - **Limestone** — A sedimentary rock composed mainly of calcium carbonate (calcite). Typically light gray to white, sometimes with fine grain or fossil texture.
    - **Shale** — A fine-grained sedimentary rock formed from compacted mud/clay. Often shows thin, flat layering (fissility) and a smoother, dull surface.
    - **Garbage** — Not a rock type; this class catches non-rock or unusable images (blurry photos, unrelated objects, etc.).

    **Note:** This model performs best on images similar to its training data (clear, close-up rock surface photos). Confidence may be lower on photos taken in very different lighting or resolution conditions.
    """)

# --- Load model ---
@st.cache_resource
def load_my_model():
    return load_model("rock_classification.h5")

with st.spinner("Loading model..."):
    model = load_my_model()

class_names = ["garbage", "limestone", "sandstone", "shale"]
CONFIDENCE_THRESHOLD = 65  # percent


def auto_center_crop(pil_img, crop_fraction=0.6):
    """Crop toward the center of the image to focus on rock texture and
    reduce the influence of background/context -- simulates a closer,
    more 'texture-crop' style photo similar to the training data.
    """
    w, h = pil_img.size

    # First crop to a centered square using the shorter side
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    square = pil_img.crop((left, top, left + side, top + side))

    # Then zoom further into the center by crop_fraction
    zoom_side = int(side * crop_fraction)
    sq_w, sq_h = square.size
    left2 = (sq_w - zoom_side) // 2
    top2 = (sq_h - zoom_side) // 2
    zoomed = square.crop((left2, top2, left2 + zoom_side, top2 + zoom_side))

    return zoomed


def run_prediction(image_source, apply_auto_crop=True):
    """Run the model on an uploaded file or file path and return
    (predicted_class, confidence, full_prediction_array, display_image).

    Resizing uses 'nearest' interpolation to match the training pipeline
    (ImageDataGenerator also uses 'nearest' by default) -- a different
    interpolation method here can shift borderline predictions.
    """
    pil_img = Image.open(image_source).convert("RGB")

    if apply_auto_crop:
        pil_img = auto_center_crop(pil_img)

    img = pil_img.resize((224, 224), resample=Image.NEAREST)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence, prediction, pil_img


def show_result(pil_image, predicted_class, confidence, prediction):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(pil_image, caption="Image analyzed by the model", use_container_width=True)

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

        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown("""
                <div class="warning-box">
                    ⚠️ <b>Low confidence prediction.</b><br>
                    This image may differ from the conditions the model was trained on
                    (lighting, resolution, or rock surface texture). Treat this result
                    as a best guess rather than a certain match.
                </div>
            """, unsafe_allow_html=True)
        elif predicted_class == "garbage" and confidence >= 90:
            st.markdown("""
                <div class="warning-box">
                    ⚠️ <b>Possible out-of-scope image.</b><br>
                    A confident "garbage" result can also happen when a real rock photo looks
                    very different from the model's training images (e.g. a full specimen with
                    a background, rather than a close-up surface texture). If you're confident
                    this is a real rock, try a closer, plain-background crop of just the surface.
                </div>
            """, unsafe_allow_html=True)

        with st.expander("See all class probabilities"):
            for cls, prob in zip(class_names, prediction[0]):
                st.write(f"{cls}: {prob * 100:.2f}%")


# --- Example images (optional demo) ---
# To enable this, create an "examples" folder in your repo with images named:
# examples/sandstone.jpg, examples/limestone.jpg, examples/shale.jpg, examples/garbage.jpg
EXAMPLES_DIR = "examples"
example_files = {}
if os.path.isdir(EXAMPLES_DIR):
    for cls in class_names:
        for ext in ("jpg", "jpeg", "png"):
            candidate = os.path.join(EXAMPLES_DIR, f"{cls}.{ext}")
            if os.path.exists(candidate):
                example_files[cls] = candidate
                break

selected_example = None
if example_files:
    st.write("**Or try an example image:**")
    ex_cols = st.columns(len(example_files))
    for i, (cls, path) in enumerate(example_files.items()):
        with ex_cols[i]:
            st.image(path, caption=cls, use_container_width=True)
            if st.button(f"Try {cls}", key=f"example_{cls}"):
                selected_example = path

# --- Upload section ---
st.markdown("""
    <div style="background-color:#eaf2fb; border-left:5px solid #3498db; padding:0.9rem 1.2rem; border-radius:8px; margin-bottom:1rem; font-size:0.92rem; color:#2c3e50;">
        📸 <b>Tip:</b> for best results, upload a well-lit photo focused on the rock's surface
        texture. If you upload a full specimen photo with a background, this app can automatically
        zoom into the center of the image to approximate a closer texture crop.
    </div>
""", unsafe_allow_html=True)

auto_crop = st.checkbox("🔍 Auto-crop to focus on rock texture (recommended for full specimen photos)", value=True)

uploaded_file = st.file_uploader("📤 Choose a rock image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Analyzing image..."):
        predicted_class, confidence, prediction, display_image = run_prediction(uploaded_file, apply_auto_crop=auto_crop)
    show_result(display_image, predicted_class, confidence, prediction)

elif selected_example is not None:
    with st.spinner("Analyzing image..."):
        predicted_class, confidence, prediction, display_image = run_prediction(selected_example, apply_auto_crop=auto_crop)
    show_result(display_image, predicted_class, confidence, prediction)

else:
    st.info("👆 Upload an image above to get started")

# --- Footer ---
st.markdown('<p class="footer-text">Made with ❤️ by Satyam Kumar</p>', unsafe_allow_html=True)
