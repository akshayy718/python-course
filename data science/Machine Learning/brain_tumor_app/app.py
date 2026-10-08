import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ----------------------------------------------------------------------------
# Settings (these must match the training notebook)
# ----------------------------------------------------------------------------
IMG_SIZE = 150                       # images are resized to 150 x 150
CLASS_NAMES = ['No Tumor', 'Tumor']  # folder order in training: 'no' = 0, 'yes' = 1
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'brain_tumor_classifier.keras'
)
LOW_CONFIDENCE = 70.0                # below this %, we warn the user

# Thresholds for the "is this really an MRI?" check. They were chosen by measuring all
# 253 MRI images of the training dataset (the values in brackets are what the real MRIs
# looked like), so real scans pass while ordinary photos are rejected.
MAX_COLOURFULNESS = 20.0    # MRI scans are grayscale (real MRIs: 0 to 12). Photos are 40+.
MAX_DARK_FRACTION = 0.85    # almost completely black image (real MRIs: up to 0.77)
MIN_CONTRAST = 20.0         # almost blank / flat image (real MRIs: 30 and above)
BRIGHT_CORNER_LEVEL = 100   # bright corners = bright background, like a photo or document
DARK_BORDER_MIN = 0.30      # MRI scans usually have a dark border around the head


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model once and keep it in memory."""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None


def preprocess_image(img):
    """Prepare a PIL image exactly the way the training notebook did.

    1. Convert to RGB (MRI files can be grayscale, RGBA, etc.)
    2. Resize to 150 x 150 (same resize method as image_dataset_from_directory)
    3. Add a batch dimension -> shape (1, 150, 150, 3)

    No division by 255 here: the model has a Rescaling layer inside it.
    """
    img = img.convert('RGB')
    arr = np.array(img, dtype=np.float32)
    arr = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE)).numpy()
    return np.expand_dims(arr, axis=0)


def predict(model, img):
    """Return the probability (in %) of each class: [No Tumor, Tumor]."""
    logits = model.predict(preprocess_image(img), verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()
    return 100.0 * probs


def check_mri_like(img):
    """Quick sanity check: does this image look like a brain MRI scan?

    Returns (True, "") if it looks like an MRI, otherwise (False, reason).
    This only looks at simple image statistics (colour, brightness, background).
    It is NOT a medical check and it is not perfect: a grayscale photo with a dark
    background can still pass.
    """
    small = img.convert('RGB')
    small.thumbnail((200, 200))
    rgb = np.asarray(small).astype(np.float32)
    gray = rgb.mean(axis=2)

    # 1. Colour: MRI scans are grayscale, normal photos (selfies, objects) are colourful
    colourfulness = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    if colourfulness > MAX_COLOURFULNESS:
        return False, "It looks like a normal colour photo. MRI scans are grayscale images."

    # 2. Almost completely black, or almost blank / flat
    if float((gray < 30).mean()) > MAX_DARK_FRACTION:
        return False, "The image is almost completely black."
    if float(gray.std()) < MIN_CONTRAST:
        return False, "The image is almost blank or has too little detail."

    # 3. Bright background (photo, document, drawing) instead of the dark MRI background
    h, w = gray.shape
    b = max(2, int(0.08 * min(h, w)))
    ring = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(),
                           gray[:, :b].ravel(), gray[:, -b:].ravel()])
    dark_border = float((ring < 40).mean())
    corners = float(np.mean([gray[:b, :b].mean(), gray[:b, -b:].mean(),
                             gray[-b:, :b].mean(), gray[-b:, -b:].mean()]))
    if dark_border < DARK_BORDER_MIN and corners > BRIGHT_CORNER_LEVEL:
        return False, "It has a bright photo-like background instead of the dark background of an MRI scan."

    return True, ""


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction"])
    st.sidebar.warning(
        "For learning only. This app is **not** a medical device and must not "
        "be used to diagnose anyone."
    )

    model = load_model()

    if page == "Home":
        st.title("🧠 Brain Tumor Detection from MRI Images")
        st.markdown("""
        ### Welcome to the Brain Tumor Detection App!

        This application uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras
        to look at a brain MRI image and predict whether it shows a **tumor** or **no tumor**.

        #### 📌 Navigation Guide:
        - **Home**: Overview of the application.
        - **Prediction**: Upload a single MRI image and let the trained model predict the result.

        👈 Use the sidebar on the left to navigate through the app.

        #### ⚙️ How it works:
        1. You upload a brain MRI image (PNG, JPG or JPEG). Photos that are clearly not MRI scans are rejected.
        2. The image is converted to RGB and resized to 150 x 150 pixels.
        3. The CNN gives a probability for **No Tumor** and for **Tumor**.
        4. The class with the higher probability is shown as the prediction.

        #### 📊 About the model:
        - Trained on the *Brain MRI Images for Brain Tumor Detection* dataset (253 images:
          155 with a tumor, 98 without).
        - The dataset is very small, so the model makes mistakes. Accuracy on held-out images
          was roughly 70-85% depending on the split.
        """)
        st.warning(
            "⚠️ **Disclaimer:** This is a student learning project. Its predictions can be wrong "
            "and must **never** replace the opinion of a doctor or radiologist."
        )

    elif page == "Prediction":
        st.title("🔮 Prediction")
        st.write("Upload a brain MRI image to predict whether it shows a tumor or not.")

        if model is None:
            st.warning(
                "⚠️ **Model not found!** Please run the training notebook first, so that "
                "`brain_tumor_classifier.keras` is saved in the same folder as `app.py`."
            )
            return

        uploaded_file = st.file_uploader("Upload an MRI image...", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                img.load()  # forces Pillow to read the whole file now, so a broken file fails here
            except (UnidentifiedImageError, OSError):
                st.error("❌ This file could not be read as an image. Please upload a valid PNG or JPG.")
                return

            st.image(img, caption='Uploaded Image', width=300)

            # Reject images that are clearly not brain MRI scans
            looks_like_mri, reason = check_mri_like(img)
            if not looks_like_mri:
                st.error("🚫 **This is not a suitable image.** Please upload a brain MRI scan.")
                st.write(reason)
                st.info(
                    "💡 **Tips:** use a brain MRI slice saved as a JPG or PNG. "
                    "MRI scans are grayscale with a dark background. "
                    "Photos of people, objects, documents or screenshots will not work."
                )
                return

            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    probs = predict(model, img)

                predicted_idx = int(np.argmax(probs))
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence = float(probs[predicted_idx])

                if predicted_class == 'Tumor':
                    st.error(f"### The model predicts: **{predicted_class}**")
                else:
                    st.success(f"### The model predicts: **{predicted_class}**")
                st.info(f"**Confidence:** {confidence:.2f}%")

                if confidence < LOW_CONFIDENCE:
                    st.warning("The model is not very sure about this image. Treat this result with extra caution.")

                st.write("#### Probability of each class")
                for name, p in zip(CLASS_NAMES, probs):
                    st.write(f"{name}: {p:.2f}%")
                    st.progress(min(max(float(p) / 100.0, 0.0), 1.0))

                st.caption(
                    "⚠️ Learning project only. This is not a medical diagnosis. "
                    "Please consult a qualified doctor for any health concern."
                )


if __name__ == "__main__":
    main()
