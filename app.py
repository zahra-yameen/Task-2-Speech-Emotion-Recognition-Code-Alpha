
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from audio_utils import extract_mfcc

# -------------------------------------------------
# 1. Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "emotion_model.keras"
LABELS_PATH = BASE_DIR / "models" / "labels.json"

# -------------------------------------------------
# 2. Page settings
# -------------------------------------------------

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Upload a speech recording to estimate its emotion "
    "using MFCC features and a trained CNN."
)

st.info(
    "This experimental model predicts patterns associated "
    "with acted emotion labels. It cannot reliably determine "
    "a person's actual feelings or mental state."
)

# -------------------------------------------------
# 3. Load model and labels once
# -------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            "labels.json not found. Run train_model.py first."
        )

    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r") as file:
        emotion_labels = json.load(file)

    return loaded_model, emotion_labels


try:
    model, emotions = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()

st.success("Trained model loaded successfully.")

# -------------------------------------------------
# 4. Upload audio
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a speech audio file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file, format="audio/wav")

    if st.button("Analyze Emotion", type="primary"):

        try:
            # Save the uploaded file temporarily.
            # It is removed automatically after processing.
            with st.spinner("Analyzing audio..."):

                import tempfile
                import os

                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav",
                        delete=False
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    # Use exactly the same feature extractor
                    # used during model training.
                    mfcc = extract_mfcc(temp_path)

                    # Add batch and channel dimensions:
                    # (40, time_frames)
                    # -> (1, 40, time_frames, 1)
                    model_input = mfcc[np.newaxis, ..., np.newaxis]

                    probabilities = model.predict(
                        model_input,
                        verbose=0
                    )[0]

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            predicted_index = int(np.argmax(probabilities))
            predicted_emotion = emotions[predicted_index]
            model_score = float(probabilities[predicted_index])

            st.subheader("Prediction")
            st.markdown(
                f"### {predicted_emotion.capitalize()}"
            )

            st.metric(
                "Highest model score",
                f"{model_score * 100:.2f}%"
            )

            st.caption(
                "This percentage is the model's softmax score, "
                "not a calibrated probability that the person "
                "actually feels this emotion."
            )

            # Show scores for all classes.
            st.subheader("Scores for all emotion categories")

            score_data = {
                emotion.capitalize(): float(probability)
                for emotion, probability
                in zip(emotions, probabilities)
            }

            st.bar_chart(score_data)

            # Display the numerical scores as well.
            for emotion, probability in zip(
                emotions, probabilities
            ):
                st.write(
                    f"**{emotion.capitalize()}**: "
                    f"{probability * 100:.2f}%"
                )

        except Exception as error:
            st.error(
                f"Could not analyze this audio file: {error}"
            )

st.divider()

st.subheader("How this application works")
st.write(
    "1. Upload a WAV recording.\n"
    "2. Extract MFCC audio features.\n"
    "3. Pass the features to the trained CNN.\n"
    "4. Display the predicted class and model scores."
)

st.caption(
    "For research and educational use. Predictions may be "
    "inaccurate for different accents, languages, recording "
    "conditions, or naturally expressed emotions."
)
