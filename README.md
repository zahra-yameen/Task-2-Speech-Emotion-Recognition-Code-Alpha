# Task-2-Speech-Emotion-Recognition-Code-Alpha
Speech Emotion Recognition using Python, MFCC audio feature extraction, and a CNN deep learning model. Classifies speech into eight emotion categories and provides an interactive Streamlit web app for audio predictions.

# Speech Emotion Recognition Using CNN and MFCC

A deep learning project that analyzes speech audio and predicts the emotion expressed in the voice. It uses **MFCC audio features** and a **Convolutional Neural Network (CNN)**, with an interactive web interface built using Streamlit.

## Project Overview

Speech Emotion Recognition (SER) is a machine learning task that identifies emotional categories from speech recordings. This project extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio and uses a trained CNN to classify the recording into one of eight emotion categories.

## Objectives

- Extract meaningful audio features using MFCCs.
- Train a CNN model to classify speech emotions.
- Evaluate the model using classification metrics.
- Build a Streamlit application for audio-based predictions.

## Emotion Categories

The model is designed to recognize these eight RAVDESS emotion classes:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

Predictions depend on the dataset, model training, and audio quality. The model does not determine a person’s true internal feelings.

## Dataset

This project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**, specifically its speech recordings.

RAVDESS contains acted emotional performances by actors. The speech dataset is used to train and evaluate the model.

Dataset information: https://zenodo.org/record/1188976

The dataset is not included in this GitHub repository. Download it separately if you want to retrain the model, and follow the dataset’s terms of use.

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn (for evaluation visualizations)

## Project Structure

```text
Speech-Emotion-Recognition/
├── app.py
├── audio_utils.py
├── train_model.py
├── requirements.txt
├── labels.json
├── README.md
└── models/
    └── emotion_model.keras
```

## How It Works

1. The user uploads a speech audio file through the Streamlit app.
2. The audio is converted to the required format and duration.
3. MFCC features are extracted from the audio.
4. The trained CNN processes the features.
5. The app displays the predicted emotion and model scores.

## Installation and Local Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app should open in your browser. Upload a supported audio file to receive a prediction.

## Model Training

To train the model yourself:

1. Download the RAVDESS speech dataset.
2. Place the dataset in the location expected by `train_model.py`.
3. Install the required dependencies.
4. Run:

```bash
python train_model.py
```

The training script should save the trained model and label mapping. Confirm the output paths in `train_model.py` match the paths used by `app.py`.

## Model Evaluation

The model should be evaluated on recordings that were not used during training. This project can report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

**Results:** Add your actual test results here after training. Do not report example or expected values as measured results.

| Metric | Test result |
|---|---|
| Accuracy | Add your result |
| Precision | Add your result |
| Recall | Add your result |
| F1-score | Add your result |

## Live Demo

[Open the Streamlit application](-----)

Replace the placeholder with your deployed app URL. If you have not deployed it yet, remove this section until the app is live.

## Limitations

- RAVDESS contains acted emotional speech, which may differ from natural everyday speech.
- Performance can vary with speakers, accents, recording conditions, and audio quality.
- A predicted class or model score is not proof of a speaker’s actual emotional state.
- The model is intended for learning and demonstration, not for high-stakes decisions.

## Future Improvements

- Test on additional speech-emotion datasets.
- Improve generalization to different speakers and recording conditions.
- Experiment with LSTM or other audio deep learning architectures.
- Add more audio formats and clearer prediction explanations.

## Author

**YOUR NAME**

GitHub: https://github.com/ZAHRA-YAMEEN

## License

Add a license if you intend to permit others to use, modify, or distribute your code. Check the dataset’s separate terms before redistributing any dataset content.
