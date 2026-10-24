
import numpy as np
import librosa

# Settings shared by training and prediction
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
MAX_SAMPLES = SAMPLE_RATE * DURATION


def extract_mfcc(audio_path):
    """
    Load an audio file and convert it into a fixed-size
    MFCC feature matrix for the CNN model.
    """

    # Load audio as mono and resample to 22050 Hz
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # Pad short recordings or trim longer recordings
    if len(audio) < MAX_SAMPLES:
        audio = np.pad(
            audio,
            (0, MAX_SAMPLES - len(audio))
        )
    else:
        audio = audio[:MAX_SAMPLES]

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    # Normalize each coefficient across time
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    mfcc = (mfcc - mean) / (std + 1e-8)

    return mfcc.astype(np.float32)
