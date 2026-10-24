
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from audio_utils import extract_mfcc

# -------------------------------------------------
# 1. Project settings
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR / "dataset" / "Audio_Speech_Actors_01-24"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "emotion_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

# RAVDESS emotion codes
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

EMOTIONS = list(EMOTION_MAP.values())

# Keep the order consistent throughout the project.
LABEL_TO_ID = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}

# Reproducible results
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------
# 2. Read files and extract MFCC features
# -------------------------------------------------

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}\n"
        "Check your dataset folder structure."
    )

features = []
labels = []
actor_ids = []
skipped_files = 0

print("Loading RAVDESS speech dataset...")
print("This step may take several minutes.")

for audio_path in DATASET_DIR.rglob("*.wav"):

    # Filename format:
    # modality-vocal_channel-emotion-intensity-
    # statement-repetition-actor.wav

    parts = audio_path.stem.split("-")

    if len(parts) != 7:
        skipped_files += 1
        continue

    modality, vocal_channel, emotion_code = parts[:3]
    actor_code = parts[6]

    # Only audio-only speech recordings
    if modality != "03" or vocal_channel != "01":
        continue

    if emotion_code not in EMOTION_MAP:
        skipped_files += 1
        continue

    try:
        mfcc = extract_mfcc(str(audio_path))

        features.append(mfcc)
        labels.append(LABEL_TO_ID[EMOTION_MAP[emotion_code]])
        actor_ids.append(int(actor_code))

    except Exception as error:
        skipped_files += 1
        print(f"Skipped {audio_path.name}: {error}")

if len(features) == 0:
    raise ValueError(
        "No audio files were processed. Check the dataset path."
    )

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels, dtype=np.int32)
groups = np.asarray(actor_ids)

print("\nDataset loaded successfully.")
print("Feature array shape:", X.shape)
print("Label array shape:", y.shape)
print("Skipped files:", skipped_files)

print("\nEmotion counts:")
for emotion_id, count in sorted(Counter(y).items()):
    print(f"{EMOTIONS[emotion_id]}: {count}")

# -------------------------------------------------
# 3. Split by actor
# -------------------------------------------------

# Actors 01-20 are used for training.
# Actors 21-24 are held out for final testing.

train_mask = groups <= 20
test_mask = groups >= 21

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError(
        "Training or test data is empty. "
        "Verify actor IDs and folder contents."
    )

if len(np.unique(y_train)) != len(EMOTIONS):
    raise ValueError(
        "Not all emotion classes are present in training data."
    )

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))
print("Training actors:", sorted(np.unique(groups[train_mask])))
print("Test actors:", sorted(np.unique(groups[test_mask])))

# Add the channel dimension for a 2D CNN.
# Shape becomes (samples, MFCC coefficients, time, channels).

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# -------------------------------------------------
# 4. Calculate class weights
# -------------------------------------------------

# Helps account for class imbalance if present.

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

# -------------------------------------------------
# 5. Build the CNN model
# -------------------------------------------------

input_shape = X_train.shape[1:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),

    tf.keras.layers.Conv2D(
        16, kernel_size=(3, 3), padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3), padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(
        64, kernel_size=(3, 3), padding="same",
        activation="relu"
    ),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.35),

    # Eight emotion probabilities
    tf.keras.layers.Dense(
        len(EMOTIONS),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------
# 6. Train the CNN
# -------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
]

print("\nTraining started...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    epochs=40,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Load the best checkpoint selected by validation loss.
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------------------------
# 7. Evaluate on unseen actors
# -------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

probabilities = model.predict(X_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nClassification report:")
report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(EMOTIONS))),
    target_names=EMOTIONS,
    zero_division=0
)

print(report)

accuracy = accuracy_score(y_test, y_pred)

# Save the report for your assignment.
with open(MODEL_DIR / "classification_report.txt", "w") as file:
    file.write("Speech Emotion Recognition - Test Results\n\n")
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test accuracy: {accuracy:.4f}\n\n")
    file.write(report)

# -------------------------------------------------
# 8. Save emotion labels
# -------------------------------------------------

with open(LABELS_PATH, "w") as file:
    json.dump(EMOTIONS, file, indent=4)

# -------------------------------------------------
# 9. Plot confusion matrix
# -------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(EMOTIONS)))
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=EMOTIONS,
    yticklabels=EMOTIONS
)

plt.title("Speech Emotion Recognition - Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix.png",
    dpi=150
)
plt.close()

# -------------------------------------------------
# 10. Plot training and validation curves
# -------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "accuracy_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "loss_curve.png", dpi=150)
plt.close()

print("\nTraining and evaluation completed.")
print("Saved model:", MODEL_PATH)
print("Saved labels:", LABELS_PATH)
print("Saved report:", MODEL_DIR / "classification_report.txt")
print("Saved confusion matrix and learning curves.")
