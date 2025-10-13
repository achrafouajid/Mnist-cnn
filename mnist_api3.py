#!/usr/bin/env python3
"""
mnist_api_kfold.py

Flask API that serves a CNN model trained on MNIST using K-Fold Cross Validation
with data augmentation, batch normalization, and adaptive learning rate.

If mnist_cnn_model.h5 doesn't exist, the script trains the CNN with 5-fold CV,
then retrains on the full dataset and saves the final model.

Run:
    python mnist_api_kfold.py
"""

import os
import base64
import io
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from flask import Flask, request, jsonify
from PIL import Image
from sklearn.model_selection import KFold
import ssl, certifi

# --- SSL fix for macOS / certifi issues ---
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# --- Global constants ---
MODEL_PATH = "mnist_emnist_cnn_model.h5"
MODEL = None
K_FOLDS = 5


# -----------------------------
# K-Fold Training with Augmentation
# -----------------------------
def build_model() -> tf.keras.Model:
    """Build an improved CNN with BatchNorm and Dropout."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_save_model(model_path: str = MODEL_PATH, epochs: int = 10, k_folds: int = K_FOLDS) -> None:
    """Train the CNN using K-Fold CV with data augmentation and save the final model."""
    print("📦 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Combine both sets
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Normalize and reshape
    X = X.astype("float32") / 255.0
    X = np.expand_dims(X, -1)  # (n,28,28,1)

    # Prepare KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    acc_per_fold, loss_per_fold = [], []

    fold_no = 1
    for train_idx, val_idx in kf.split(X):
        print(f"\n🔹 Training Fold {fold_no}/{k_folds}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model()

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        datagen.fit(X_train)

        # Learning rate scheduler
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-5)

        # Train
        model.fit(
            datagen.flow(X_train, y_train, batch_size=128),
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=1,
            callbacks=[reduce_lr]
        )

        scores = model.evaluate(X_val, y_val, verbose=0)
        print(f"✅ Fold {fold_no} - Loss: {scores[0]:.4f} - Accuracy: {scores[1]*100:.2f}%")

        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no += 1

    print("\n📊 K-Fold Summary:")
    print(f"Average Accuracy: {np.mean(acc_per_fold):.2f}% (+/- {np.std(acc_per_fold):.2f})")
    print(f"Average Loss: {np.mean(loss_per_fold):.4f}")

    # Retrain final model on all data
    print("\n🔁 Retraining final model on all data...")
    final_model = build_model()
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(X)
    final_model.fit(datagen.flow(X, y, batch_size=128), epochs=epochs, verbose=1)
    final_model.save(model_path)
    print(f"✅ Final model saved to {model_path}")


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(base64_string: str) -> np.ndarray:
    """Decode base64 image → grayscale → resize → normalize."""
    if not isinstance(base64_string, str):
        raise ValueError("image_data must be a base64-encoded string")

    if base64_string.startswith("data:"):
        try:
            _, base64_string = base64_string.split(",", 1)
        except ValueError:
            raise ValueError("Invalid data URL format for image_data")

    try:
        image_bytes = base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Base64 decoding failed: {e}")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as e:
        raise ValueError(f"Unable to open image from decoded bytes: {e}")

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.ANTIALIAS

    if img.size != (28, 28):
        img = img.resize((28, 28), resample)

    arr = np.array(img).astype("float32") / 255.0

    # Optional inversion if background is dark
    if np.mean(arr) > 0.5:
        arr = 1 - arr

    arr = np.reshape(arr, (1, 28, 28, 1))
    return arr


# -----------------------------
# Flask API
# -----------------------------
app = Flask(__name__)

@app.route("/predict_digit", methods=["POST"])
def predict_digit():
    global MODEL
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be application/json"}), 400
        data = request.get_json()
        if "image_data" not in data:
            return jsonify({"error": "JSON body must contain 'image_data' field"}), 400

        img_b64 = data["image_data"]
        img_arr = preprocess_image(img_b64)
        preds = MODEL.predict(img_arr)

        if preds.ndim == 2 and preds.shape[1] == 10:
            probs = preds[0]
        else:
            return jsonify({"error": f"Unexpected prediction shape: {preds.shape}"}), 500

        pred_digit = int(np.argmax(probs))
        confidence = float(np.round(float(probs[pred_digit]), 4))

        return jsonify({
            "predicted_digit": pred_digit,
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {e}"}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "mnist_api running", "model_loaded": MODEL is not None})


# -----------------------------
# Model Loader
# -----------------------------
def load_model_or_train(model_path: str = MODEL_PATH) -> None:
    global MODEL
    if not os.path.exists(model_path):
        print(f"⚠️ Model file '{model_path}' not found. Training a new model with {K_FOLDS}-Fold CV...")
        train_and_save_model(model_path=model_path, epochs=10, k_folds=K_FOLDS)
    else:
        print(f"✅ Found existing model at '{model_path}'. Skipping training.")

    print(f"Loading model from {model_path} ...")
    MODEL = tf.keras.models.load_model(model_path)
    print("✅ Model loaded and ready.")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF logs
    load_model_or_train(MODEL_PATH)
    print("🚀 Starting Flask server on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001)
