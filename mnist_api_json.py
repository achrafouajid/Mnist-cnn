#!/usr/bin/env python3
"""
mnist_api_kfold_tfjs.py

Flask API that serves a CNN model trained on MNIST using K-Fold Cross Validation.
Exports model architecture and weights as TF.js compatible JSON files.

Run:
    python mnist_api_kfold_tfjs.py
"""

import os
import base64
import io
import json
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from flask import Flask, request, jsonify
from PIL import Image
from sklearn.model_selection import KFold
import ssl, certifi

# --- SSL fix for macOS / certifi issues ---
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# --- Global constants ---
MODEL_PATH = "mnist_emnist_cnn_model.h5"
JSON_MODEL_PATH = "model_export"  # Directory for JSON export
MODEL = None
K_FOLDS = 5
EPOCHS = 10
BATCH_SIZE = 128
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10


# -----------------------------
# Build Keras v2 compatible model
# -----------------------------
def build_model() -> tf.keras.Model:
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten

    inputs = Input(batch_shape=(None, 28, 28, 1), name="input_1", dtype="float32")

    x = Conv2D(32, (3, 3), padding="valid", activation="linear", name="conv0")(inputs)
    x = BatchNormalization(name="bn0", axis=3, momentum=0.99, epsilon=0.001)(x)
    x = Activation("relu", name="activation_1")(x)

    x = Conv2D(32, (3, 3), padding="valid", activation="linear", name="conv1")(x)
    x = BatchNormalization(name="bn1", axis=3, momentum=0.99, epsilon=0.001)(x)
    x = Activation("relu", name="activation_2")(x)
    x = MaxPooling2D((2, 2), name="MP1")(x)

    x = Conv2D(64, (3, 3), padding="valid", activation="linear", name="conv2")(x)
    x = BatchNormalization(name="bn2", axis=3, momentum=0.99, epsilon=0.001)(x)
    x = Activation("relu", name="activation_3")(x)

    x = Conv2D(64, (3, 3), padding="valid", activation="linear", name="conv3")(x)
    x = BatchNormalization(name="bn3", axis=3, momentum=0.99, epsilon=0.001)(x)
    x = Activation("relu", name="activation_4")(x)
    x = MaxPooling2D((2, 2), name="MP2")(x)
    x = Dropout(0.2, name="dropout_1")(x)

    x = Flatten(name="flatten_1")(x)
    x = Dense(256, activation="relu", name="fc1")(x)
    x = Dropout(0.4, name="dropout_2")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", name="fco")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MNIST_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# -----------------------------
# Export model to TF.js JSON
# -----------------------------
def export_model_to_json(model: tf.keras.Model, export_dir: str = JSON_MODEL_PATH) -> None:
    os.makedirs(export_dir, exist_ok=True)

    # Export Keras v2 JSON
    keras_json_path = os.path.join(export_dir, "model_keras.json")
    with open(keras_json_path, "w") as f:
        f.write(model.to_json())
    print(f"✅ Keras JSON saved to {keras_json_path}")

    # Export weights as JSON (optionally for direct parsing)
    weights_data = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            layer_data = {"layer_name": layer.name, "weights": []}
            for idx, w in enumerate(layer_weights):
                layer_data["weights"].append({
                    "index": idx,
                    "shape": w.shape,
                    "data": w.flatten().tolist()
                })
            weights_data.append(layer_data)
    weights_path = os.path.join(export_dir, "weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights_data, f)
    print(f"✅ Weights saved to {weights_path}")


# -----------------------------
# Train K-Fold with augmentation
# -----------------------------
def train_and_save_model(model_path=MODEL_PATH, json_export_dir=JSON_MODEL_PATH, epochs=EPOCHS, k_folds=K_FOLDS):
    print("📦 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    X = np.concatenate((x_train, x_test), axis=0).astype("float32") / 255.0
    X = np.expand_dims(X, -1)  # (n,28,28,1)
    y = np.concatenate((y_train, y_test), axis=0)
    y = to_categorical(y, NUM_CLASSES)  # One-hot for categorical_crossentropy

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_no = 1

    for train_idx, val_idx in kf.split(X):
        print(f"\n🔹 Training Fold {fold_no}/{k_folds}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = build_model()

        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                     height_shift_range=0.1, zoom_range=0.1)
        datagen.fit(X_train)

        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-5)

        model.fit(datagen.flow(X_train, y_train_fold, batch_size=BATCH_SIZE),
                  validation_data=(X_val, y_val_fold),
                  epochs=epochs, verbose=1, callbacks=[reduce_lr])

        scores = model.evaluate(X_val, y_val_fold, verbose=0)
        print(f"✅ Fold {fold_no} - Loss: {scores[0]:.4f} - Accuracy: {scores[1]*100:.2f}%")
        fold_no += 1

    # Retrain final model on all data
    print("\n🔁 Retraining final model on all data...")
    final_model = build_model()
    datagen.fit(X)
    final_model.fit(datagen.flow(X, y, batch_size=BATCH_SIZE), epochs=epochs, verbose=1)

    final_model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    export_model_to_json(final_model, json_export_dir)


# -----------------------------
# Preprocess input image
# -----------------------------
def preprocess_image(base64_string: str) -> np.ndarray:
    if base64_string.startswith("data:"):
        _, base64_string = base64_string.split(",", 1)
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.ANTIALIAS

    if img.size != (28, 28):
        img = img.resize((28, 28), resample)

    arr = np.array(img).astype("float32") / 255.0
    if np.mean(arr) > 0.5:  # invert if background dark
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
    data = request.get_json()
    if "image_data" not in data:
        return jsonify({"error": "'image_data' required"}), 400

    img_arr = preprocess_image(data["image_data"])
    preds = MODEL.predict(img_arr)
    pred_digit = int(np.argmax(preds[0]))
    confidence = float(np.round(float(preds[0][pred_digit]), 4))
    return jsonify({"predicted_digit": pred_digit, "confidence": confidence})


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "mnist_api running",
                    "model_loaded": MODEL is not None,
                    "json_export_available": os.path.exists(JSON_MODEL_PATH)})


# -----------------------------
# Load or train model
# -----------------------------
def load_model_or_train():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found. Training...")
        train_and_save_model()
    else:
        print("✅ Loading existing model...")
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        if not os.path.exists(JSON_MODEL_PATH):
            export_model_to_json(MODEL)
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model ready.")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    load_model_or_train()
    print("🚀 Starting Flask server on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001)
