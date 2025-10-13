#!/usr/bin/env python3
"""
mnistapi.py
Flask server for MNIST digit recognition
"""

import os
import io
import base64
import ssl
import certifi
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = "best_mnist_cnn.h5"
MODEL = None

# --- SSL fix for macOS / certifi issues ---
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

def preprocess_image(image_b64: str) -> np.ndarray:
    import cv2
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = np.array(img)

    # Invert if background is white
    if np.mean(img) > 127:
        img = 255 - img

    # Threshold to binary
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Resize digit to fit 20x20 box
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    padded = np.pad(digit, ((4,4), (4,4)), mode='constant', constant_values=0)

    # Normalize
    padded = padded.astype("float32") / 255.0
    padded = padded.reshape(1, 28, 28, 1)
    return padded

@app.route("/predict_digit", methods=["POST"])
def predict_digit():
    global MODEL
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
        if "image_data" not in data:
            return jsonify({"error": "Missing 'image_data' field"}), 400

        img_arr = preprocess_image(data["image_data"])
        preds = MODEL.predict(img_arr)
        probs = preds[0]
        pred_digit = int(np.argmax(probs))
        confidence = float(np.round(probs[pred_digit], 4))

        return jsonify({
            "predicted_digit": pred_digit,
            "confidence": confidence
        }), 200
    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "mnist_api running", "model_loaded": MODEL is not None})

def load_model_or_train():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first using train_mnist_model.py.")
    print(f"📦 Loading model from {MODEL_PATH} ...")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    load_model_or_train()
    print("🚀 Starting Flask server at http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001)
