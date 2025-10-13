import os
import base64
import io
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

MODEL_PATH = "mnist_mobilenet.h5"
IMG_SIZE = 96  # MobileNetV2 expects >=96x96

app = Flask(__name__)
model = None

# -------------------------
# Model training & saving
# -------------------------
def train_and_save_model(model_path=MODEL_PATH, epochs=5):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # expand dims & normalize
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    # Resize to MobileNet input (96x96 RGB)
    x_train = tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE)).numpy()
    x_test = tf.image.resize(x_test, (IMG_SIZE, IMG_SIZE)).numpy()
    x_train = np.repeat(x_train, 3, axis=-1)
    x_test = np.repeat(x_test, 3, axis=-1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build MobileNetV2 backbone with pretrained ImageNet weights
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"   # ✅ Pretrained weights
    )
    base_model.trainable = False  # ✅ Freeze backbone first

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train only top layers
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=epochs,
              batch_size=128)

    # ✅ Optionally unfreeze last 50 layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=2,
              batch_size=128)

    model.save(model_path)
    print(f"✅ Model trained and saved at {model_path}")


# -------------------------
# Preprocessing function
# -------------------------
def preprocess_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("L")  # grayscale

        # Invert to black digit on white background
        image = ImageOps.invert(image)

        # Binarize
        image = image.point(lambda x: 0 if x < 128 else 255, "1").convert("L")

        # Smooth strokes
        image = image.filter(ImageFilter.SMOOTH)

        # Crop bounding box
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

        # Resize to 20x20 inside 28x28 canvas
        image = image.resize((20, 20), Image.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(image, ((28 - 20) // 2, (28 - 20) // 2))
        image = canvas

        # Convert to MobileNet input (96x96 RGB)
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.stack([img_array] * 3, axis=-1)  # grayscale → RGB
        img_array = np.expand_dims(img_array, 0)  # (1, 96, 96, 3)

        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


# -------------------------
# API Endpoint
# -------------------------
@app.route("/predict_digit", methods=["POST"])
def predict_digit():
    try:
        data = request.get_json()
        if "image_data" not in data:
            return jsonify({"error": "Missing 'image_data'"}), 400

        img_array = preprocess_image(data["image_data"])
        preds = model.predict(img_array)[0]

        top_indices = preds.argsort()[-2:][::-1]
        top_probs = preds[top_indices]

        if top_probs[0] < 0.7:  # ✅ Uncertainty handling
            return jsonify({
                "predicted_digit": None,
                "confidence": float(top_probs[0]),
                "message": "Uncertain, please retry",
                "top_candidates": [
                    {"digit": int(top_indices[i]), "confidence": float(top_probs[i])}
                    for i in range(len(top_indices))
                ]
            })

        return jsonify({
            "predicted_digit": int(top_indices[0]),
            "confidence": float(top_probs[0]),
            "top_candidates": [
                {"digit": int(top_indices[i]), "confidence": float(top_probs[i])}
                for i in range(len(top_indices))
            ]
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


# -------------------------
# Startup: load or train
# -------------------------
def load_model_or_train(model_path=MODEL_PATH):
    global model
    if not os.path.exists(model_path):
        print("⚠️ Model not found. Training...")
        train_and_save_model(model_path=model_path, epochs=5)
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded.")


if __name__ == "__main__":
    load_model_or_train()
    app.run(host="0.0.0.0", port=5001)
