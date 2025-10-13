#!/usr/bin/env python3
"""
convert_to_tfjs.py

Manually converts a trained Keras model (.h5) to TensorFlow.js format
without using tensorflowjs_converter.

Produces:
    tfjs_model/
      ├── model.json
      └── weights.bin

Compatible with:
    - Keras v2.1.2 JSON format
    - TensorFlow.js Converter v4.22.0
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "mnist_emnist_cnn_model.h5"
OUTPUT_DIR = "tfjs_model"


# -------------------------------------------------------------------------
# Export model weights as binary (.bin)
# -------------------------------------------------------------------------
def export_weights_as_binary(model, output_dir):
    """Export model weights as binary files (TensorFlow.js format)."""
    all_weights = []
    weight_specs = []

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for weight in weights:
                all_weights.append(weight)
                weight_specs.append({
                    "name": f"{layer.name}/{len(weight_specs)}",
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype)
                })

    os.makedirs(output_dir, exist_ok=True)
    weights_file = os.path.join(output_dir, "weights.bin")

    with open(weights_file, 'wb') as f:
        for weight in all_weights:
            f.write(weight.astype(np.float32).tobytes())

    return weight_specs, os.path.getsize(weights_file)


# -------------------------------------------------------------------------
# Create TFJS model.json (Keras 2.1.2 structure)
# -------------------------------------------------------------------------
def create_tfjs_model_json(model, weight_specs, weights_size, output_dir):
    """Create a TensorFlow.js compatible model.json matching Keras v2.1.2 structure."""
    keras_version = "2.1.2"
    tfjs_converter_version = "4.22.0"

    model_config = model.get_config()

    tfjs_model = {
        "format": "layers-model",
        "generatedBy": f"keras v{keras_version}",
        "convertedBy": f"TensorFlow.js Converter v{tfjs_converter_version}",
        "modelTopology": {
            "keras_version": keras_version,
            "backend": "tensorflow",
            "model_config": {
                "class_name": model.__class__.__name__,
                "config": model_config
            }
        },
        "weightsManifest": [
            {
                "paths": ["weights.bin"],
                "weights": weight_specs
            }
        ]
    }

    model_json_path = os.path.join(output_dir, "model.json")
    with open(model_json_path, "w") as f:
        json.dump(tfjs_model, f, indent=2)

    return model_json_path


# -------------------------------------------------------------------------
# Main conversion process
# -------------------------------------------------------------------------
def convert_model_to_tfjs(model_path, output_dir):
    """Main conversion function."""
    print(f"📦 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    print(f"🔄 Converting to TensorFlow.js format...")
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ Export weights
    weight_specs, weights_size = export_weights_as_binary(model, output_dir)
    print(f"   ✅ Weights exported: {weights_size / 1024:.2f} KB")

    # 2️⃣ Create model.json
    model_json_path = create_tfjs_model_json(model, weight_specs, weights_size, output_dir)
    print(f"   ✅ Model JSON created: {model_json_path}")

    print("\n🎉 Conversion complete!")
    print(f"📂 Output directory: {output_dir}/")
    print(f"   - model.json (model architecture + metadata)")
    print(f"   - weights.bin (model weights)")

    return output_dir


# -------------------------------------------------------------------------
# Script entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print(f"   Please ensure the .h5 model exists in the current directory.")
        exit(1)

    convert_model_to_tfjs(MODEL_PATH, OUTPUT_DIR)

    print("\n📱 To use in your Expo app:")
    print("   1. Copy the tfjs_model/ folder to your app's assets")
    print("   2. Load the model with:")
    print("""
      import * as tf from '@tensorflow/tfjs';
      import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

      const model = await tf.loadLayersModel(
        bundleResourceIO(
          require('./tfjs_model/model.json'),
          require('./tfjs_model/weights.bin')
        )
      );
    """)
