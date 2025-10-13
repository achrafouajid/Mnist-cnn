import tensorflow as tf
import tensorflowjs as tfjs
import os

# Path to your .h5 Keras model
h5_model_path = "mnist_emnist_cnn_model.h5"

# Output folder for the converted TFJS model (same folder)
tfjs_output_dir = "."

# Make sure output directory exists
os.makedirs(tfjs_output_dir, exist_ok=True)

# Load Keras model
model = tf.keras.models.load_model(h5_model_path)

# Convert and save as TensorFlow.js format
tfjs.converters.save_keras_model(model, tfjs_output_dir)

print(f"Model converted successfully! Check the folder: {tfjs_output_dir}")
