#!/usr/bin/env python3
"""
test_client.py

Simple client script to test mnist_api.py Flask server.

Usage:
    1. Start the API server:
        python mnist_api.py

    2. Run this client with a test image (28x28 grayscale PNG or JPEG):
        python test_client.py digit.png
"""

import sys
import base64
import json
import requests

API_URL = "http://127.0.0.1:5001/predict_digit"

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_client.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img_b64 = encode_image_to_base64(image_path)

    payload = {"image_data": img_b64}
    headers = {"Content-Type": "application/json"}

    print(f"Sending {image_path} to API...")
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        print("Prediction result:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    main()
