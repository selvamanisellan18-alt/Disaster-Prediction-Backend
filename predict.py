import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# --- CONFIGURATION ---
MODEL_PATH = "model/disaster_model.h5"
IMG_SIZE = 224

# --- LOAD AI ENGINE ---
print("Loading AI Engine...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ AI Engine Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

CLASSES = ["Cyclone", "Fire", "Flood", "Normal"]

# --- SMART VALIDATION LOGIC ---
def is_invalid_image(image_path):
    img_cv = cv2.imread(image_path)
    if img_cv is None: 
        return True, "Unreadable Image Data"

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.shape[0] * gray.shape[1]

    # GUARD 1: Human Face Detection (Tuned to avoid false positives on clouds)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    # minSize increased so it only catches real, large faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        return True, "Human Face Detected! Strictly Non-Satellite."

    # GUARD 2: Document / Diagram Guard (Adjusted for Satellite Clouds)
    # Documents have PURE white (> 250). We changed threshold from 30% to 65% to allow heavily cloudy satellite images.
    pure_white_pixels = np.sum(gray > 250)
    white_ratio = pure_white_pixels / total_pixels

    if white_ratio > 0.65: 
        return True, "Document/Diagram Detected (Excessive pure white background)."

    # GUARD 3: Low Texture Block (Filters out totally blank or simple graphics)
    std_dev = np.std(gray)
    if std_dev < 10: # Lowered from 15 to 10 to allow smooth sea/ocean satellite images
        return True, "Low Texture Detected (Looks like a blank graphic)."

    return False, "Valid Satellite Terrain"

# --- MAIN PREDICTION ENGINE ---
def predict_disaster(image_path):
    # 1. Run Smart Guard Analysis
    is_invalid, reason = is_invalid_image(image_path)
    if is_invalid:
        print(f"❌ Security Block Triggered: {reason}")
        return "INVALID INPUT", 99.99

    # 2. Preprocess the Image for AI
    try:
        img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"❌ Image Processing Error: {e}")
        return "INVALID INPUT", 99.99

    # 3. Predict using MobileNetV2
    prediction = model.predict(img_array)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    
    result = CLASSES[class_index]
    
    # GUARD 4: AI Confusion Block (Tuned down to 60%)
    # Sometimes valid satellite images have slightly lower confidence. 
    # 60% is a safe sweet spot to allow AI to decide without rejecting too fast.
    if confidence < 0.60:
        print(f"❌ Security Block: Unrecognized Object (Low AI Confidence: {confidence:.2f})")
        return "INVALID INPUT", 99.99

    return result, confidence