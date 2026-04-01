import os
# This line fixes the Keras version conflict (very important for Streamlit Cloud)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Emotion Detector + Mood Booster", layout="centered")

st.title("😊 Face Emotion Detector with AI Mood Booster")
st.write("Upload photo or use webcam → Get emotion + smart suggestions")

# ====================== LOAD MODEL (Safe Loading) ======================
@st.cache_resource
def load_emotion_model():
    return load_model('emotion_model.h5', compile=False)

model = load_emotion_model()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ====================== MOOD BOOSTER SUGGESTIONS ======================
mood_tips = {
    'Sad': ["Listen to your favourite upbeat song 🎵", "Take a short walk outside 🌳", "Call a friend ❤️", "Watch funny videos 😂"],
    'Fear': ["Take 5 slow deep breaths 🧘", "Write 3 things you are grateful for ✨", "Drink warm tea ☕"],
    'Angry': ["Do 10 jumping jacks 💪", "Listen to calm music 🎧", "Write and tear the paper"],
    'Disgust': ["Watch cute animal videos 🐶", "Take a refreshing shower 🚿"],
    'Happy': ["You're already awesome! Spread the smile 😊"],
    'Surprise': ["Enjoy the surprise moment! 🎉"],
    'Neutral': ["You're calm and balanced. Try something new today 🚀"]
}

# ====================== 1. PHOTO UPLOAD ======================
st.subheader("1. Upload a Photo")
uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        st.warning("⚠️ No face detected. Try another photo with clear face and good lighting.")
    else:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
            
            prediction = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(prediction)
            emotion = emotion_labels[emotion_idx]
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"✅ Detected Emotion: **{emotion}**")
        
        if emotion in ['Sad', 'Fear', 'Angry', 'Disgust']:
            st.subheader("💡 AI Suggestions to Boost Your Mood")
            for tip in mood_tips[emotion]:
                st.write(f"• {tip}")
        else:
            st.success(f"You're feeling **{emotion}**! Keep it up 😊")

# ====================== 2. LIVE WEBCAM (Snapshot) ======================
st.subheader("2. Live Webcam")
st.write("Click below to take a photo from your webcam")

camera_image = st.camera_input("📸 Capture from webcam")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        st.warning("⚠️ No face detected. Try again with better lighting.")
    else:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
            
            prediction = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(prediction)
            emotion = emotion_labels[emotion_idx]
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"✅ Detected Emotion: **{emotion}**")
        
        if emotion in ['Sad', 'Fear', 'Angry', 'Disgust']:
            st.subheader("💡 AI Suggestions to Boost Your Mood")
            for tip in mood_tips[emotion]:
                st.write(f"• {tip}")
        else:
            st.success(f"You're feeling **{emotion}**! Keep it up 😊")
