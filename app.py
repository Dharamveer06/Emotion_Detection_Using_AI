import os
# MUST be at the VERY TOP before any tensorflow/keras import
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Emotion Detector + Mood Booster", layout="centered")

st.title("😊 Face Emotion Detector with AI Mood Booster")
st.write("Upload a photo or use webcam → Detect emotion + get mood suggestions")

# ====================== SAFE MODEL LOADING ======================
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model('emotion_model.h5', compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

model = load_emotion_model()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ====================== MOOD SUGGESTIONS ======================
mood_tips = {
    'Sad': ["Listen to upbeat music 🎵", "Take a short walk 🌳", "Call a friend ❤️", "Watch funny videos 😂"],
    'Fear': ["Take 5 deep breaths 🧘", "Write 3 things you're grateful for ✨", "Drink warm tea ☕"],
    'Angry': ["Do jumping jacks 💪", "Listen to calm music 🎧", "Write and rip the paper"],
    'Disgust': ["Watch cute animal videos 🐶", "Take a refreshing shower 🚿"],
    'Happy': ["You're already great! Spread the positivity 😊"],
    'Surprise': ["Enjoy the moment! Life is full of surprises 🎉"],
    'Neutral': ["You're balanced. Try something new today 🚀"]
}

# ====================== PHOTO UPLOAD ======================
st.subheader("1. Upload a Photo")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        st.warning("No face detected. Try a clearer photo with good lighting.")
    else:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
            
            prediction = model.predict(face_roi, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Detected: **{emotion}**")
        
        if emotion in ['Sad', 'Fear', 'Angry', 'Disgust']:
            st.subheader("💡 AI Suggestions to Boost Your Mood")
            for tip in mood_tips[emotion]:
                st.write(f"• {tip}")
        else:
            st.success(f"Great! You're feeling **{emotion}** 😊")

# ====================== WEBCAM SNAPSHOT ======================
st.subheader("2. Live Webcam")
camera_image = st.camera_input("📸 Take a photo from webcam")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        st.warning("No face detected. Try again.")
    else:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
            
            prediction = model.predict(face_roi, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Detected: **{emotion}**")
        
        if emotion in ['Sad', 'Fear', 'Angry', 'Disgust']:
            st.subheader("💡 AI Suggestions to Boost Your Mood")
            for tip in mood_tips[emotion]:
                st.write(f"• {tip}")
        else:
            st.success(f"You're feeling **{emotion}** 😊")
