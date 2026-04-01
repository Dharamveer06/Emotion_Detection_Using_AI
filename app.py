import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Important for compatibility

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("😊 Face Emotion Detector + Mood Booster")

# Load model
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model('emotion_model.keras', compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_emotion_model()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

mood_tips = {
    'Sad': ["Listen to upbeat music 🎵", "Take a short walk 🌳", "Call a friend ❤️", "Watch funny videos 😂"],
    'Fear': ["Take 5 deep breaths 🧘", "Write 3 things you're grateful for ✨", "Drink warm tea ☕"],
    'Angry': ["Do jumping jacks 💪", "Listen to calm music 🎧"],
    'Disgust': ["Watch cute animal videos 🐶", "Take a shower 🚿"],
    'Happy': ["You're already awesome! 😊"],
    'Surprise': ["Enjoy the moment! 🎉"],
    'Neutral': ["You're calm. Try something new 🚀"]
}

# Photo Upload
st.subheader("1. Upload Photo")
uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = roi.reshape(1,48,48,1) / 255.0
            pred = model.predict(roi, verbose=0)
            emotion = emotion_labels[np.argmax(pred)]
            
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Detected: **{emotion}**")
        
        if emotion in mood_tips and emotion in ['Sad','Fear','Angry','Disgust']:
            st.subheader("💡 Mood Boost Suggestions")
            for tip in mood_tips[emotion]:
                st.write("• " + tip)

# Webcam
st.subheader("2. Webcam Snapshot")
cam_img = st.camera_input("Take photo")

if cam_img is not None:
    # Same processing as above (copy the if uploaded_file block and adapt for cam_img)
    bytes_data = cam_img.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    # ... (repeat the gray, face detection, prediction code here - same as photo part)
    # For brevity, you can duplicate the processing logic
