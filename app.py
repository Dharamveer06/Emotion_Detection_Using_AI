import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model
model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Mood boosting suggestions (AI-style)
suggestions = {
    'Sad': ["Listen to your favorite upbeat song 🎵", "Take a short walk outside 🌳", "Call a friend and chat ❤️", "Watch funny videos 😂"],
    'Fear': ["Take 5 deep breaths slowly 🧘", "Write down 3 things you're grateful for ✨", "Drink a warm cup of tea ☕"],
    'Angry': ["Do 10 jumping jacks 💪", "Listen to calm music 🎧", "Write down why you're angry and then rip the paper"],
    'Disgust': ["Watch cute animal videos 🐶", "Take a refreshing shower 🚿"],
    'Happy': ["Keep smiling! You're already awesome 😊"],
    'Surprise': ["Enjoy the moment! Life is full of surprises 🎉"],
    'Neutral': ["You're doing great! Maybe try something new today 🚀"]
}

st.title("😊 Emotion Detector + Mood Booster")
st.write("Upload photo or use webcam → Get emotion + suggestions")

# Option 1: Upload Photo
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face.reshape(1,48,48,1) / 255.0
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Detected Emotion: **{emotion}**")
        
        # Show suggestions if needed
        if emotion in ['Sad', 'Fear', 'Angry', 'Disgust']:
            st.subheader("💡 AI Suggestions to Boost Your Mood")
            for tip in suggestions[emotion]:
                st.write("• " + tip)

# Option 2: Live Webcam (Real-time)

st.subheader("2. Live Webcam (Take Photo)")

camera_image = st.camera_input("Click to take a photo from your webcam")

if camera_image is not None:
    # Process the captured image
    bytes_data = camera_image.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    detected_emotion = "No face detected"
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
            
            prediction = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(prediction)
            detected_emotion = emotion_labels[emotion_idx]
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, detected_emotion, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"Detected Emotion: **{detected_emotion}**")
        
        # Show mood suggestions
        if detected_emotion in ['Sad', 'Fear', 'Angry', 'Disgust']:
            st.subheader("💡 AI Suggestions to Boost Your Mood")
            for tip in mood_tips[detected_emotion]:
                st.write(f"• {tip}")
        else:
            st.success(f"You're feeling **{detected_emotion}**! Great! 😊")
    else:
        st.warning("No face detected. Try again with better lighting.")
