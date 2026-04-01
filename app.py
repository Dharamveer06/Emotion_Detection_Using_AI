import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="😊 Emotion Detector", layout="centered")
st.title("😊 Advanced Face Emotion Detector + Mood Booster")
st.write("Using **Hugging Face Vision Transformer** (better accuracy than basic CNN)")

# Load the emotion classification pipeline (downloads ~1GB first time, then cached)
@st.cache_resource
def load_emotion_model():
    # Good performing model - you can change to other HF models
    return pipeline(
        "image-classification", 
        model="Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large",
        # Alternative lighter model: "abhilash88/face-emotion-detection"
    )

pipe = load_emotion_model()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Mood suggestions (same as before)
mood_tips = {
    'Sad': ["Listen to upbeat music 🎵", "Take a short walk 🌳", "Call a friend ❤️", "Watch funny videos 😂"],
    'Fear': ["Take 5 deep breaths 🧘", "Write 3 things you're grateful for ✨", "Drink warm tea ☕"],
    'Angry': ["Do jumping jacks 💪", "Listen to calm music 🎧", "Write and rip the paper"],
    'Disgust': ["Watch cute animal videos 🐶", "Take a refreshing shower 🚿"],
    'Happy': ["You're already awesome! Spread the smile 😊"],
    'Surprise': ["Enjoy the moment! 🎉"],
    'Neutral': ["You're calm. Try something new today 🚀"]
}

def predict_emotion(image):
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    results = pipe(image)
    # Get top prediction
    top_result = results[0]
    emotion = top_result['label'].capitalize()
    confidence = top_result['score'] * 100
    return emotion, confidence

# ====================== PHOTO UPLOAD ======================
st.subheader("1. Upload a Photo")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Face detection for better cropping
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Take first face
        face_img = img[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face_img)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{emotion} ({confidence:.1f}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    else:
        # No face detected - use full image
        emotion, confidence = predict_emotion(img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    
    # Show mood suggestions
    if emotion in ['Sad', 'Fear', 'Angry', 'Disgust']:
        st.subheader("💡 AI Suggestions to Boost Your Mood")
        for tip in mood_tips.get(emotion, []):
            st.write(f"• {tip}")
    else:
        st.success(f"You're feeling **{emotion}**! Keep it up 😊")

# ====================== WEBCAM ======================
st.subheader("2. Live Webcam")
camera_image = st.camera_input("📸 Take a photo from webcam")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    emotion, confidence = predict_emotion(img)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
             caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    
    if emotion in ['Sad', 'Fear', 'Angry', 'Disgust']:
        st.subheader("💡 AI Suggestions to Boost Your Mood")
        for tip in mood_tips.get(emotion, []):
            st.write(f"• {tip}")
