import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="😊 Emotion Detector", layout="centered")

st.title("😊 Face Emotion Detector + Mood Booster")
st.write("Using Vision Transformer - Fixed Emotion Names")

# Load the model
@st.cache_resource
def load_emotion_pipeline():
    return pipeline(
        "image-classification",
        model="abhilash88/face-emotion-detection",
        device=-1
    )

pipe = load_emotion_pipeline()

# ✅ FIXED & STRONG Emotion Mapping
emotion_map = {
    "label_0": "Angry",
    "label_1": "Disgust",
    "label_2": "Fear",
    "label_3": "Happy",
    "label_4": "Sad",
    "label_5": "Surprise",
    "label_6": "Neutral"
}

mood_tips = {
    "Sad": ["Listen to upbeat music 🎵", "Take a short walk outside 🌳", "Call a friend ❤️", "Watch funny videos 😂"],
    "Fear": ["Take 5 slow deep breaths 🧘", "Write 3 things you are grateful for ✨", "Drink warm tea ☕"],
    "Angry": ["Do 10 jumping jacks 💪", "Listen to calm music 🎧", "Write and tear the paper"],
    "Disgust": ["Watch cute animal videos 🐶", "Take a refreshing shower 🚿"],
    "Happy": ["You're already awesome! Spread the positivity 😊"],
    "Surprise": ["Enjoy this surprise moment! 🎉"],
    "Neutral": ["You're calm. Try something new today 🚀"]
}

def predict_emotion(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    results = pipe(image)
    top = results[0]
    
    raw_label = top['label']
    # Force mapping
    emotion = emotion_map.get(raw_label, raw_label)
    confidence = top['score'] * 100
    
    return emotion, confidence

# ====================== 1. PHOTO UPLOAD ======================
st.subheader("1. Upload a Photo")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = img[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face_img)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{emotion} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    else:
        emotion, confidence = predict_emotion(img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    
    if emotion in ["Sad", "Fear", "Angry", "Disgust"]:
        st.subheader("💡 AI Suggestions to Boost Your Mood")
        for tip in mood_tips.get(emotion, []):
            st.write(f"• {tip}")
    else:
        st.success(f"You're feeling **{emotion}**! Keep it up 😊")

# ====================== 2. WEBCAM ======================
st.subheader("2. Live Webcam")
camera_image = st.camera_input("📸 Take a photo from webcam")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = img[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face_img)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{emotion} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    else:
        emotion, confidence = predict_emotion(img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                 caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    
    if emotion in ["Sad", "Fear", "Angry", "Disgust"]:
        st.subheader("💡 AI Suggestions to Boost Your Mood")
        for tip in mood_tips.get(emotion, []):
            st.write(f"• {tip}")
    else:
        st.success(f"You're feeling **{emotion}**! Keep it up 😊")
