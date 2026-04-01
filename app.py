import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="😊 Emotion Detector", layout="centered")

st.title("😊 Face Emotion Detector + Mood Booster")
st.write("Using lightweight Vision Transformer (ViT) for better accuracy")

# Load pipeline with a stable, lightweight model
@st.cache_resource
def load_emotion_pipeline():
    return pipeline(
        "image-classification",
        model="abhilash88/face-emotion-detection",
        device=-1  # Force CPU (more stable on Streamlit Cloud)
    )

pipe = load_emotion_pipeline()

emotion_map = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral"
}

mood_tips = {
    "Sad": ["Listen to upbeat music 🎵", "Take a short walk 🌳", "Call a friend ❤️", "Watch funny videos 😂"],
    "Fear": ["Take 5 deep breaths 🧘", "Write 3 things you're grateful for ✨", "Drink warm tea ☕"],
    "Angry": ["Do jumping jacks 💪", "Listen to calm music 🎧", "Write and rip the paper"],
    "Disgust": ["Watch cute animal videos 🐶", "Take a refreshing shower 🚿"],
    "Happy": ["You're already awesome! Spread the smile 😊"],
    "Surprise": ["Enjoy the moment! 🎉"],
    "Neutral": ["You're calm. Try something new today 🚀"]
}

def predict_emotion(image):
    # Convert OpenCV image to PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    results = pipe(image)
    # Get top result
    top = results[0]
    raw_label = top['label'].lower()
    emotion = emotion_map.get(raw_label, raw_label.capitalize())
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    else:
        emotion, confidence = predict_emotion(img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    
    if emotion in mood_tips and emotion in ["Sad", "Fear", "Angry", "Disgust"]:
        st.subheader("💡 AI Suggestions to Boost Your Mood")
        for tip in mood_tips[emotion]:
            st.write(f"• {tip}")

# ====================== 2. WEBCAM ======================
st.subheader("2. Live Webcam")
camera_image = st.camera_input("📸 Take a photo from webcam")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    emotion, confidence = predict_emotion(img)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption=f"Detected: **{emotion}** ({confidence:.1f}%)")
    
    if emotion in ["Sad", "Fear", "Angry", "Disgust"]:
        st.subheader("💡 AI Suggestions to Boost Your Mood")
        for tip in mood_tips.get(emotion, []):
            st.write(f"• {tip}")
