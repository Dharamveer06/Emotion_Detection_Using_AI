import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("😊 Face Emotion Detector + Mood Booster")
st.write("Simple & Clean Emotion Detection")

# Load the model

@st.cache\_resource
def load\_emotion\_pipeline():
return pipeline(
"image-classification",
model="abhilash88/face-emotion-detection",
device=-1
)

pipe = load\_emotion\_pipeline()

# Strong & Clean Emotion Mapping

emotion\_map = {
"label\_0": "Angry",   "LABEL\_0": "Angry",
"label\_1": "Disgust", "LABEL\_1": "Disgust",
"label\_2": "Fear",    "LABEL\_2": "Fear",
"label\_3": "Happy",   "LABEL\_3": "Happy",
"label\_4": "Sad",     "LABEL\_4": "Sad",
"label\_5": "Surprise","LABEL\_5": "Surprise",
"label\_6": "Neutral", "LABEL\_6": "Neutral"
}

# Mood Suggestions

mood\_tips = {
"Sad": ["Listen to upbeat music 🎵", "Take a short walk outside 🌳", "Call a friend ❤️", "Watch funny videos 😂"],
"Fear": ["Take 5 slow deep breaths 🧘", "Write 3 things you are grateful for ✨", "Drink warm tea ☕"],
"Angry": ["Do 10 jumping jacks 💪", "Listen to calm music 🎧", "Write and tear the paper"],
"Disgust": ["Watch cute animal videos 🐶", "Take a refreshing shower 🚿"],
"Happy": ["You're already awesome! Spread the positivity 😊"],
"Surprise": ["Enjoy this surprise moment! 🎉"],
"Neutral": ["You're calm. Try something new today 🚀"]
}

def predict\_emotion(image):
if isinstance(image, np.ndarray):
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR\_BGR2RGB))

```
results = pipe(image)
top = results[0]

raw = str(top['label']).strip().upper()
emotion = emotion_map.get(raw, "Neutral")   # Default to Neutral if unknown
confidence = round(top['score'] * 100, 1)

return emotion, confidence
```

# ====================== PHOTO UPLOAD ======================

st.subheader("1. Upload a Photo")
uploaded\_file = st.file\_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded\_file is not None:
file\_bytes = np.asarray(bytearray(uploaded\_file.read()), dtype=np.uint8)
img = cv2.imdecode(file\_bytes, cv2.IMREAD\_COLOR)

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) > 0:
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]
    emotion, confidence = predict_emotion(face_img)
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f"{emotion} ({confidence}%)", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"**{emotion}**")
else:
    emotion, confidence = predict_emotion(img)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"**{emotion}**")

if emotion in mood_tips and emotion in ["Sad", "Fear", "Angry", "Disgust"]:
    st.subheader("💡 Suggestions to Boost Your Mood")
    for tip in mood_tips[emotion]:
        st.write(f"• {tip}")
```

# ====================== WEBCAM ======================

st.subheader("2. Live Webcam")
camera\_image = st.camera\_input("📸 Take a photo from webcam")

if camera\_image is not None:
bytes\_data = camera\_image.getvalue()
img = cv2.imdecode(np.frombuffer(bytes\_data, np.uint8), cv2.IMREAD\_COLOR)

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
```

)

faces = face\_cascade.detectMultiScale(
gray,
scaleFactor=1.3,
minNeighbors=5
)

```
if len(faces) > 0:
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]
    emotion, confidence = predict_emotion(face_img)
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f"{emotion} ({confidence}%)", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"**{emotion}**")
else:
    emotion, confidence = predict_emotion(img)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"**{emotion}**")

if emotion in mood_tips and emotion in ["Sad", "Fear", "Angry", "Disgust"]:
    st.subheader("💡 Suggestions to Boost Your Mood")
    for tip in mood_tips[emotion]:
        st.write(f"• {tip}")
```
