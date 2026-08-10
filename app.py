import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


# ====================== PAGE CONFIG ======================

st.set_page_config(
    page_title="Emotion Detector",
    layout="centered"
)

st.title("😊 Face Emotion Detector + Mood Booster")
st.write("Simple & Clean Emotion Detection")


# ====================== LOAD MODEL ======================

@st.cache_resource
def load_emotion_pipeline():
    return pipeline(
        "image-classification",
        model="abhilash88/face-emotion-detection",
        device=-1
    )


pipe = load_emotion_pipeline()


# ====================== EMOTION MAPPING ======================

emotion_map = {
    "label_0": "Angry",
    "LABEL_0": "Angry",

    "label_1": "Disgust",
    "LABEL_1": "Disgust",

    "label_2": "Fear",
    "LABEL_2": "Fear",

    "label_3": "Happy",
    "LABEL_3": "Happy",

    "label_4": "Sad",
    "LABEL_4": "Sad",

    "label_5": "Surprise",
    "LABEL_5": "Surprise",

    "label_6": "Neutral",
    "LABEL_6": "Neutral"
}


# ====================== MOOD SUGGESTIONS ======================

mood_tips = {
    "Sad": [
        "Listen to upbeat music 🎵",
        "Take a short walk outside 🌳",
        "Call a friend ❤️",
        "Watch funny videos 😂"
    ],

    "Fear": [
        "Take 5 slow deep breaths 🧘",
        "Write 3 things you are grateful for ✨",
        "Drink warm tea ☕"
    ],

    "Angry": [
        "Do 10 jumping jacks 💪",
        "Listen to calm music 🎧",
        "Write your thoughts on paper"
    ],

    "Disgust": [
        "Watch cute animal videos 🐶",
        "Take a refreshing shower 🚿"
    ],

    "Happy": [
        "You're already awesome! Spread the positivity 😊"
    ],

    "Surprise": [
        "Enjoy this surprise moment! 🎉"
    ],

    "Neutral": [
        "You're calm. Try something new today 🚀"
    ]
}


# ====================== PREDICT EMOTION ======================

def predict_emotion(image):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

    results = pipe(image)

    top = results[0]

    raw = str(top["label"]).strip()

    emotion = emotion_map.get(raw)

    # Try uppercase version if direct match fails
    if emotion is None:
        emotion = emotion_map.get(raw.upper(), "Neutral")

    confidence = round(top["score"] * 100, 1)

    return emotion, confidence


# ====================== PHOTO UPLOAD ======================

st.subheader("1. Upload a Photo")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) > 0:

        # Take first detected face
        x, y, w, h = faces[0]

        face_img = img[
            y:y + h,
            x:x + w
        ]

        # Predict emotion
        emotion, confidence = predict_emotion(face_img)

        # Draw rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Add emotion text
        cv2.putText(
            img,
            f"{emotion} ({confidence}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Display image
        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"{emotion} - {confidence}%"
        )

    else:

        # If no face detected,
        # predict using complete image
        emotion, confidence = predict_emotion(img)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"{emotion} - {confidence}%"
        )


    # ====================== MOOD TIPS ======================

    if emotion in mood_tips and emotion in [
        "Sad",
        "Fear",
        "Angry",
        "Disgust"
    ]:

        st.subheader("💡 Suggestions to Boost Your Mood")

        for tip in mood_tips[emotion]:
            st.write(f"• {tip}")


# ====================== WEBCAM ======================

st.subheader("2. Live Webcam")

camera_image = st.camera_input(
    "📸 Take a photo from webcam"
)


if camera_image is not None:

    # Get camera image bytes
    bytes_data = camera_image.getvalue()

    # Convert bytes to OpenCV image
    img = cv2.imdecode(
        np.frombuffer(
            bytes_data,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) > 0:

        # First detected face
        x, y, w, h = faces[0]

        face_img = img[
            y:y + h,
            x:x + w
        ]

        # Predict emotion
        emotion, confidence = predict_emotion(face_img)

        # Draw rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display emotion
        cv2.putText(
            img,
            f"{emotion} ({confidence}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"{emotion} - {confidence}%"
        )

    else:

        # No face detected
        emotion, confidence = predict_emotion(img)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"{emotion} - {confidence}%"
        )


    # ====================== MOOD TIPS ======================

    if emotion in mood_tips and emotion in [
        "Sad",
        "Fear",
        "Angry",
        "Disgust"
    ]:

        st.subheader("💡 Suggestions to Boost Your Mood")

        for tip in mood_tips[emotion]:
            st.write(f"• {tip}")
