import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


# ==========================================
# STREAMLIT CONFIG
# ==========================================

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered"
)


st.title("😊 Face Emotion Detector")
st.subheader("AI Emotion Detection + Mood Booster")


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = pipeline(
        "image-classification",
        model="abhilash88/face-emotion-detection",
        device=-1
    )

    return model


try:

    emotion_model = load_model()

except Exception as error:

    st.error(
        f"❌ Model loading error: {error}"
    )

    st.stop()


# ==========================================
# EMOTION MAPPING
# ==========================================

EMOTION_MAP = {

    "LABEL_0": "Angry",
    "label_0": "Angry",

    "LABEL_1": "Disgust",
    "label_1": "Disgust",

    "LABEL_2": "Fear",
    "label_2": "Fear",

    "LABEL_3": "Happy",
    "label_3": "Happy",

    "LABEL_4": "Sad",
    "label_4": "Sad",

    "LABEL_5": "Surprise",
    "label_5": "Surprise",

    "LABEL_6": "Neutral",
    "label_6": "Neutral"
}


# ==========================================
# MOOD TIPS
# ==========================================

MOOD_TIPS = {

    "Sad": [
        "🎵 Listen to some upbeat music.",
        "🌳 Take a short walk outside.",
        "❤️ Talk to a friend or family member.",
        "😂 Watch something funny."
    ],

    "Fear": [
        "🧘 Take five slow deep breaths.",
        "✨ Think about three things you are grateful for.",
        "☕ Have a warm drink and relax."
    ],

    "Angry": [
        "💪 Take a short physical break.",
        "🎧 Listen to calm music.",
        "📝 Write down what is bothering you."
    ],

    "Disgust": [
        "🐶 Watch some cute animal videos.",
        "🚿 Take a refreshing shower."
    ],

    "Happy": [
        "😊 Keep spreading the positivity!"
    ],

    "Surprise": [
        "🎉 Enjoy the moment!"
    ],

    "Neutral": [
        "🚀 Try something new today!"
    ]
}


# ==========================================
# OPENCV CHECK
# ==========================================

def check_opencv():

    required_functions = [
        "imdecode",
        "cvtColor",
        "rectangle",
        "putText"
    ]

    missing = []

    for function_name in required_functions:

        if not hasattr(
            cv2,
            function_name
        ):

            missing.append(
                function_name
            )

    return missing


missing_functions = check_opencv()


if missing_functions:

    st.error(
        "OpenCV installation is incomplete."
    )

    st.write(
        "Missing OpenCV functions:",
        missing_functions
    )

    st.stop()


# ==========================================
# LOAD FACE DETECTOR
# ==========================================

@st.cache_resource
def load_face_detector():

    cascade_path = cv2.data.haarcascades

    cascade_path += (
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(
        cascade_path
    )

    if detector.empty():

        raise RuntimeError(
            "Haar Cascade XML file could not be loaded."
        )

    return detector


try:

    face_detector = load_face_detector()

except Exception as error:

    st.error(
        f"❌ Face detector error: {error}"
    )

    st.write(
        "OpenCV version:",
        cv2.__version__
    )

    st.write(
        "OpenCV location:",
        cv2.__file__
    )

    st.stop()


# ==========================================
# EMOTION PREDICTION
# ==========================================

def predict_emotion(face_image):

    rgb_image = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb_image
    )

    results = emotion_model(
        pil_image
    )

    best_result = results[0]

    label = str(
        best_result["label"]
    ).strip()

    emotion = EMOTION_MAP.get(
        label,
        EMOTION_MAP.get(
            label.upper(),
            "Neutral"
        )
    )

    confidence = round(
        best_result["score"] * 100,
        1
    )

    return emotion, confidence


# ==========================================
# PROCESS IMAGE
# ==========================================

def process_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:

        return image, None, None


    x, y, width, height = faces[0]

    face_image = image[
        y:y + height,
        x:x + width
    ]

    emotion, confidence = predict_emotion(
        face_image
    )


    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )


    cv2.putText(
        image,
        f"{emotion} ({confidence}%)",
        (x, max(y - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    return image, emotion, confidence


# ==========================================
# PHOTO UPLOAD
# ==========================================

st.divider()

st.header("📷 1. Upload a Photo")


uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )


    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    if image is None:

        st.error(
            "❌ Could not read the image."
        )

        st.stop()


    try:

        result_image, emotion, confidence = process_image(
            image
        )

    except Exception as error:

        st.error(
            f"❌ Prediction error: {error}"
        )

        st.stop()


    if emotion is None:

        st.warning(
            "😕 No face detected. Please upload another photo."
        )

    else:

        st.image(
            cv2.cvtColor(
                result_image,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"{emotion} - {confidence}%"
        )


        st.success(
            f"Detected Emotion: {emotion}"
        )

        st.info(
            f"Confidence: {confidence}%"
        )


        if emotion in MOOD_TIPS:

            st.subheader(
                "💡 Mood Suggestions"
            )

            for tip in MOOD_TIPS[emotion]:

                st.write(
                    tip
                )


# ==========================================
# WEBCAM
# ==========================================

st.divider()

st.header("📸 2. Webcam")


camera_image = st.camera_input(
    "Take a picture"
)


if camera_image is not None:

    image_bytes = camera_image.getvalue()


    image = cv2.imdecode(
        np.frombuffer(
            image_bytes,
            dtype=np.uint8
        ),
        cv2.IMREAD_COLOR
    )


    if image is None:

        st.error(
            "❌ Could not read webcam image."
        )

        st.stop()


    try:

        result_image, emotion, confidence = process_image(
            image
        )

    except Exception as error:

        st.error(
            f"❌ Prediction error: {error}"
        )

        st.stop()


    if emotion is None:

        st.warning(
            "😕 No face detected. Please try again."
        )

    else:

        st.image(
            cv2.cvtColor(
                result_image,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"{emotion} - {confidence}%"
        )


        st.success(
            f"Detected Emotion: {emotion}"
        )

        st.info(
            f"Confidence: {confidence}%"
        )


        if emotion in MOOD_TIPS:

            st.subheader(
                "💡 Mood Suggestions"
            )

            for tip in MOOD_TIPS[emotion]:

                st.write(
                    tip
                )
