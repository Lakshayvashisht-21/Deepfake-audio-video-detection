import streamlit as st
import os
import cv2
import torch
import numpy as np
import uuid
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
import ffmpeg
import librosa

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="TrustNet – Multimodal Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <h1 style='text-align: center;'>🛡️ TrustNet</h1>
    <h4 style='text-align: center; color: gray;'>
    Agentic Multimodal AI for Deepfake Detection (Video + Audio)
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.header("⚙️ System Overview")
    st.markdown("""
    **Agents**
    - 🎥 Visual Agent (Faces)
    - 🎧 Audio Agent (Speech)
    - 🧠 Fusion Agent (Decision)
    """)
    st.markdown("---")
    st.markdown("**Inference:** CPU")
    st.markdown("**Learning:** Offline (Feedback-driven)")

# ==================================================
# FOLDERS
# ==================================================
UPLOAD_FOLDER = "uploaded_videos"
FRAMES_FOLDER = "extracted_frames"
FACES_FOLDER = "extracted_faces"
AUDIO_FOLDER = "extracted_audio"
FEEDBACK_FILE = "feedback_log.csv"

for f in [UPLOAD_FOLDER, FRAMES_FOLDER, FACES_FOLDER, AUDIO_FOLDER]:
    os.makedirs(f, exist_ok=True)

# ==================================================
# VIDEO UPLOAD
# ==================================================
st.subheader("📤 Upload Video")
uploaded_video = st.file_uploader(
    "Supported formats: MP4, AVI, MOV",
    type=["mp4", "avi", "mov"]
)

# ==================================================
# FRAME EXTRACTION (ADAPTIVE)
# ==================================================
def extract_frames(video_path, output_folder, max_cap=400):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    MIN_FRAMES = 20
    desired_frames = min(max(int(duration * 0.8), MIN_FRAMES), max_cap)
    interval = max(total_frames // desired_frames, 1)

    count = saved = 0
    os.makedirs(output_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(
                os.path.join(output_folder, f"frame_{saved}.jpg"),
                frame
            )
            saved += 1
            if saved >= desired_frames:
                break
        count += 1

    cap.release()
    return saved

# ==================================================
# FACE EXTRACTION
# ==================================================
def extract_faces(frames_folder, faces_folder):
    detector = MTCNN(keep_all=True, device="cpu")
    os.makedirs(faces_folder, exist_ok=True)
    face_count = 0

    for img_name in os.listdir(frames_folder):
        image = Image.open(os.path.join(frames_folder, img_name)).convert("RGB")
        boxes, _ = detector.detect(image)
        if boxes is None:
            continue

        # Keep only largest faces
        boxes = sorted(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )[:2]

        img_np = np.array(image)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img_np[y1:y2, x1:x2]
            if face.size == 0:
                continue
            Image.fromarray(face).save(
                os.path.join(faces_folder, f"face_{face_count}.jpg")
            )
            face_count += 1

    return face_count

# ==================================================
# LOAD VIDEO MODEL (CACHED)
# ==================================================
@st.cache_resource
def load_video_model():
    processor = AutoImageProcessor.from_pretrained(
        "prithivMLmods/Deep-Fake-Detector-Model"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "prithivMLmods/Deep-Fake-Detector-Model"
    )
    model.eval()
    return processor, model

# ==================================================
# LOAD AUDIO MODEL (CACHED)
# ==================================================
@st.cache_resource
def load_audio_model():
    extractor = AutoFeatureExtractor.from_pretrained(
        "anton-l/wav2vec2-base-superb-sd"
    )
    model = AutoModelForAudioClassification.from_pretrained(
        "anton-l/wav2vec2-base-superb-sd"
    )
    model.eval()
    return extractor, model

# ==================================================
# VIDEO ANALYSIS
# ==================================================
def analyze_video_faces(faces_folder):
    processor, model = load_video_model()
    fake_scores = []

    for face in os.listdir(faces_folder):
        image = Image.open(os.path.join(faces_folder, face)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        fake_scores.append(probs[0][1].item())

    if not fake_scores:
        return None, None

    return float(np.median(fake_scores)), float(np.var(fake_scores))

# ==================================================
# AUDIO EXTRACTION + ANALYSIS
# ==================================================
def extract_audio(video_path, audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path

def analyze_audio(audio_path):
    extractor, model = load_audio_model()
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs[0][1].item()  # fake probability

# ==================================================
# PIPELINE
# ==================================================
if uploaded_video:
    video_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.video(video_path)
    progress = st.progress(10)

    frames_path = os.path.join(FRAMES_FOLDER, video_id)
    faces_path = os.path.join(FACES_FOLDER, video_id)
    audio_path = os.path.join(AUDIO_FOLDER, f"{video_id}.wav")

    extract_frames(video_path, frames_path)
    progress.progress(30)

    face_count = extract_faces(frames_path, faces_path)
    progress.progress(50)

    video_fake, video_var = analyze_video_faces(faces_path)
    progress.progress(70)

    try:
        extract_audio(video_path, audio_path)
        audio_fake = analyze_audio(audio_path)
    except:
        audio_fake = None

    progress.progress(90)

    # ==================================================
    # FUSION AGENT
    # ==================================================
    if audio_fake is not None and video_fake is not None:
        final_fake = 0.6 * video_fake + 0.4 * audio_fake
    elif video_fake is not None:
        final_fake = video_fake
    else:
        final_fake = audio_fake

    st.subheader("🧠 Multimodal Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("🎥 Video Fake Risk", f"{video_fake:.2f}" if video_fake else "N/A")
    c2.metric("🎧 Audio Fake Risk", f"{audio_fake:.2f}" if audio_fake else "N/A")
    c3.metric("🧠 Combined Risk", f"{final_fake:.2f}")

    st.markdown("### 🧾 Final Verdict")

    if final_fake > 0.75:
        st.error("🚨 High Confidence Deepfake")
    elif final_fake < 0.4:
        st.success("✅ Likely Authentic")
    else:
        st.warning("⚠️ Suspicious – Manual Review Recommended")

    # ==================================================
    # USER FEEDBACK
    # ==================================================
    st.markdown("### 🙋 User Feedback")

    feedback = st.radio(
        "Are you satisfied with this result?",
        ["Agree", "Disagree"]
    )

    if st.button("Submit Feedback"):
        import csv
        file_exists = os.path.isfile(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "video_id", "timestamp",
                    "video_fake", "audio_fake",
                    "final_fake", "user_feedback"
                ])
            writer.writerow([
                video_id, timestamp,
                video_fake, audio_fake,
                final_fake, feedback
            ])
        st.success("✅ Feedback recorded")

    st.caption("© TrustNet – Multimodal Agentic AI Prototype")


