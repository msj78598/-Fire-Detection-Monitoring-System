import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import urllib.request
import av
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import time
from ultralytics import YOLO

# ✅ تثبيت المتطلبات اللازمة
os.system("pip install --upgrade ultralytics opencv-python-headless streamlit-webrtc")

# ✅ تحميل النموذج إذا لم يكن موجودًا أو كان تالفًا
MODEL_PATH = "best.pt"
MODEL_URL = "https://raw.githubusercontent.com/msj78598/Fire-Detection-Monitoring-System/main/best.pt"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
    st.warning("📥 يتم تحميل النموذج... يرجى الانتظار!")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("✅ تم تحميل النموذج بنجاح!")

# ✅ تحميل نموذج YOLOv5 مع معالجة خطأ `WindowsPath`
try:
    model = YOLO(str(Path(MODEL_PATH)))  # ✅ ضمان توافق المسار مع Linux
    st.session_state.model = model
    st.success("✅ تم تحميل نموذج YOLOv5 بنجاح!")
except Exception as e:
    st.error(f"❌ خطأ في تحميل YOLOv5: {e}")
    st.stop()  # 🔴 إيقاف التطبيق إذا لم يتم تحميل النموذج

# ✅ إعداد واجهة التطبيق
st.set_page_config(page_title="Fire Detection Monitoring", page_icon="🔥", layout="wide")
st.title("🔥 Fire Detection Monitoring System")
st.markdown("<h4 style='text-align: center; color: #FF5733;'>نظام مراقبة لاكتشاف الحريق</h4>", unsafe_allow_html=True)

# ✅ اختيار الإدخال (كاميرا أو رفع ملف)
mode = st.sidebar.radio("📌 اختر طريقة الإدخال:", ["🎥 الكاميرا المباشرة", "📂 رفع صورة أو فيديو"])

# ✅ 1️⃣ تشغيل الكاميرا عبر `Streamlit WebRTC`
if mode == "🎥 الكاميرا المباشرة":
    st.sidebar.warning("⚠️ تأكد من السماح للمتصفح بالوصول إلى الكاميرا.")

    class FireDetectionTransformer(VideoTransformerBase):
        def __init__(self):
            self.fire_detected = False

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # 🔹 تشغيل YOLOv5 على الإطار الحالي
            results = st.session_state.model(img)
            fire_detected = False

            # 🔹 رسم المربعات على الصورة عند اكتشاف الحريق
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.1:  # ✅ عتبة الثقة 0.1
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "🔥 Fire Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    fire_detected = True

            # ✅ تشغيل الإنذار الضوئي والصوتي عند اكتشاف الحريق
            if fire_detected:
                st.markdown("<div style='background-color: red; color: white; font-size: 24px; text-align: center;'>🚨🔥 إنذار حريق! 🔥🚨</div>", unsafe_allow_html=True)
                os.system("aplay mixkit-urgent-simple-tone-loop-2976.wav &")  # 🔊 تشغيل الصوت على Linux
                time.sleep(0.5)  # ⏳ وميض الإنذار

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="fire-detection", video_transformer_factory=FireDetectionTransformer)

# ✅ 2️⃣ رفع صورة أو فيديو وتحليلها
elif mode == "📂 رفع صورة أو فيديو":
    uploaded_file = st.sidebar.file_uploader("📸 قم برفع صورة أو فيديو", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded_file:
        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            # ✅ تحليل الصورة
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # 🔹 تشغيل YOLOv5 على الصورة
            results = st.session_state.model(image_np)

            # 🔹 رسم المربعات على الصورة
            fire_detected = False
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.1:  # ✅ عتبة الثقة 0.1
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    fire_detected = True

            # ✅ تشغيل الإنذار المرئي والصوتي
            if fire_detected:
                st.markdown("<div style='background-color: red; color: white; font-size: 24px; text-align: center;'>🚨🔥 إنذار حريق! 🔥🚨</div>", unsafe_allow_html=True)
                os.system("aplay mixkit-urgent-simple-tone-loop-2976.wav &")

            # ✅ عرض الصورة مع النتائج
            st.image(image_np, caption="🔍 نتيجة تحليل الصورة", use_column_width=True)
            st.success("✅ تم تحليل الصورة بنجاح!")

        elif file_type == "video":
            # ✅ تشغيل الفيديو وتحليله إطار بإطار
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 🔹 تشغيل YOLOv5 على الإطار الحالي
                results = st.session_state.model(frame)

                # 🔹 رسم المربعات على الصورة
                fire_detected = False
                for *xyxy, conf, cls in results.xyxy[0]:
                    if conf > 0.1:  # ✅ عتبة الثقة 0.1
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        fire_detected = True

                # ✅ تشغيل الإنذار عند اكتشاف الحريق
                if fire_detected:
                    st.markdown("<div style='background-color: red; color: white; font-size: 24px; text-align: center;'>🚨🔥 إنذار حريق! 🔥🚨</div>", unsafe_allow_html=True)
                    os.system("aplay mixkit-urgent-simple-tone-loop-2976.wav &")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, caption="🔍 تحليل الفيديو", use_column_width=True)

            cap.release()
            st.success("✅ تم تحليل الفيديو بنجاح!")
