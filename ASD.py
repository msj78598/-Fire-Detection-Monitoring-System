import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import torch
import cv2
import numpy as np
from PIL import Image
import time

# ✅ تحميل نموذج YOLOv5
MODEL_PATH = "best.pt"

# ✅ تحميل النموذج
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, source="github")
    model.conf = 0.1  # ضبط العتبة إلى 0.1
    print("✅ تم تحميل YOLOv5 بنجاح!")
except Exception as e:
    print(f"❌ خطأ في تحميل YOLOv5: {e}")

# ✅ تنسيق الصفحة
st.set_page_config(page_title="Fire Detection Monitoring", page_icon="🔥", layout="wide")

# ✅ واجهة التطبيق
st.title("🔥 Fire Detection Monitoring System")
st.markdown("<h4 style='text-align: center; color: #FF5733;'>نظام مراقبة لاكتشاف الحريق</h4>", unsafe_allow_html=True)

# ✅ إضافة خيار لاختيار الإدخال
mode = st.sidebar.radio("📌 اختر طريقة الإدخال:", ["🎥 الكاميرا المباشرة", "📂 رفع صورة أو فيديو"])

# ✅ 1️⃣ تشغيل الكاميرا عبر `Streamlit WebRTC`
if mode == "🎥 الكاميرا المباشرة":
    st.sidebar.warning("⚠️ ملاحظة: تأكد من السماح للمتصفح بالوصول إلى الكاميرا.")

    class FireDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # 🔹 تشغيل YOLOv5 على الإطار الحالي
            results = model(img, conf=0.1)  # ضبط العتبة إلى 0.1

            fire_detected = False
            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "🔥 Fire Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                fire_detected = True

            if fire_detected:
                # 🔴 إنذار ضوئي وميض أحمر
                for _ in range(5):
                    st.markdown("<div style='background-color: red; color: white; font-size: 24px; text-align: center;'>🚨🔥 إنذار حريق! 🔥🚨</div>", unsafe_allow_html=True)
                    time.sleep(0.5)
                    st.markdown("<div style='background-color: white; color: white; font-size: 24px; text-align: center;'> </div>", unsafe_allow_html=True)
                    time.sleep(0.5)

                # 🔊 تشغيل إنذار صوتي
                st.audio("mixkit-urgent-simple-tone-loop-2976.wav", autoplay=True)

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
            results = model(image_np, conf=0.1)  # ضبط العتبة إلى 0.1

            fire_detected = False
            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
                fire_detected = True

            # ✅ عرض الصورة مع النتائج
            st.image(image_np, caption="🔍 نتيجة تحليل الصورة", use_column_width=True)

            if fire_detected:
                st.success("✅ 🔥 تم اكتشاف حريق!")
                st.audio("mixkit-urgent-simple-tone-loop-2976.wav", autoplay=True)
            else:
                st.success("✅ لا يوجد حريق في الصورة.")

        elif file_type == "video":
            # ✅ تشغيل الفيديو وتحليله إطار بإطار
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            fire_detected = False
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 🔹 تشغيل YOLOv5 على الإطار الحالي
                results = model(frame, conf=0.1)  # ضبط العتبة إلى 0.1

                for *xyxy, conf, cls in results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    fire_detected = True

                # ✅ عرض الفيديو بعد التحليل
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, caption="🔍 تحليل الفيديو", use_column_width=True)

            cap.release()

            if fire_detected:
                st.success("✅ 🔥 تم اكتشاف حريق!")
                st.audio("mixkit-urgent-simple-tone-loop-2976.wav", autoplay=True)
            else:
                st.success("✅ لا يوجد حريق في الفيديو.")
