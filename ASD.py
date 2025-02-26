import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import urllib.request
import cv2
import av
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import time

# ✅ تثبيت المتطلبات تلقائيًا عند الحاجة
os.system("pip install --upgrade ultralytics opencv-python-headless streamlit-webrtc")

# ✅ تحميل `best.pt` إذا لم يكن موجودًا أو كان تالفًا
MODEL_PATH = "best.pt"
MODEL_URL = "https://raw.githubusercontent.com/msj78598/Fire-Detection-Monitoring-System/main/best.pt"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
    print("❌ ملف best.pt غير موجود أو تالف، سيتم إعادة تحميله...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ تم تحميل best.pt بنجاح!")

# ✅ تحميل YOLOv5 باستخدام `torch.hub.load()`
try:
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=MODEL_PATH,
        source="github",
        force_reload=True
    )
    print("✅ تم تحميل YOLOv5 بنجاح!")
except Exception as e:
    print(f"❌ خطأ في تحميل YOLOv5: {e}")

# ✅ إعداد الصفحة
st.set_page_config(page_title="Fire Detection Monitoring", page_icon="🔥", layout="wide")

# ✅ الشريط الجانبي
st.sidebar.title("⚙️ الإعدادات")
st.sidebar.subheader("📊 إصدار تقرير")
start_date = st.sidebar.date_input("📅 تاريخ البداية")
end_date = st.sidebar.date_input("📅 تاريخ النهاية")

if st.sidebar.button("📥 استخراج التقرير"):
    if "fire_detections" in st.session_state and st.session_state.fire_detections:
        filtered_detections = [
            detection for detection in st.session_state.fire_detections
            if start_date <= datetime.strptime(detection['time'], "%Y-%m-%d %H:%M:%S").date() <= end_date
        ]

        if filtered_detections:
            df = pd.DataFrame(filtered_detections)
            df['image_link'] = df['image'].apply(lambda x: f'=HYPERLINK("{x}", "عرض الصورة")')

            excel_file = "fire_detections_report.xlsx"
            df.to_excel(excel_file, index=False)

            with open(excel_file, "rb") as file:
                st.sidebar.download_button(
                    label="📥 تحميل التقرير",
                    data=file,
                    file_name="fire_detections_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.sidebar.error("❌ لا توجد اكتشافات في الفترة المحددة.")
    else:
        st.sidebar.error("❌ لا توجد اكتشافات لاستخراج التقرير.")

# ✅ واجهة التطبيق
st.title("🔥 Fire Detection Monitoring System")
st.markdown("<h4 style='text-align: center; color: #FF5733;'>نظام مراقبة لاكتشاف الحريق</h4>", unsafe_allow_html=True)

# ✅ اختيار الإدخال
mode = st.sidebar.radio("📌 اختر طريقة الإدخال:", ["🎥 الكاميرا المباشرة", "📂 رفع صورة أو فيديو"])

# ✅ 1️⃣ تشغيل الكاميرا عبر `Streamlit WebRTC`
if mode == "🎥 الكاميرا المباشرة":
    st.sidebar.warning("⚠️ تأكد من السماح للمتصفح بالوصول إلى الكاميرا.")

    class FireDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # 🔹 تشغيل YOLOv5 على الإطار الحالي
            results = model(img, conf=0.1)

            # 🔹 رسم المربعات على الصورة
            fire_detected = False
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.1:  # 🔥 عتبة الثقة 0.1
                    fire_detected = True
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "🔥 Fire Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if fire_detected:
                st.warning("🚨🔥 تم اكتشاف حريق! 🔥🚨")
                st.audio("mixkit-urgent-simple-tone-loop-2976.wav", autoplay=True)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="fire-detection", video_transformer_factory=FireDetectionTransformer)

# ✅ 2️⃣ تحليل صورة أو فيديو
elif mode == "📂 رفع صورة أو فيديو":
    uploaded_file = st.sidebar.file_uploader("📸 قم برفع صورة أو فيديو", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded_file:
        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # 🔹 تشغيل YOLOv5 على الصورة
            results = model(image_np, conf=0.1)

            # 🔹 رسم المربعات على الصورة
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.1:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)

            st.image(image_np, caption="🔍 نتيجة تحليل الصورة", use_column_width=True)
            st.success("✅ تم تحليل الصورة بنجاح!")

        elif file_type == "video":
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=0.1)

                for *xyxy, conf, cls in results.xyxy[0]:
                    if conf > 0.1:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, caption="🔍 تحليل الفيديو", use_column_width=True)

            cap.release()
            st.success("✅ تم تحليل الفيديو بنجاح!")
