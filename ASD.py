import streamlit as st
import os
import torch
import cv2
from PIL import Image
from datetime import datetime
import pandas as pd
import time
import urllib.request

# ✅ ضبط مسار المشروع
PROJECT_DIR = "best.pt"
MODEL_PATH = os.path.join(PROJECT_DIR, "best.pt")
MODEL_URL = "https://raw.githubusercontent.com/msj78598/Fire-Detection-Monitoring-System/main/best.pt"

# ✅ التأكد من تحميل وتثبيت المكتبات المطلوبة
os.system("pip install --upgrade torch torchvision ultralytics pandas streamlit opencv-python")

# ✅ التحقق من وجود النموذج أو إعادة تحميله إذا كان غير موجود أو تالفًا
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
    print("❌ ملف best.pt غير موجود أو تالف، سيتم إعادة تحميله...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ تم تحميل best.pt بنجاح!")

# ✅ تحميل نموذج YOLOv5 باستخدام `torch.hub.load()`
if "model" not in st.session_state:
    st.session_state.model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=MODEL_PATH,
        source='github',
        trust_repo=True
    )

# ✅ تنسيق الصفحة
st.set_page_config(page_title="Fire Detection Monitoring", page_icon="🔥", layout="wide")

# ✅ شريط جانبي للإعدادات
st.sidebar.title("⚙️ الإعدادات")
st.sidebar.subheader("📊 إصدار تقرير")

# 📅 تحديد الفترة الزمنية لاستخراج التقرير
start_date = st.sidebar.date_input("📅 تاريخ البداية")
end_date = st.sidebar.date_input("📅 تاريخ النهاية")

# 🔽 استخراج التقرير
if st.sidebar.button("📥 استخراج التقرير"):
    if "fire_detections" in st.session_state and st.session_state.fire_detections:
        filtered_detections = [
            detection for detection in st.session_state.fire_detections
            if start_date <= datetime.strptime(detection['time'], "%Y-%m-%d %H:%M:%S").date() <= end_date
        ]

        if filtered_detections:
            df = pd.DataFrame(filtered_detections)
            image_folder = PROJECT_DIR
            df['image_link'] = df['image'].apply(lambda x: f'=HYPERLINK("{image_folder}/{x}", "عرض الصورة")')

            excel_file = os.path.join(PROJECT_DIR, "fire_detections_report.xlsx")
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

# ✅ نظام اكتشاف الحرائق
st.title("🔥 Fire Detection Monitoring System")
st.markdown("<h4 style='text-align: center; color: #FF5733;'>نظام مراقبة لاكتشاف الحريق</h4>", unsafe_allow_html=True)

# ✅ تهيئة المتغيرات في `st.session_state`
if "fire_detections" not in st.session_state:
    st.session_state.fire_detections = []
if "fire_images" not in st.session_state:
    st.session_state.fire_images = []

# ✅ زر تشغيل الكشف عن الحريق
start_detection = st.button('🚨 ابدأ الكشف عن الحريق 🚨')

# ✅ تهيئة عناصر واجهة المستخدم
alert_box = st.empty()
stframe = st.empty()
fire_images_placeholder = st.empty()

if start_detection:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ فشل في فتح الكاميرا. تأكد من أنها متصلة وجرب مرة أخرى.")
    else:
        fire_classes = [0, 1, 2, 3, 4]
        conf_threshold = 0.5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ خطأ في التقاط الفيديو.")
                break

            results = st.session_state.model(frame)
            detections = results.pandas().xyxy[0]
            detections = detections[detections['confidence'] > conf_threshold]

            fire_detected = False
            for _, detection in detections.iterrows():
                if detection['class'] in fire_classes:
                    fire_detected = True
                    x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
                    confidence = detection['confidence'] * 100

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"🔥 Fire: {confidence:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    now = datetime.now()
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    image_filename = os.path.join(PROJECT_DIR, f"fire_detected_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(image_filename, frame)

                    st.session_state.fire_images.insert(0, {'image': image_filename, 'timestamp': timestamp})
                    st.session_state.fire_detections.insert(0, {'time': timestamp, 'image': image_filename, 'confidence': confidence})

                    # 🔴 تشغيل الإنذار المرئي
                    for _ in range(5):
                        alert_box.markdown("<div style='background-color: red; color: white; font-size: 24px; text-align: center;'>🚨🔥 إنذار حريق! 🔥🚨</div>", unsafe_allow_html=True)
                        time.sleep(0.5)
                        alert_box.markdown("<div style='background-color: white; font-size: 24px; text-align: center;'> </div>", unsafe_allow_html=True)
                        time.sleep(0.5)

            # 📌 عرض الفيديو
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            stframe.image(img_pil, width=700)

            # 📌 عرض الصور المكتشفة
            if st.session_state.fire_images:
                fire_images_placeholder.subheader("🔥 الصور المكتشفة:")
                cols = fire_images_placeholder.columns(3)
                for idx, fire_image in enumerate(st.session_state.fire_images):
                    cols[idx % 3].image(fire_image['image'], caption=f"🕒 {fire_image['timestamp']}", use_column_width=True)

        cap.release()
