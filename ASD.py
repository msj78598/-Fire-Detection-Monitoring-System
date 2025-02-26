import os
import streamlit as st
import torch
import urllib.request
import pathlib
from ultralytics import YOLO

# ✅ تحميل النموذج وتحديثه تلقائيًا عند الحاجة
MODEL_PATH = "best.pt"
MODEL_URL = "https://raw.githubusercontent.com/msj78598/Fire-Detection-Monitoring-System/main/best.pt"

# 🔹 التأكد من أن الملف موجود وصالح
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
    st.warning("📥 يتم تحميل النموذج... يرجى الانتظار!")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("✅ تم تحميل النموذج بنجاح!")

try:
    # 🔹 تحويل المسار إلى نص (تجنب WindowsPath)
    model = YOLO(str(MODEL_PATH))  # ✅ تمرير المسار كنص عادي
    st.session_state.model = model
    st.success("✅ تم تحميل نموذج YOLOv5 بنجاح!")
except Exception as e:
    st.error(f"❌ خطأ في تحميل YOLOv5: {e}")
    st.stop()  # 🔴 إيقاف التطبيق إذا لم يتم تحميل النموذج بنجاح
