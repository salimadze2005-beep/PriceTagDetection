import streamlit as st
import tempfile
from pathlib import Path
from run_pipeline import process_video

st.set_page_config(page_title="Lenta Price Checker")
st.title("Распознавание ценников")

uploaded_file = st.file_uploader("Загрузите видео", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Идёт обработка..."):
        csv_path = process_video(Path(tmp_path))

    if csv_path and Path(csv_path).exists():
        st.success("Готово!")
        with open(csv_path, 'rb') as f:
            st.download_button("Скачать CSV", f, file_name=Path(csv_path).name)
    else:
        st.error("Не удалось обработать видео.")