import streamlit as st
import cv2
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Detector con Guardado", layout="wide")

st.title("📷 Detector y Guardado de Fotos")

# Cargador de archivos que abre la cámara
img_file = st.file_uploader("Capturar Foto", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # --- PROCESO DE GUARDADO ---
    # Generamos un nombre único basado en la hora para que no se sobrescriban
    nombre_archivo = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    bytes_data = img_file.getvalue()

    # Botón destacado para asegurar que la foto vaya a tu galería
    st.download_button(
        label="💾 GUARDAR FOTO EN GALERÍA",
        data=bytes_data,
        file_name=nombre_archivo,
        mime="image/jpeg",
        help="Haz clic aquí para que la foto se guarde en tu carpeta de Descargas/Galería"
    )

    # --- PROCESO DE ANÁLISIS (Tu lógica de etiquetas) ---
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # (Aquí va el resto de tu código de detección de contornos...)
    st.image(image, caption="Foto cargada correctamente", use_container_width=True)
    st.success(f"Foto recibida: {nombre_archivo}")