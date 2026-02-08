import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

# Filtro de Gradiente para detectar bordes sin flash
def detectar_bordes_v2(image, dist, prom, width):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Suavizado para evitar el ruido del grano del cartón
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Gradiente vertical (Sobel)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_y = np.absolute(grad_y)
    
    perfil = np.mean(abs_grad_y[:, abs_grad_y.shape[1]//2-30 : abs_grad_y.shape[1]//2+30], axis=1)
    picos, _ = find_peaks(perfil, distance=dist, prominence=prom, width=width)
    return picos, perfil

st.set_page_config(page_title="Contador Ultra Precisión", layout="wide")

st.title("📦 Contador de Alta Densidad (Modo Masivo)")

# Sliders con rangos extendidos para el caso de 150+ unidades
st.sidebar.header("🕹️ Calibración Masiva")
s_prom = st.sidebar.slider("Sensibilidad (Menos es más)", 0.1, 30.0, 3.0)
s_dist = st.sidebar.slider("Separación mínima (Píxeles)", 1, 50, 5)
s_width = st.sidebar.slider("Ancho de ranura", 0.0, 10.0, 0.5)

img_file = st.camera_input("Enfoca el paquete completo de 150 unidades")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 1. Convertir a Gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Realce de Bordes Horizontales (Sobel)
    # Esto ignora las fibras verticales del cartón y resalta las separaciones
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobely = np.uint8(np.absolute(sobely))
    
    # 3. Análisis de perfil
    alto, ancho = sobely.shape
    centro = ancho // 2
    perfil = np.mean(sobely[:, centro-40 : centro+40], axis=1)

    # En este modo, buscamos picos directos (no invertidos)
    picos, _ = find_peaks(perfil, distance=s_dist, prominence=s_prom, width=s_width)
    ia_total = len(picos)
    
    # Dibujo
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (255, 0, 0), 2)

    st.metric("Conteo Detectado", f"{ia_total} unidades")
    st.image(img_res, use_container_width=True)