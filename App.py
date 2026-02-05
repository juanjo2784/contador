import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador Industrial Flash", layout="wide")

st.title("📦 Contador de Precisión (Modo Flash)")

# Selector de tipo de material
producto = st.selectbox("Tipo de producto:", ["Cajas / Láminas Gruesas", "Separadores (Muy delgados)"])
modo = st.radio("Densidad:", ["Pocas / Cerca", "Muchas / Lejos"], horizontal=True)

# Configuración automática optimizada para luz de Flash
if producto == "Cajas / Láminas Gruesas":
    def_params = (35, 20, 7, 1.5) if modo == "Pocas / Cerca" else (10, 10, 3, 0.5)
else:
    def_params = (6, 8, 3, 0.2) if modo == "Pocas / Cerca" else (3, 5, 1, 0.1)

dist, prom, blur, width = def_params

# Sliders de ajuste fino
st.sidebar.header("🕹️ Ajuste con Flash")
s_dist = st.sidebar.slider("Separación (Distance)", 1, 150, dist)
s_prom = st.sidebar.slider("Sensibilidad (Prominence)", 1, 100, prom)
s_blur = st.sidebar.slider("Filtro de Reflejo (Blur)", 1, 25, blur, step=2)

img_file = st.file_uploader("Captura o sube la foto con flash", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # El Flash puede crear ruido blanco; el Blur lo suaviza sin perder la sombra
    blurred = cv2.GaussianBlur(gray, (1, s_blur), 0)
    
    alto, ancho = gray.shape
    centro = ancho // 2
    # Si el flash brilla mucho en el centro, analizamos dos franjas laterales
    # Esto evita la mancha blanca central del flash
    franja_izq = np.mean(blurred[:, centro-60 : centro-20], axis=1)
    franja_der = np.mean(blurred[:, centro+20 : centro+60], axis=1)
    perfil = (franja_izq + franja_der) / 2
    
    perfil_inv = 255 - perfil 

    picos, _ = find_peaks(perfil_inv, distance=s_dist, prominence=s_prom)
    
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 2)
        if len(picos) < 101:
            cv2.putText(img_res, str(i+1), (10, p - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.metric("Total detectado", len(picos))
    st.image(img_res, use_container_width=True)
    
    with st.expander("Ver análisis de sombras del Flash"):
        st.line_chart(perfil_inv)