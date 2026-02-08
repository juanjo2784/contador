import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador con Cámara", layout="wide")

st.title("📷 Contador Industrial en Tiempo Real")

# 1. Selectores de Modo
col_p, col_d = st.columns(2)
with col_p:
    producto = st.selectbox("Producto:", ["Cajas / Gruesas", "Separadores (Muy delgados)"])
with col_d:
    distancia_foto = st.radio("Distancia:", ["Cerca", "Lejos (Masivo)"], horizontal=True)

# 2. Parámetros Base (Ajustados para evitar conteo doble)
if producto == "Separadores (Muy delgados)":
    def_params = (18, 15, 3, 4.0) if distancia_foto == "Lejos (Masivo)" else (12, 10, 5, 2.0)
else:
    def_params = (40, 20, 7, 2.0)

dist, prom, blur, width = def_params

# 3. Sidebar de Ajuste Fino
st.sidebar.header("🕹️ Calibración")
s_dist = st.sidebar.slider("Separación (Distance)", 1, 200, dist)
s_prom = st.sidebar.slider("Sensibilidad (Prominence)", 1.0, 100.0, float(prom))
s_width = st.sidebar.slider("Grosor mínimo (Width)", 0.1, 20.0, width)
s_blur = st.sidebar.slider("Filtro Flash (Blur)", 1, 31, blur, step=2)

# --- NUEVO COMPONENTE DE CÁMARA ---
img_file = st.camera_input("Enfoca el material y toma la foto con Flash")

if img_file is not None:
    # Procesamiento instantáneo
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, s_blur), 0)
    
    alto, ancho = gray.shape
    centro = ancho // 2
    perfil = np.mean(blurred[:, centro-50 : centro+50], axis=1)
    perfil_inv = 255 - perfil 

    picos, _ = find_peaks(perfil_inv, distance=s_dist, prominence=s_prom, width=s_width)
    ia_total = len(picos)
    
    # Dibujo
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 2)
        if ia_total < 300:
            cv2.putText(img_res, str(i+1), (10, p - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resultados y Ajuste Manual
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.image(img_res, use_container_width=True, caption="Resultado del análisis")
    
    with col_res2:
        st.metric("Conteo IA", f"{ia_total} unidades")
        conteo_final = st.number_input("🔢 Ajuste Manual Final:", min_value=0, value=int(ia_total), step=1)
        
        if st.button("💾 Registrar en Inventario"):
            st.success(f"Registrado: {conteo_final} {producto}")

    with st.expander("📉 Diagnóstico de Señal"):
        st.line_chart(perfil_inv)