import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador Industrial Pro", layout="wide")

st.title("📦 Sistema de Conteo de Alta Densidad")

# 1. Selectores de Modo
col_select1, col_select2 = st.columns(2)
with col_select1:
    producto = st.selectbox("Producto:", ["Cajas / Gruesas", "Separadores (Muy delgados)"])
with col_select2:
    distancia_foto = st.radio("Foto tomada desde:", ["Cerca", "Lejos (Masivo)"], horizontal=True)

# 2. Configuración de Parámetros Base (Ajustados para evitar exceso)
if producto == "Separadores (Muy delgados)":
    if distancia_foto == "Lejos (Masivo)":
        # Subimos dist y width para ser más selectivos
        def_params = (18, 15, 3, 4.0) 
    else:
        def_params = (12, 10, 5, 2.0)
else:
    def_params = (40, 20, 7, 2.0)

dist, prom, blur, width = def_params

# 3. Sidebar para Ajustes en Vivo
st.sidebar.header("🕹️ Calibración Anti-Error")
# RECUERDA: Si cuenta de más, SUBE estos tres valores:
s_dist = st.sidebar.slider("Separación (Distancia)", 1, 200, dist)
s_prom = st.sidebar.slider("Sensibilidad (Subir para contar MENOS)", 1.0, 100.0, float(prom))
s_width = st.sidebar.slider("Grosor mínimo de sombra", 0.1, 20.0, width)
s_blur = st.sidebar.slider("Filtro de Nitidez", 1, 31, blur, step=2)

img_file = st.file_uploader("Sube la foto con Flash", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Desenfoque bidireccional para eliminar el ruido del corrugado
    blurred = cv2.GaussianBlur(gray, (5, s_blur), 0)
    
    alto, ancho = gray.shape
    centro = ancho // 2
    # Analizamos una franja central un poco más robusta
    perfil = np.mean(blurred[:, centro-50 : centro+50], axis=1)
    perfil_inv = 255 - perfil 

    # Detección con los nuevos filtros
    picos, _ = find_peaks(perfil_inv, 
                          distance=s_dist, 
                          prominence=s_prom,
                          width=s_width)
    
    ia_total = len(picos)
    
    # Dibujo de líneas
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 2)
        if ia_total < 300:
            cv2.putText(img_res, str(i+1), (10, p - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- RESULTADOS ---
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        st.image(img_res, use_container_width=True)
    
    with col_res2:
        st.metric("Sugerencia de la IA", f"{ia_total} un")
        
        conteo_final = st.number_input("🔢 Conteo Final (Ajuste Manual):", 
                                       min_value=0, 
                                       value=int(ia_total), 
                                       step=1)
        
        if conteo_final == ia_total:
            st.success("✅ Conteo validado.")
        else:
            st.info(f"📝 Ajuste manual aplicado: {conteo_final}")

    with st.expander("📉 Análisis de Picos (Si hay ruido, verás picos pequeños que debes filtrar)"):
        st.line_chart(perfil_inv)