import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador de Alta Precisión", layout="wide")

st.title("📦 Contador con Potenciador de Sombras")
st.info("💡 Consejo: Si no tienes flash, intenta que la luz venga de un lado para que las ranuras se vean más oscuras.")

# 1. Selectores de Modo
col_p, col_d = st.columns(2)
with col_p:
    producto = st.selectbox("Tipo de material:", ["Cajas / Gruesas", "Separadores (Muy delgados)"])
with col_d:
    modo = st.radio("Densidad:", ["Pocas / Cerca", "Muchas / Lejos"], horizontal=True)

# 2. Sidebar: Calibración Visual
st.sidebar.header("🕹️ Ajuste de Visión")
# Este es el nuevo control para cuando faltan unidades
potencia_sombra = st.sidebar.slider("Potenciador de Sombras (Gamma)", 0.5, 3.0, 1.5)
s_prom = st.sidebar.slider("Sensibilidad (Baja para contar MÁS)", 0.1, 50.0, 5.0)
s_dist = st.sidebar.slider("Separación mínima (Píxeles)", 1, 100, 10)
s_width = st.sidebar.slider("Grosor de ranura", 0.0, 10.0, 1.0)

# Volvemos al modo de cámara que te funcionó bien
img_file = st.camera_input("Toma la foto lo más derecho posible")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # --- PROCESAMIENTO AVANZADO DE CONTRASTE ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicamos Corrección Gamma para resaltar sombras tenues
    # Valores > 1 oscurecen las sombras; Valores < 1 aclaran la imagen
    inv_gamma = 1.0 / potencia_sombra
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray_potenciado = cv2.LUT(gray, table)
    
    # Filtro de nitidez para marcar bordes
    blurred = cv2.GaussianBlur(gray_potenciado, (1, 3), 0)
    
    alto, ancho = gray.shape
    centro = ancho // 2
    perfil = np.mean(blurred[:, centro-50 : centro+50], axis=1)
    perfil_inv = 255 - perfil 

    picos, _ = find_peaks(perfil_inv, distance=s_dist, prominence=s_prom, width=s_width)
    ia_total = len(picos)
    
    # Dibujo de resultados
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 2)
        if ia_total < 300:
            cv2.putText(img_res, str(i+1), (15, p - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.image(img_res, use_container_width=True)
    
    with col_res2:
        st.metric("IA detectó:", f"{ia_total} un")
        conteo_final = st.number_input("🔢 Ajuste Manual Final:", min_value=0, value=int(ia_total))
        
        if st.button("💾 Guardar Conteo"):
            st.success(f"Guardado: {conteo_final} unidades")

    with st.expander("📉 Mapa de Sombras (Si los picos son bajos, sube el Potenciador Gamma)"):
        st.line_chart(perfil_inv)