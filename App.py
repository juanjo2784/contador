import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador Industrial Pro", layout="wide")

st.title("📦 Sistema de Conteo de Alta Densidad")

# Modos simplificados
producto = st.selectbox("Producto:", ["Cajas / Gruesas", "Separadores (Muy delgados)"])
distancia_foto = st.radio("Foto tomada desde:", ["Cerca", "Lejos (Masivo)"], horizontal=True)

# Lógica de calibración automática basada en tus pruebas
if producto == "Separadores (Muy delgados)":
    if distancia_foto == "Lejos (Masivo)":
        # Si contaba 296, necesitamos subir la distancia y el ancho mínimo
        def_params = (15, 12, 1, 2.0) # Dist, Prom, Blur, Width
    else:
        def_params = (10, 8, 3, 1.5)
else:
    def_params = (40, 15, 7, 2.0)

dist, prom, blur, width = def_params

# Sidebar con el nuevo filtro de 'Ancho de pico'
st.sidebar.header("🕹️ Ajuste Fino")
s_dist = st.sidebar.slider("Separación (Evita doble conteo)", 1, 150, dist)
s_prom = st.sidebar.slider("Sensibilidad (Fuerza sombra)", 1, 100, prom)
s_width = st.sidebar.slider("Filtro de 'Grosor' de sombra", 3.5, 10.0, width)
s_blur = st.sidebar.slider("Filtro de Reflejo Flash", 1, 25, blur, step=2)

img_file = st.file_uploader("Sube la foto con Flash", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # 1. Cargar y procesar
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, s_blur), 0)
    
    # 2. Análisis lateral (Evita el brillo central del Flash)
    alto, ancho = gray.shape
    centro = ancho // 2
    perfil = np.mean(blurred[:, centro-60 : centro+60], axis=1)
    perfil_inv = 255 - perfil 

    # 3. Detección con filtro de ANCHO (width)
    # Esto es lo que filtrará el ruido del corrugado
    picos, _ = find_peaks(perfil_inv, 
                          distance=s_dist, 
                          prominence=s_prom,
                          width=s_width)
    
    total = len(picos)
    
    # 4. Dibujo
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 1)
        if total < 200:
            cv2.putText(img_res, str(i+1), (10, p - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    st.metric("Resultado", f"{total} unidades")
    st.image(img_res, use_container_width=True)
    
    with st.expander("Gráfico de picos (Analiza el grosor de las sombras)"):
        st.line_chart(perfil_inv)