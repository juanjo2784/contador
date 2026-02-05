import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador de Cajas Pro", layout="wide")

st.title("📦 Contador de Cajas Inteligente")

# 1. Selector de modo (Combo Box)
modo = st.selectbox(
    "Selecciona el tipo de carga:",
    ("Pocas unidades (Cerca/Grandes)", "Carga Masiva (Lejos/Delgadas)")
)

# 2. Configuración de parámetros según el modo
if modo == "Pocas unidades (Cerca/Grandes)":
    # Valores recomendados para cajas grandes o fotos de cerca
    def_dist = 45
    def_prom = 15
    def_blur = 11
    def_width = 2.0
else:
    # Valores recomendados para más de 50-100 cajas o fotos de lejos
    def_dist = 10
    def_prom = 5
    def_blur = 3
    def_width = 0.5

# 3. Sliders para ajuste fino en tiempo real (Sidebar)
st.sidebar.header("Ajuste Fino (Tiempo Real)")
dist_min = st.sidebar.slider("Separación entre líneas", 1, 200, def_dist)
prominencia = st.sidebar.slider("Sensibilidad de sombra", 1, 50, def_prom)
blur_v = st.sidebar.slider("Suavizado de imagen", 1, 31, def_blur, step=2)
ancho_pico = st.sidebar.slider("Grosor de línea detectada", 0.0, 10.0, def_width)

img_file = st.file_uploader("Sube la foto aquí", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Procesamiento inmediato
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, blur_v), 0)
    
    alto, ancho = gray.shape
    centro = ancho // 2
    # Analizamos el 10% del centro de la foto
    franja = int(ancho * 0.05)
    perfil = np.mean(blurred[:, centro-franja : centro+franja], axis=1)
    perfil_inv = 255 - perfil 

    # Detección con parámetros de los sliders
    picos, _ = find_peaks(perfil_inv, 
                          distance=dist_min, 
                          prominence=prominencia,
                          width=ancho_pico)
    
    total = len(picos)
    
    # Dibujo de resultados
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 2)
        # Solo numeramos si no son demasiadas para no saturar la vista
        if total < 100:
            cv2.putText(img_res, str(i+1), (15, p - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Métricas y visualización
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric("Conteo Final", f"{total} unidades")
        st.image(img_res, use_container_width=True)
    
    with col2:
        st.write("📈 Perfil de sombras")
        # Gráfico pequeño para referencia rápida
        st.line_chart(perfil_inv, height=400)
        if st.button("🔄 Resetear Parámetros"):
            st.rerun()