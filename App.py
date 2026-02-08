import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador Pro con Flash", layout="wide")

st.title("📷 Contador Industrial (Uso de Flash Nativo)")

# --- INTERFAZ DE CONTROL ---
col_p, col_d = st.columns(2)
with col_p:
    producto = st.selectbox("Producto:", ["Cajas / Gruesas", "Separadores (Muy delgados)"])
with col_d:
    distancia_foto = st.radio("Distancia:", ["Cerca", "Lejos (Masivo)"], horizontal=True)

# Ajuste de sensibilidad automático
if producto == "Separadores (Muy delgados)":
    # Bajamos la prominencia (prom) para que detecte sombras más débiles
    def_params = (12, 5.0, 3, 2.0) if distancia_foto == "Lejos (Masivo)" else (8, 4.0, 3, 1.0)
else:
    def_params = (40, 15.0, 7, 2.0)

dist, prom, blur, width = def_params

st.sidebar.header("🕹️ Calibración de Precisión")
st.sidebar.info("Si faltan unidades: BAJA la Sensibilidad y el Grosor.")

s_dist = st.sidebar.slider("Separación mínima", 1, 150, dist)
s_prom = st.sidebar.slider("Sensibilidad (Baja para contar MÁS)", 0.5, 50.0, float(prom))
s_width = st.sidebar.slider("Grosor mínimo de sombra", 0.0, 15.0, width)
s_blur = st.sidebar.slider("Filtro de Ruido", 1, 25, blur, step=2)

# --- BOTÓN DE CÁMARA NATURALEZA ---
# El secreto: capture="camera" permite usar la cámara del sistema con FLASH
img_file = st.file_uploader("📸 PASO 1: Toma la foto con FLASH activo", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Procesamiento con realce de contraste
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ecualización limitada para resaltar sombras tenues (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (1, s_blur), 0)
    
    alto, ancho = gray.shape
    centro = ancho // 2
    perfil = np.mean(blurred[:, centro-40 : centro+40], axis=1)
    perfil_inv = 255 - perfil 

    # Detección
    picos, _ = find_peaks(perfil_inv, distance=s_dist, prominence=s_prom, width=s_width)
    ia_total = len(picos)
    
    # Visualización
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 2)
        cv2.putText(img_res, str(i+1), (15, p - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.image(img_res, use_container_width=True)
    
    with col_res2:
        st.metric("Contados por IA", f"{ia_total} un")
        conteo_final = st.number_input("🔢 Ajuste Manual:", min_value=0, value=int(ia_total))
        st.write(f"**Resultado Final: {conteo_final}**")

    with st.expander("📉 Análisis de picos (Si ves picos que no tienen línea verde, baja la Sensibilidad)"):
        st.line_chart(perfil_inv)