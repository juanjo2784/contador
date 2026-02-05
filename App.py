import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador Industrial Pro", layout="wide")

st.title("📦 Sistema de Conteo de Precisión")

# 1. Selectores de Modo
col_select1, col_select2 = st.columns(2)
with col_select1:
    producto = st.selectbox("Producto:", ["Cajas / Gruesas", "Separadores (Muy delgados)"])
with col_select2:
    distancia_foto = st.radio("Foto tomada desde:", ["Cerca", "Lejos (Masivo)"], horizontal=True)

# 2. Configuración de Parámetros Base
if producto == "Separadores (Muy delgados)":
    if distancia_foto == "Lejos (Masivo)":
        # Valores optimizados para filtrar el ruido del corrugado (el error de 296 a 150)
        def_params = (15, 12, 1, 3.0) # Dist, Prom, Blur, Width (3.0 es clave)
    else:
        def_params = (10, 8, 3, 1.5)
else:
    def_params = (40, 15, 7, 2.0)

dist, prom, blur, width = def_params

# 3. Sidebar para Ajustes en Vivo
st.sidebar.header("🕹️ Calibración de la IA")
s_dist = st.sidebar.slider("Separación (Evita doble línea)", 1, 150, dist)
s_prom = st.sidebar.slider("Sensibilidad (Fuerza sombra)", 1, 100, prom)
s_width = st.sidebar.slider("Filtro de 'Grosor' de sombra", 2.0, 10.0, width)
s_blur = st.sidebar.slider("Filtro de Reflejo Flash", 1, 25, blur, step=2)

img_file = st.file_uploader("Sube la foto con Flash", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Procesamiento de Imagen
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, s_blur), 0)
    
    # Análisis lateral para evitar el reflejo central del flash
    alto, ancho = gray.shape
    centro = ancho // 2
    perfil = np.mean(blurred[:, centro-60 : centro+60], axis=1)
    perfil_inv = 255 - perfil 

    # Detección de picos con filtro de ancho
    picos, _ = find_peaks(perfil_inv, 
                          distance=s_dist, 
                          prominence=s_prom,
                          width=s_width)
    
    ia_total = len(picos)
    
    # Dibujo de líneas
    img_res = image.copy()
    for i, p in enumerate(picos):
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 1)
        if ia_total < 201:
            cv2.putText(img_res, str(i+1), (10, p - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # --- ZONA DE RESULTADOS Y AJUSTE MANUAL ---
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        st.subheader("Visualización")
        st.image(img_res, use_container_width=True)
    
    with col_res2:
        st.subheader("Validación")
        st.metric("Sugerencia de la IA", f"{ia_total} un")
        
        # El ajuste manual toma el valor de la IA como base
        conteo_final = st.number_input("🔢 Conteo Final (Ajuste Manual):", 
                                       min_value=0, 
                                       value=int(ia_total), 
                                       step=1)
        
        if conteo_final == ia_total:
            st.success("✅ Conteo validado.")
        else:
            st.info(f"📝 Ajuste manual aplicado: {conteo_final}")
            
        # Botón para descargar el resultado (opcional)
        st.button("💾 Guardar en Inventario")

    with st.expander("📉 Análisis Técnico de Ondas"):
        st.line_chart(perfil_inv)