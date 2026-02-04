import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador de Capas", layout="wide")

st.title("📦 Contador de Láminas (Análisis de Perfil)")

st.sidebar.header("⚙️ Ajuste de Sensibilidad")
distancia = st.sidebar.slider("Distancia mínima entre láminas", 1, 50, 10)
promedio_ancho = st.sidebar.slider("Grosor de línea", 1, 21, 5, step=2)

img_file = st.file_uploader("Sube la foto de las láminas", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Tomar una franja central de la imagen para analizar el brillo
    # Analizamos el promedio de las columnas centrales para evitar ruido
    alto, ancho = gray.shape
    centro = ancho // 2
    franja = gray[:, centro-20 : centro+20]
    perfil_brillo = np.mean(franja, axis=1)
    
    # 3. Invertir el brillo (las sombras oscuras ahora serán "picos" altos)
    perfil_invertido = 255 - perfil_brillo
    
    # 4. Encontrar los picos (cada pico es una separación entre láminas)
    # Ajustamos la prominencia para detectar sombras sutiles
    picos, _ = find_peaks(perfil_invertido, distance=distancia, prominence=5)
    
    total_laminas = len(picos)
    
    # --- VISUALIZACIÓN ---
    img_res = image.copy()
    for i, p in enumerate(picos):
        # Dibujar una línea horizontal donde se detectó cada lámina
        cv2.line(img_res, (0, p), (ancho, p), (0, 255, 0), 2)
        cv2.putText(img_res, str(i+1), (20, p - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    st.metric("Conteo Total", f"{total_laminas} láminas")
    
    st.image(img_res, caption="Líneas de separación detectadas", use_container_width=True)
    
    # Mostrar el gráfico de lo que ve la IA (Opcional, ayuda a calibrar)
    with st.expander("Ver gráfico de análisis"):
        st.line_chart(perfil_invertido)