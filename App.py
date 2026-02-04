import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Contador de Precisión", layout="wide")

st.sidebar.header("⚙️ Ajustes de Detección")
# Sliders para ajustar en tiempo real desde el móvil
sensibilidad = st.sidebar.slider("Umbral de bordes (Canny)", 10, 255, (50, 150))
area_min = st.sidebar.slider("Tamaño mínimo del cartón", 100, 5000, 1000)
iteraciones = st.sidebar.slider("Grosor de bordes (Dilatación)", 1, 5, 2)

st.title("📦 Contador de Cartones de Alta Precisión")

# Volvemos al cargador de archivos porque da más calidad que la cámara en vivo
img_file = st.file_uploader("Sube una foto clara", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 1. Limpieza de imagen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75) # Mantiene bordes, quita ruido

    # 2. Detección de bordes con los valores del slider
    edged = cv2.Canny(blurred, sensibilidad[0], sensibilidad[1])
    
    # 3. Cerrar huecos en los contornos
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=iteraciones)
    
    # 4. Encontrar contornos
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objetos = 0
    img_res = image.copy()

    for c in cnts:
        area = cv2.contourArea(c)
        if area > area_min:
            objetos += 1
            # Dibujar caja y número
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_res, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img_res, str(objetos), (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    st.metric("Total Detectado", f"{objetos} cartones")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(dilated, caption="Mapa de bordes (Lo que ve la IA)", use_container_width=True)
    with col2:
        st.image(img_res, caption="Resultado Final", use_container_width=True)