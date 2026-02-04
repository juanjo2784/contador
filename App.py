import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Contador en Vivo", layout="centered")

st.title("📷 Contador de Cartones Pro")
st.write("Apunta con tu cámara a los cartones y captura la imagen.")

# El secreto para móviles: 'camera_input' abre directamente la cámara del celular
img_file = st.camera_input("Tomar foto de los cartones")

if img_file is not None:
    # 1. Convertir la captura de la cámara a formato OpenCV
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # 2. Pre-procesamiento (Gris y desenfoque para evitar ruido)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Detección de bordes avanzada (Canny)
    edged = cv2.Canny(blurred, 40, 130)
    
    # 4. Dilatación (Une bordes que hayan quedado separados)
    dilated = cv2.dilate(edged, None, iterations=2)

    # 5. Encontrar contornos
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objetos_detectados = 0
    img_dibujo = image.copy()

    for c in cnts:
        # Filtro por área: evita contar pequeñas manchas o sombras
        area = cv2.contourArea(c)
        if area > 1000:  # Ajusta este número según la distancia a la que tomes la foto
            objetos_detectados += 1
            # Dibujamos un círculo o rectángulo sobre lo detectado
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_dibujo, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img_dibujo, f"#{objetos_detectados}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- RESULTADOS ---
    st.metric(label="Cartones contados", value=f"{objetos_detectados} unidades")
    
    st.image(img_dibujo, caption="Resultado del Análisis", use_container_width=True)

    if st.button("🔄 Reiniciar cámara"):
        st.rerun()