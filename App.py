import streamlit as st
import cv2
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Contador Pro", layout="centered")
st.title("📦 Analizador de Cartones")
st.write("Sube una foto clara para realizar el conteo automático.")

# Subida de archivo
img_file = st.file_uploader("Capturar o seleccionar imagen", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Convertir archivo a imagen OpenCV
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # --- PROCESAMIENTO ---
    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Suavizado para reducir ruido
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 3. Detección de bordes (Canny)
    # Ajusta los números 50 y 150 según la iluminación
    edged = cv2.Canny(blur, 50, 150)
    
    # 4. Dilatación para cerrar huecos en los bordes
    dilated = cv2.dilate(edged, None, iterations=1)
    
    # 5. Encontrar contornos
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos pequeños (ruido)
    objetos_detectados = 0
    output_image = image.copy()
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 500:  # Ajusta este valor según el tamaño de los cartones
            objetos_detectados += 1
            # Dibujar el contorno en verde
            cv2.drawContours(output_image, [c], -1, (0, 255, 0), 3)
    
    # --- MOSTRAR RESULTADOS ---
    st.metric(label="Total de Cartones Detectados", value=objetos_detectados)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(output_image, caption="Detección", use_container_width=True)

    st.info("💡 Consejo: Si el número no es exacto, intenta tomar la foto con fondo contrastado.")
  
