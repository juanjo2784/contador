import streamlit as st
import cv2
import numpy as np

# Configuración de página y límite de subida (para fotos de alta resolución)
st.set_page_config(page_title="Contador Pro", layout="wide")

# Barra lateral para calibrar en vivo
st.sidebar.header("⚙️ Ajustes de Precisión")
sensibilidad = st.sidebar.slider("Sensibilidad de Bordes", 10, 255, (50, 150))
area_min = st.sidebar.slider("Tamaño del Cartón (Área)", 500, 10000, 2000)
dilatacion = st.sidebar.slider("Cierre de Contornos", 1, 5, 2)

st.title("📦 Contador de Cartones")
st.write("Pulsa abajo para abrir la cámara oficial, haz zoom y captura.")

# 1. El cargador de archivos (En móvil abre la cámara Pro con Zoom)
img_file = st.file_uploader("Capturar o seleccionar imagen", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # 2. Guardar la foto inmediatamente (Por si Android no la guarda en galería)
    bytes_data = img_file.getvalue()
    st.download_button(
        label="💾 Descargar foto original al móvil",
        data=bytes_data,
        file_name="foto_cartones.jpg",
        mime="image/jpeg"
    )

    # 3. Convertir para OpenCV
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 4. Procesamiento Visual
    # Convertimos a gris y aplicamos CLAHE (Mejora el contraste para ver mejor los bordes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_adj = clahe.apply(gray)
    
    # Suavizado para ignorar texturas del cartón y centrarse en la forma
    blurred = cv2.bilateralFilter(gray_adj, 9, 75, 75)
    
    # Detección de bordes y dilatación
    edged = cv2.Canny(blurred, sensibilidad[0], sensibilidad[1])
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=dilatacion)
    
    # 5. Conteo de objetos
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    conteo = 0
    res_img = image.copy()
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > area_min:
            conteo += 1
            x, y, w, h = cv2.boundingRect(c)
            # Dibujamos marco y número
            cv2.rectangle(res_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(res_img, f"#{conteo}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 6. Mostrar resultados
    st.success(f"✅ Se han detectado {conteo} unidades.")
    
    col1, col2 = st.columns(2)
    with col1: