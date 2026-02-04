import streamlit as st
import cv2
import numpy as np

# --- Configuración de Streamlit ---
st.set_page_config(page_title="Detector de Etiquetas Pro", layout="wide")

st.title("🏷️ Detector de Etiquetas Transparentes")
st.write("Sube una foto clara para detectar el contorno de la etiqueta.")

# --- Controles en la Barra Lateral (Tus variables originales) ---
st.sidebar.header("⚙️ Ajustes de Detección")
ROI_X = st.sidebar.number_input("ROI X (Inicio)", 0, 2000, 150)
ROI_Y = st.sidebar.number_input("ROI Y (Inicio)", 0, 2000, 100)
ROI_W = st.sidebar.number_input("Ancho ROI", 50, 2000, 350)
ROI_H = st.sidebar.number_input("Alto ROI", 50, 2000, 250)

BRIGHTNESS_THRESHOLD = st.sidebar.slider("Umbral de Brillo", 0, 255, 200)
KERNEL_SIZE = st.sidebar.slider("Grosor de Conexión (Kernel)", 3, 11, 5, step=2)

# --- Entrada de Imagen ---
# Usamos uploader para permitir la cámara del teléfono con zoom
img_file = st.file_uploader("Capturar Foto", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Leer imagen
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # 1. Definir ROI (Región de Interés)
    y1, y2 = ROI_Y, ROI_Y + ROI_H
    x1, x2 = ROI_X, ROI_X + ROI_W
    
    # Asegurar límites
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi_original = frame[y1:y2, x1:x2].copy()

    if roi_original.size == 0:
        st.error("⚠️ El ROI está fuera de los límites de la foto. Ajusta los valores en la barra lateral.")
    else:
        # --- Tu Lógica de Procesamiento ---
        # 1. HSV y Brillo
        hsv = cv2.cvtColor(roi_original, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # 2. Umbral de Brillo
        _, bright_mask = cv2.threshold(v_channel, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)

        # 3. Limpieza Morfológica
        kernel_small = np.ones((3,3), np.uint8)
        bright_mask_cleaned = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_small)

        # 4. Canny y Cierre (Conectar bordes)
        edges_canny = cv2.Canny(bright_mask_cleaned, 80, 180)
        kernel_large = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        edges_continuous = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        edges_continuous = cv2.dilate(edges_continuous, kernel_large, iterations=1)

        # 5. Detección de Contornos
        contours, _ = cv2.findContours(edges_continuous, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        res_roi = roi_original.copy()
        found = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                found = True
                # Dibujar contorno verde
                cv2.drawContours(res_roi, [largest_contour], -1, (0, 255, 0), 3)
                st.success("✅ ¡Etiqueta Detectada!")
        
        if not found:
            st.warning("⚠️ No se detectó la etiqueta. Prueba bajando el 'Umbral de Brillo'.")

        # --- Visualización ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(bright_mask_cleaned, caption="Máscara de Brillo (Lo que ve el sensor)", use_container_width=True)
        
        with col2:
            # Dibujar el ROI en el frame completo para mostrar dónde buscó
            frame_visual = frame.copy()
            cv2.rectangle(frame_visual, (x1, y1), (x2, y2), (255, 0, 0), 5)
            st.image(frame_visual, caption="Área de Búsqueda (ROI)", use_container_width=True)
            
        st.subheader("Resultado Final (Zoom al ROI)")
        st.image(res_roi, width=600)