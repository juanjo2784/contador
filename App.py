import streamlit as st
import cv2
import numpy as np

# Configuración de la interfaz
st.set_page_config(page_title="Contador de Precisión", layout="wide")

st.title("📦 Contador de Etiquetas (Tu Lógica)")
st.write("Usa el botón para tomar una foto. Se aplicará tu filtro de brillo y morfología.")

# --- BARRA LATERAL PARA TUS AJUSTES ---
st.sidebar.header("⚙️ Parámetros de Tu Código")
ROI_X = st.sidebar.slider("ROI X", 0, 1000, 150)
ROI_Y = st.sidebar.slider("ROI Y", 0, 1000, 100)
ROI_W = st.sidebar.slider("Ancho ROI", 100, 1500, 350)
ROI_H = st.sidebar.slider("Alto ROI", 100, 1500, 250)

BRIGHTNESS_THRESHOLD = st.sidebar.slider("Umbral de Brillo (V)", 0, 255, 200)
KERNEL_SIZE_LARGE = st.sidebar.slider("Kernel de Cierre", 3, 15, 5, step=2)

# --- CAPTURA DE IMAGEN ---
img_file = st.file_uploader("Capturar Foto con Cámara Pro", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Convertir a formato OpenCV
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # --- 1. TU LÓGICA DE ROI ---
    x1, y1 = ROI_X, ROI_Y
    x2, y2 = ROI_X + ROI_W, ROI_Y + ROI_H
    
    # Asegurar que no se salga de la imagen
    h_img, w_img = frame.shape[:2]
    x2, y2 = min(x2, w_img), min(y2, h_img)
    roi_original = frame[y1:y2, x1:x2].copy()

    if roi_original.size > 0:
        # --- 2. TU PROCESAMIENTO (HSV + MÁSCARA DE BRILLO) ---
        hsv = cv2.cvtColor(roi_original, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # Umbral binario (tu técnica crítica)
        _, bright_mask = cv2.threshold(v_channel, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Limpieza morfológica (Apertura)
        kernel_open = np.ones((3, 3), np.uint8)
        bright_mask_cleaned = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Canny sobre la máscara
        edges_canny = cv2.Canny(bright_mask_cleaned, 80, 180)
        
        # Cierre morfológico (Tu técnica para unir bordes)
        kernel_close = np.ones((KERNEL_SIZE_LARGE, KERNEL_SIZE_LARGE), np.uint8)
        edges_continuous = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        edges_continuous = cv2.dilate(edges_continuous, kernel_close, iterations=1)

        # --- 3. CONTEO DE CONTORNOS ---
        contours, _ = cv2.findContours(edges_continuous, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        conteo = 0
        res_roi = roi_original.copy()
        
        for c in contours:
            area = cv2.contourArea(c)
            if area > 300: # Filtro de área mínima
                conteo += 1
                # Dibujar contorno y número (Tu petición)
                cv2.drawContours(res_roi, [c], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(c)
                cv2.putText(res_roi, f"#{conteo}", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 4. MOSTRAR RESULTADOS ---
        st.metric("Total Detectado", f"{conteo} unidades")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(bright_mask_cleaned, caption="Máscara de Brillo (Tu lógica)", use_container_width=True)
        with col2:
            st.image(edges_continuous, caption="Bordes Continuos", use_container_width=True)
            
        st.subheader("Imagen Final con Conteo")
        st.image(res_roi, caption="Resultado dentro del ROI", width=800)
        
        # Botón para descargar la foto procesada
        _, result_img_encoded = cv2.imencode('.jpg', res_roi)
        st.download_button(
            label="💾 Guardar Resultado en el Celular",
            data=result_img_encoded.tobytes(),
            file_name="conteo_resultado.jpg",
            mime="image/jpeg"
        )
    else:
        st.error("Ajusta el ROI, está fuera de la imagen.")