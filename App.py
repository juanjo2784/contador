import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Contador de Precisión", layout="wide")

st.title("📦 Contador de Etiquetas (Lógica Original)")

# --- CONFIGURACIÓN DE LA BARRA LATERAL ---
st.sidebar.header("⚙️ Ajustes de Detección")
ROI_X = st.sidebar.slider("ROI X", 0, 1500, 150)
ROI_Y = st.sidebar.slider("ROI Y", 0, 1500, 100)
ROI_W = st.sidebar.slider("Ancho ROI", 100, 2000, 350)
ROI_H = st.sidebar.slider("Alto ROI", 100, 2000, 250)
BRIGHTNESS_THRESHOLD = st.sidebar.slider("Umbral de Brillo", 0, 255, 200)
AREA_MIN = st.sidebar.slider("Área Mínima", 50, 5000, 300)

# --- CARGA DE FOTO ---
img_file = st.file_uploader("Capturar Foto", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    try:
        # Convertir el archivo subido a un array de bytes
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        # Decodificar la imagen para OpenCV
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("No se pudo leer la imagen. Intenta tomar otra foto.")
        else:
            # --- TU LÓGICA DE PROCESAMIENTO ---
            h_img, w_img = frame.shape[:2]
            
            # Ajustar ROI para que no se salga de la foto capturada
            x1, y1 = max(0, ROI_X), max(0, ROI_Y)
            x2, y2 = min(x1 + ROI_W, w_img), min(y1 + ROI_H, h_img)
            
            roi_original = frame[y1:y2, x1:x2].copy()

            if roi_original.size > 0:
                # 1. Tu técnica de Brillo (Canal V)
                hsv = cv2.cvtColor(roi_original, cv2.COLOR_BGR2HSV)
                v_channel = hsv[:, :, 2]
                _, bright_mask = cv2.threshold(v_channel, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
                
                # 2. Limpieza y Bordes (Canny)
                kernel = np.ones((3, 3), np.uint8)
                mask_cleaned = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
                edges = cv2.Canny(mask_cleaned, 80, 180)
                
                # 3. Conectar bordes (Cierre morfológico)
                edges_conn = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                # 4. Encontrar contornos y contar
                contours, _ = cv2.findContours(edges_conn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                conteo = 0
                for c in contours:
                    if cv2.contourArea(c) > AREA_MIN:
                        conteo += 1
                        cv2.drawContours(roi_original, [c], -1, (0, 255, 0), 2)
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.putText(roi_original, f"#{conteo}", (x, y-5), 1, 1.5, (0, 255, 0), 2)

                # --- MOSTRAR RESULTADOS ---
                st.success(f"Detección completada: {conteo} objetos encontrados.")
                
                # Mostrar comparación
                col1, col2 = st.columns(2)
                with col1:
                    st.image(mask_cleaned, caption="Máscara de Brillo", use_container_width=True)
                with col2:
                    st.image(roi_original, caption="Resultado Final", use_container_width=True)
            else:
                st.warning("El cuadro del ROI es inválido. Ajusta los valores en la barra lateral.")

    except Exception as e:
        st.error(f"Error crítico al procesar: {e}")