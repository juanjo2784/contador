import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Contador de cajas", layout="wide")

st.title("📦 Contador de cajas")

# Sidebar para calibración fina
st.sidebar.header("Calibración")
# Aumentamos el rango de distancia. Para 20 láminas en esa foto, 
# la distancia entre picos debe ser mayor.
dist_min = st.sidebar.slider("Separación mínima (píxeles)", 10, 200, 45)
prominencia = st.sidebar.slider("Fuerza de la sombra", 1, 50, 15)

img_file = st.file_uploader("Sube la foto", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 1. Procesamiento de imagen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicamos un desenfoque vertical para ignorar texturas pequeñas 
    # pero mantener las sombras horizontales
    blurred = cv2.GaussianBlur(gray, (1, 15), 0)
    
    # 2. Análisis de perfil (Columna central)
    alto, ancho = gray.shape
    centro = ancho // 2
    # Promediamos una franja para mayor estabilidad
    perfil = np.mean(blurred[:, centro-30 : centro+30], axis=1)
    
    # 3. Invertir para que las sombras sean picos
    perfil_inv = 255 - perfil
    
    # 4. Encontrar picos con los nuevos parámetros
    picos, properties = find_peaks(perfil_inv, 
                                   distance=dist_min, 
                                   prominence=prominencia)
    
    total = len(picos)
    
    # 5. Dibujar resultados
    img_res = image.copy()
    for i, p in enumerate(picos):
        color = (0, 255, 0) if total == 20 or total==25 else (0, 165, 255) # Verde si es exacto
        cv2.line(img_res, (0, p), (ancho, p), color, 3)
        cv2.putText(img_res, f"L{i+1}", (15, p - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Mostrar métricas
    if total == 20:
        st.success(f"✅ ¡LOGRADO! Se detectaron exactamente {total} láminas.")
    else:
        st.warning(f"✅ {total} unidades detectatas.")
    
    st.image(img_res, use_container_width=True)

    # Gráfico para entender qué está contando
    with st.expander("Gráfico de Análisis de Sombras"):
        st.line_chart(perfil_inv)