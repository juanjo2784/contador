import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Contador de Láminas", layout="wide")

st.sidebar.header("📏 Ajuste de Precisión")
sensibilidad = st.sidebar.slider("Sensibilidad de línea", 10, 255, 100)
min_ancho = st.sidebar.slider("Ancho mínimo de lámina", 50, 500, 200)

st.title("📦 Contador de Láminas Azules")

img_file = st.file_uploader("Sube la foto de las láminas", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 1. Pasar a gris y enfocar las sombras horizontales
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar un filtro para resaltar líneas horizontales
    # Usamos un kernel estirado (ancho) para ignorar ruidos verticales
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_lines = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_horiz)
    
    # 3. Umbralizado fuerte
    _, thresh = cv2.threshold(detect_lines, sensibilidad, 255, cv2.THRESH_BINARY)
    
    # 4. Dilatación horizontal para unir líneas rotas
    dilated = cv2.dilate(thresh, kernel_horiz, iterations=2)
    
    # 5. Contar las láminas basándonos en las sombras largas
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contador = 0
    img_res = image.copy()

    # Ordenar de arriba a abajo para numerar bien
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][1]))

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Filtro: Solo contamos si la línea es suficientemente ancha (una lámina)
        if w > min_ancho:
            contador += 1
            cv2.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_res, f"Lámina {contador}", (10, y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    st.metric("Total de láminas", f"{contador} unidades")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(dilated, caption="Líneas detectadas (Sombras)", use_container_width=True)
    with col2:
        st.image(img_res, caption="Resultado del Conteo", use_container_width=True)