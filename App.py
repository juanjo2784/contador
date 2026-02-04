import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Escaneo Total", layout="wide")

st.sidebar.header("🔍 Ajustes de Escaneo")
# Bajamos el mínimo para que no ignore nada
area_min = st.sidebar.slider("Sensibilidad de detección (Área)", 1000, 2000, 100)
brillo = st.sidebar.slider("Contraste / Brillo", 0.5, 3.0, 1.0)

st.title("📦 Escáner de Superficie Completa")

img_file = st.file_uploader("Sube la foto aquí", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Ajuste dinámico de contraste (ayuda a ver en toda la foto)
    image = cv2.convertScaleAbs(image, alpha=brillo, beta=0)
    
    # 1. Procesamiento de imagen completa
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Usamos Umbral Adaptativo: analiza la foto por bloques, no globalmente.
    # Esto es clave para que busque igual de bien en las esquinas que en el centro.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. Limpieza de ruido
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Encontrar contornos en TODA la imagen
    cnts, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objetos = 0
    img_res = image.copy()

    for c in cnts:
        area = cv2.contourArea(c)
        # 1. Filtro de área: Subimos el mínimo para ignorar esos cuadritos diminutos
        # Si tus objetos son grandes, este número debería ser mayor (ej. 1500 o 3000)
        if area > area_min:
            
            # 2. Filtro de forma (Opcional): Para asegurar que es algo sólido
            # Obtenemos el rectángulo que encierra el contorno
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w)/h
            
            # Dibujamos solo si es un objeto con un tamaño decente
            objetos += 1
            cv2.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 3) # Verde más grueso
            cv2.putText(img_res, f"Obj {objetos}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)