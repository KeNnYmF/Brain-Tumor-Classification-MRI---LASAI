# -----------------------------------------------------------------------------
# Servidor de Inferencia del Modelo de Clasificación de Tumores
#
# Descripción:
# Este script implementa un servidor web ligero usando Flask para desplegar
# el modelo de clasificación de tumores cerebrales entrenado.
#
# Funcionalidad:
# 1. Carga el modelo Keras (.keras) y los nombres de las clases al iniciar.
# 2. Define una ruta principal ('/') para servir la interfaz web (index.html).
# 3. Define una ruta API ('/predict') que acepta una imagen vía POST.
# 4. Pre-procesa la imagen de entrada (redimensión, normalización).
# 5. Ejecuta la inferencia del modelo.
# 6. Devuelve la clase predicha y la confianza como una respuesta JSON.
# -----------------------------------------------------------------------------

# Importación de bibliotecas y módulos necesarios
import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

# --- Parámetros de Configuración de la Aplicación ---
# Se instancia la aplicación Flask
app = Flask(__name__)

# Ruta al archivo del modelo Keras entrenado
MODEL_PATH = 'brain_tumor_vgg16_final.keras'
# Ruta al archivo de texto que contiene los nombres de las clases
CLASSES_PATH = 'class_names.txt'
# Dimensiones de imagen requeridas por el modelo
IMAGE_SIZE = (224, 224)
# -----------------------------------------------------

print(f"Cargando modelo desde {MODEL_PATH}...")
# Carga del modelo Keras desde el archivo .keras
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado exitosamente.")

# Carga de los nombres de las clases desde el archivo .txt
with open(CLASSES_PATH, 'r') as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]
print(f"Clases cargadas: {CLASS_NAMES}")

# Obtención de la función de pre-procesamiento específica de VGG16
preprocess_input = tf.keras.applications.vgg16.preprocess_input

def prepare_image(img_bytes):
    """
    Pre-procesa la imagen de entrada para que coincida con el formato
    requerido por el modelo VGG16.
    
    Argumentos:
        img_bytes (bytes): La imagen en formato de bytes.
    
    Retorna:
        tf.Tensor: El tensor de imagen pre-procesado y listo para la inferencia.
    """
    # Cargar la imagen desde bytes usando PIL
    img = Image.open(io.BytesIO(img_bytes))

    # Asegurar que la imagen esté en formato RGB (3 canales)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Redimensionar la imagen a las dimensiones de entrada del modelo
    img = img.resize(IMAGE_SIZE)
    
    # Convertir la imagen PIL a un array de NumPy
    img_array = tf.keras.utils.img_to_array(img)
    
    # Expandir las dimensiones para crear un 'batch' de tamaño 1
    # Formato de (224, 224, 3) a (1, 224, 224, 3)
    img_array = tf.expand_dims(img_array, 0)
    
    # Aplicar la normalización específica de VGG16 (escala de píxeles)
    return preprocess_input(img_array)

@app.route('/', methods=['GET'])
def index():
    """
    Ruta principal ('/'). Sirve la interfaz de usuario (frontend).
    """
    # Renderiza el archivo HTML principal desde la carpeta 'templates'
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Ruta de la API ('/predict'). Recibe una imagen, realiza la inferencia
    y devuelve los resultados.
    """
    # Validar que un archivo fue enviado en la petición
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo en la petición'}), 400

    file = request.files['file']
    
    try:
        # Leer el contenido del archivo en bytes
        img_bytes = file.read()
        
        # Preparar la imagen para el modelo
        processed_image = prepare_image(img_bytes)
        
        # Realizar la predicción (inferencia)
        # model.predict() devuelve un array de probabilidades
        prediction = model.predict(processed_image)[0]
        
        # Post-procesamiento: encontrar la clase con la probabilidad más alta
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction[predicted_index])
        
        # Devolver los resultados en formato JSON
        return jsonify({
            'class': predicted_class,
            'confidence': f"{(confidence * 100):.2f}%"
        })

    except Exception as e:
        # Manejo de errores durante el procesamiento o la inferencia
        print(f"Error durante la predicción: {e}")
        return jsonify({'error': 'Error interno al procesar la imagen'}), 500

# Punto de entrada para ejecutar el servidor Flask
if __name__ == '__main__':
    # app.run(debug=True) permite recargar el servidor automáticamente con cambios
    # y provee información de depuración detallada.
    app.run(debug=True)

