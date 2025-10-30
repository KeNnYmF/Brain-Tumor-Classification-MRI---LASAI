# -----------------------------------------------------------------------------
# Script de Entrenamiento para Clasificación de Tumores Cerebrales (MRI)
#
# Descripción:
# Este script implementa un pipeline de Deep Learning para clasificar imágenes
# de MRI en cuatro categorías: glioma_tumor, meningioma_tumor, no_tumor,
# y pituitary_tumor.
#
# Metodología:
# 1. Carga de datos de entrenamiento y prueba.
# 2. Cálculo de ponderaciones de clase (class weights) para manejar el
#    desbalance en el dataset (específicamente la clase 'no_tumor').
# 3. Definición de una capa de aumento de datos (data augmentation) robusta
#    para mejorar la generalización y reducir el sobreajuste.
# 4. Implementación de Transfer Learning usando VGG16 pre-entrenado en ImageNet.
# 5. FASE 1: Entrenamiento del clasificador (head) con el 'backbone' congelado.
# 6. FASE 2: Ajuste fino (Fine-Tuning) del último bloque convolucional de VGG16
#    con una tasa de aprendizaje (learning rate) muy baja.
# 7. Evaluación del modelo final contra el conjunto de prueba.
# 8. Generación de reportes de métricas (Reporte de Clasificación, Matriz
#    de Confusión) y gráficas de historial de entrenamiento.
# -----------------------------------------------------------------------------

# Importación de bibliotecas y módulos necesarios
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Flatten, Dropout, GlobalAveragePooling2D,
    RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast
)
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# --- Parámetros de Configuración del Modelo y Entrenamiento ---
IMAGE_SIZE = (224, 224)  # Dimensión de entrada requerida por VGG16
BATCH_SIZE = 32
TRAIN_DIR = 'dataset/Training'
TEST_DIR = 'dataset/Testing'

# Configuración de épocas para las dos fases de entrenamiento
EPOCHS_FASE_1 = 25  # Épocas para el entrenamiento inicial del 'head'
EPOCHS_FASE_2 = 25  # Épocas máximas para la fase de Fine-Tuning

# --- 1. Carga de Datos y Pre-procesamiento ---
print("Cargando datasets...")

# Carga del conjunto de entrenamiento y creación de un subconjunto de validación (80/20 split)
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Carga del conjunto de prueba. shuffle=False es crucial para una evaluación correcta.
test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Guardar los nombres de las clases inferidos de la estructura de carpetas
class_names = train_ds.class_names
with open('class_names.txt', 'w') as f:
    for item in class_names:
        f.write(f"{item}\n")
print(f"Clases encontradas: {class_names}")

# --- 2. Cálculo de Ponderaciones de Clase (Class Weights) ---
# Para mitigar el desbalance de clases (menos instancias de 'no_tumor'),
# se calculan pesos inversamente proporcionales a la frecuencia de cada clase.
print("Calculando pesos de clase para manejar el desbalance...")
y_train = np.concatenate([y for x, y in train_ds], axis=0)
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights_array))
print("Pesos de clase calculados (clases minoritarias tendrán peso > 1.0):")
print(class_weight_dict)

# --- 3. Definición de la Capa de Aumento de Datos (Data Augmentation) ---
# Se aplica un conjunto de transformaciones aleatorias robustas para
# incrementar la variabilidad del dataset de entrenamiento, mejorar la
# robustez del modelo y reducir la confusión entre clases similares.
data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2),
        RandomBrightness(0.2)
    ],
    name="data_augmentation"
)

# --- 4. Construcción del Modelo de Transfer Learning (VGG16) ---
print("Construyendo modelo...")

# Pre-procesamiento específico (normalización) requerido por VGG16
preprocess_input = tf.keras.applications.vgg16.preprocess_input

# Se instancia el modelo base VGG16 pre-entrenado en ImageNet
base_model = VGG16(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,  # Excluir la capa clasificadora original de 1000 clases
    weights='imagenet'
)

# Se congela el 'backbone' (base_model.trainable = False)
# Solo los pesos del clasificador (head) serán entrenados en la Fase 1.
base_model.trainable = False

# Creación del modelo final (Sequential API)
inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
x = data_augmentation(inputs)      # Aplicar aumentos
x = preprocess_input(x)            # Normalizar píxeles
x = base_model(x, training=False)  # Modo inferencia para el 'backbone'
x = GlobalAveragePooling2D()(x)    # Aplanar características 3D a 1D
x = Dropout(0.5)(x)                # Capa de regularización
outputs = Dense(len(class_names), activation='softmax')(x) # Capa de salida
model = tf.keras.Model(inputs, outputs)

# --- 5. FASE 1: Entrenamiento del Clasificador (Head) ---
print("--- Iniciando FASE 1: Entrenamiento del 'Head' ---")

# Compilación del modelo para la Fase 1
model.compile(
    optimizer=Adam(learning_rate=1e-3), # Tasa de aprendizaje estándar
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Se implementa EarlyStopping para monitorear val_loss y prevenir el sobreajuste
early_stopper_fase1 = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Número de épocas sin mejora antes de detenerse
    verbose=1,
    restore_best_weights=True  # Restaura los pesos del modelo de la mejor época
)

# Entrenamiento de la Fase 1
history_fase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FASE_1,
    callbacks=[early_stopper_fase1],
    class_weight=class_weight_dict  # Aplicar los pesos de clase
)
print("--- FASE 1 Completada ---")

# --- 6. FASE 2: Ajuste Fino (Fine-Tuning) ---
print("--- Iniciando FASE 2: Fine-Tuning ---")

# Descongelar el 'backbone' para el ajuste fino
base_model.trainable = True

# Estrategia de Fine-Tuning: Descongelar solo el último bloque convolucional (block5)
# Esto permite que el modelo ajuste las características de alto nivel
# sin perturbar las características de bajo nivel (bordes, texturas).
set_trainable = False
for layer in base_model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print("Capas descongeladas para Fine-Tuning:")
for layer in base_model.layers:
    if layer.trainable:
        print(f"  {layer.name} (Entrenable)")

# Re-compilación del modelo para la Fase 2
# Es CRÍTICO usar una tasa de aprendizaje (learning_rate) muy baja (1e-5)
# para evitar que grandes gradientes destruyan los pesos pre-entrenados.
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary() # Notar el incremento en 'Trainable params'

# Callback de EarlyStopping para la Fase 2
# Se incrementa la paciencia a 5 para dar más tiempo al ajuste fino
early_stopper_fase2 = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Continuar el entrenamiento (Fine-Tuning)
initial_epochs_fase2 = len(history_fase1.history['loss'])
total_epochs = initial_epochs_fase2 + EPOCHS_FASE_2

history_fase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=initial_epochs_fase2, # Continuar desde la última época
    callbacks=[early_stopper_fase2],
    class_weight=class_weight_dict # Aplicar pesos de clase también en F2
)

# --- 7. Evaluación y Guardado del Modelo ---
print("Evaluando modelo final (Fine-Tuned y Ponderado) con el set de prueba...")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Accuracy en el set de prueba: {test_accuracy * 100:.2f}%")
print(f"Loss en el set de prueba: {test_loss:.4f}")

# Guardado del modelo final en el formato .keras nativo
print("Guardando el mejor modelo...")
model_save_path = 'brain_tumor_vgg16_final.keras'
model.save(model_save_path)
print(f"Entrenamiento completado y modelo guardado en '{model_save_path}'")


# ==============================================================================
# SECCIÓN DE GENERACIÓN DE REPORTES Y GRÁFICAS
# ==============================================================================

print("Generando reportes de métricas...")

# Combinar los historiales de ambas fases para una visualización completa
epochs_ran_fase1 = len(history_fase1.history['loss'])
epochs_ran_fase2 = len(history_fase2.history['loss'])
total_epochs_ran = epochs_ran_fase1 + epochs_ran_fase2
epochs_range = range(total_epochs_ran)

train_acc = history_fase1.history['accuracy'] + history_fase2.history['accuracy']
train_loss = history_fase1.history['loss'] + history_fase2.history['loss']
val_acc = history_fase1.history['val_accuracy'] + history_fase2.history['val_accuracy']
val_loss = history_fase1.history['val_loss'] + history_fase2.history['val_loss']

# --- 1. Gráficas de Historial (Precisión y Pérdida) ---
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Precisión de Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de Validación')
# Marcar el inicio del Fine-Tuning
plt.axvline(x=epochs_ran_fase1 - 1, color='red', linestyle='--', label='Inicio Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Historial de Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
# Marcar el inicio del Fine-Tuning
plt.axvline(x=epochs_ran_fase1 - 1, color='red', linestyle='--', label='Inicio Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Historial de Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Loss)')

# Guardar la gráfica
plot_save_path = 'training_metrics_plot_final.png'
plt.savefig(plot_save_path)
print(f"Gráfica de entrenamiento guardada en '{plot_save_path}'")

# --- 2. Obtención de Predicciones ---
# Se obtienen las predicciones del modelo sobre el conjunto de prueba
# para generar la matriz de confusión y el reporte de clasificación.
y_true = []
for images, labels in test_ds:
    y_true.extend(labels)
y_true = np.array(y_true)

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1) # Convertir probabilidades a la clase predicha

# --- 3. Matriz de Confusión ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión (Set de Prueba)')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')

# Guardar la matriz
cm_save_path = 'confusion_matrix_final.png'
plt.savefig(cm_save_path)
print(f"Matriz de confusión guardada en '{cm_save_path}'")

# --- 4. Reporte de Clasificación (Archivo de Texto) ---
# Se genera un reporte detallado con Precision, Recall, y F1-Score por clase.
report_text = classification_report(y_true, y_pred, target_names=class_names)

report_title = "Reporte de Clasificación del Modelo VGG16 (Fine-Tuned y Ponderado)\n"
divider = "=" * 70 + "\n"
accuracy_summary = f"Precisión Global en el Set de Prueba (Accuracy): {test_accuracy * 100:.2f}%\n"
loss_summary = f"Pérdida Global en el Set de Prueba (Loss): {test_loss:.4f}\n\n"
full_report = report_title + divider + accuracy_summary + loss_summary + report_text

# Guardar el reporte en un archivo .txt
report_save_path = 'classification_report_final.txt'
with open(report_save_path, 'w') as f:
    f.write(full_report)

print(f"Reporte de clasificación guardado en '{report_save_path}'")
print("\n--- Reporte Final ---")
print(full_report)
print("¡Proceso completado!")
