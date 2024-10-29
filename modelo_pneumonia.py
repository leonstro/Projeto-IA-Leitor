# Importação das Bibliotecas
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping  # Importar EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Caminhos das pastas de imagens (substitua "path/to" pelo caminho correto)
train_dir = './imagens/train'
val_dir = './imagens/validation'
test_dir = './imagens/test'

# Aumento de dados para melhorar a generalização
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Carregar e Processar as Imagens
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Construir o Modelo com DenseNet121
model = Sequential([
    DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')  # Camada de saída para classificação binária
])

# Compilar o Modelo
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Configurar o EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitorar a perda de validação
    patience=5,          # Quantidade de epochs sem melhoria antes de parar
    restore_best_weights=True  # Restaura os pesos do modelo na melhor iteração
)

# Treinar o Modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[early_stopping]  # Adicionar o callback ao treinamento
)

# Salvar o modelo treinado
model.save('modelo_pneumonia.h5')

# Avaliar o Modelo
test_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

# Relatório de Classificação e Matriz de Confusão
print(classification_report(test_labels, predicted_classes, target_names=['Normal', 'Pneumonia']))
print(confusion_matrix(test_labels, predicted_classes))
