from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Carregar o modelo
MODEL_PATH = 'modelo_pneumonia.h5'
model = load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Renderiza o HTML para upload de imagem

# Endpoint para upload de imagem e classificação
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo foi enviado"}), 400
    
    file = request.files["file"]
    
    try:
        # Processar a imagem para o modelo
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Fazer a predição
        prediction = model.predict(image)
        label = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        
        return jsonify({"prediction": label, "confidence": float(prediction[0][0])}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
