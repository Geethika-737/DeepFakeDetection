from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("deepfake_cnn_model.keras")

IMG_SIZE = (128, 128)

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/prediction")
def prediction():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", prediction="No image uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", prediction="Please choose an image")

    image = Image.open(file).convert("RGB")
    processed = preprocess_image(image)

    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        result = "🟢 REAL IMAGE"
    else:
        result = "🔴 FAKE IMAGE"

    confidence = round(max(prediction, 1 - prediction) * 100, 2)

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)