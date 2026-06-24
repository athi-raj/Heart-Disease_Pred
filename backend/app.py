from flask import Flask,request,jsonify
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model(
    "heart_disease_model.keras"
)

@app.route("/")
def home():
    return "HeartSense AI Backend Running"

@app.route("/predict",methods=["POST"])
def predict():
    data = request.json

    # preprocessing

    prediction = model.predict(...)

    return jsonify({
        "prediction":"Moderate Risk",
        "confidence":"82%"
    })

if __name__=="__main__":
    app.run()
