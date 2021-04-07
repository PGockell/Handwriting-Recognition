from model import model
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/test', methods=['POST'])
def predict():
    data = request.get_json()
    image = data['image']
    prediction = model.predict(image)
    prediction = str(prediction).strip('[]')
    print("Prediction", prediction)
    return jsonify(prediction)
