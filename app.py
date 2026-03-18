from flask import Flask, request, jsonify
import joblib, numpy as np

app = Flask(__name__)
model     = joblib.load('model.pkl')
label_maps = joblib.load('label_maps.pkl')

def encode_input(data):
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    encoded = []
    for feat in features:
        val = data[feat]
        encoded.append(label_maps[feat][val])
    return np.array(encoded).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_new = encode_input(data)
    pred  = model.predict(X_new)[0]
    prob  = model.predict_proba(X_new)[0].tolist()
    return jsonify({
        'play_tennis': 'Yes' if pred == 1 else 'No',
        'confidence': round(max(prob) * 100, 1),
    })

@app.route('/')
def home():
    return "Tennis Prediction API ✅"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)