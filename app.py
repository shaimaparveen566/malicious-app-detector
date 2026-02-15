from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and features
model = joblib.load("model/decision_tree_model.pkl")
top_features = joblib.load("model/top_features.pkl")
le = joblib.load("model/label_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html', features=top_features)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []

    input_data = []

    for feature in top_features:
        value = request.form.get(feature)
        if value is None:
            input_data.append(0)
        else:
            input_data.append(1)


    prediction = model.predict([input_data])[0]
    result = le.inverse_transform([prediction])[0]

    return render_template('index.html',
                           features=top_features,
                           prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
