from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = [float(request.form[i]) for i in request.form]
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)[0]
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)