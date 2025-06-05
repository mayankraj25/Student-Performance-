from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [
            float(request.form['feature1']),  # Hours Studied
            float(request.form['feature2']),  # Previous Score
            int(request.form['feature3']),    # Extracurricular
            float(request.form['feature4']),  # Sleep Hours
            float(request.form['feature5'])   # Sample Papers
        ]

        # Scale inputs before prediction
        scaled_inputs = scaler.transform([inputs])
        prediction = model.predict(scaled_inputs)

        if prediction[0] > 100:
            prediction_text = "You can touch 100%"
        else:
            prediction_text = f'You can score : {prediction[0]:.2f}%'
        return render_template('index.html', prediction_text=prediction_text)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)