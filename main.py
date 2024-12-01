import os
import sys
from bagger import BaggingBootstrapper , NoiseEnhancer
from flask import Flask, render_template, jsonify
import pickle
import joblib

sys.modules['__main__.BaggingBootstrapper'] = BaggingBootstrapper
sys.modules['__main__.NoiseEnhancer'] = NoiseEnhancer


app = Flask(__name__)

# Global variable to store the model
model = None

# Route to load the model
@app.route('/load_model')
def load_model():
    global model
    try:
        path = os.path.join('FOREX_ML','MEAN_REVERT_EURUSD', 'long', 'MEAN_REVERT_EURUSD_1.joblib')
        model = joblib.load(path)
        message = "Model successfully loaded!"
    except Exception as e:
        message = f"Error loading model: {e}"
    return render_template('index.html', message=message)

# Home route to display the button and message
@app.route('/')
def home():
    return render_template('index.html', message="Click the button to load the model.")


if __name__ == '__main__':
    app.run(debug=True)
