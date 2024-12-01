import os
import sys
from bagger import BaggingBootstrapper , BaggingBootstrapper
from flask import Flask, render_template, jsonify
import pickle
import joblib
import __main__


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone ,TransformerMixin
from arch.bootstrap import MovingBlockBootstrap
setattr(__main__, "BaggingBootstrapper", BaggingBootstrapper)



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
        message = f"Error loading model: {e}_{__name__}"
    return render_template('index.html', message=message)

# Home route to display the button and message
@app.route('/')
def home():
    return render_template('index.html', message="Click the button to load the model.")


if __name__ == '__main__':
    app.run(debug=True)
