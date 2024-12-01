import os
import sys
from flask import Flask, render_template, jsonify
import pickle
import joblib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone ,TransformerMixin
from arch.bootstrap import MovingBlockBootstrap

class BaggingBootstrapper(BaseEstimator , ClassifierMixin) :
    def __init__(self , estimator , n_estimators=5 , threshold=0.5 , random_state=None) :
        self.estimator=estimator
        self.threshold=threshold
        self.n_estimators=n_estimators
        self.random_state=random_state
        self.fitted_models=[]
        self.strapper=None

    def fit(self , X , y) :
        block_size=X.shape[0] // self.n_estimators
        self.strapper=MovingBlockBootstrap(block_size , X , y , seed=self.random_state)
        self.fitted_models.clear()

        for sample in self.strapper.bootstrap(self.n_estimators) :
            X_train , y_train=sample[0][0] , sample[0][1]
            clone_estimator=clone(self.estimator)
            clone_estimator.fit(X_train , y_train)
            self.fitted_models.append(clone_estimator)

        return self

    def predict_proba(self , X) :
        probabilities=[model.predict_proba(X) for model in self.fitted_models]
        return np.mean(probabilities , axis=0)

    def predict(self , X) :
        predictions=(self.predict_proba(X)[: , 1] > self.threshold).astype(int)
        return predictions


class NoiseEnhancer(BaseEstimator , TransformerMixin) :
    def __init__(self , mean=0.0 , sigma=0.0 , skip_cols=[] , random_state=None) :
        self.mu=mean
        self.sigma=sigma
        self.skip_cols=skip_cols
        self.random_state=random_state

    def fit(self , X , y=None) :
        return self

    def transform(self , X) :
        X=X.copy()
        col=[c for c in X.columns if c not in self.skip_cols]
        np.random.seed(self.random_state)
        noise=np.random.normal(self.mu , self.sigma , X[col].shape) if self.sigma else 0
        X[col]+=noise
        return X



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
