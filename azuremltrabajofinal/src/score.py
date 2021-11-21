import json
import numpy as np
import os
import pickle
import joblib
from tensorflow import keras

def init():
    global model, scaler
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).

    scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'house-price-model-scaler', '1' , 'scaler.pkl')
    scaler = joblib.load(scaler_path)

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'house-price-model', '3' , 'my_model.h5')
    model = keras.models.load_model(model_path, compile=False)

def run(raw_data):

    data = np.array(json.loads(raw_data)['data'])
    scaled_data = scaler.transform(data)

    # Make prediction.
    y_hat = model.predict(scaled_data)

    # return the result back
    return json.dumps({"predicted_price": float(y_hat)})