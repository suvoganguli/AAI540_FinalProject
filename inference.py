import pandas as pd
import pickle
import os
from io import StringIO

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        return pd.read_csv(StringIO(request_body), header=None)
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    if content_type == "text/csv":
        return "\n".join(str(x) for x in prediction)
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))
