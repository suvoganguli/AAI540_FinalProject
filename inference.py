import joblib
import os
import pandas as pd

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    return joblib.load(model_path)

def input_fn(request_body, content_type):
    if content_type == "text/csv":
        return pd.read_csv(pd.compat.StringIO(request_body))
    raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    if accept == "text/csv":
        return ",".join(str(x) for x in prediction), "text/csv"
    raise ValueError("Unsupported accept type: {}".format(accept))
