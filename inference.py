import pandas as pd
import joblib
import os
from io import StringIO
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_fn(model_dir):
    """
    Loads the scikit-learn model from the specified directory.
    This function is called once when the model is loaded into the endpoint container.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    logger.info(f"model_fn: Attempting to load model from: {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info("model_fn: Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"model_fn: Error loading model: {e}")
        raise

def input_fn(request_body, request_content_type):
    """
    Parses the input request body into a pandas DataFrame.
    It specifically handles 'text/csv' content type.
    Now expects a clean CSV string (str type) directly.
    """
    logger.info(f"input_fn: Received request with content type: {request_content_type}")
    # Log the raw request body to confirm it's a string and looks like CSV
    logger.info(f"input_fn: Raw request body (first 500 chars): '{request_body[:500]}' (length: {len(request_body)})")

    if request_content_type == "text/csv":
        try:
            # request_body is now already a string, so no need to decode.
            # Just strip any whitespace for robustness.
            cleaned_request_body = request_body.strip()
            logger.info(f"input_fn: Cleaned request body: '{cleaned_request_body}' (length: {len(cleaned_request_body)})")

            # Read the CSV string into a DataFrame.
            # header=None: indicates the input CSV does not have a header row.
            # engine='python': Use python engine for robustness with single lines.
            df = pd.read_csv(StringIO(cleaned_request_body), header=None, engine='python')

            logger.info(f"input_fn: DataFrame created by pd.read_csv. Shape: {df.shape}")
            logger.info(f"input_fn: DataFrame head:\n{df.head().to_string()}")
            logger.info(f"input_fn: DataFrame dtypes before numeric conversion:\n{df.dtypes.to_string()}")

            # Ensure all columns are numeric. Your model expects numbers.
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='raise')

            logger.info(f"input_fn: DataFrame dtypes after numeric conversion:\n{df.dtypes.to_string()}")
            logger.info(f"input_fn: Final DataFrame head for prediction:\n{df.head().to_string()}")

            if df.isnull().any().any():
                logger.warning("input_fn: Warning: Input DataFrame contains NaN values after numeric conversion.")

            return df
        except Exception as e:
            logger.error(f"input_fn: Error processing CSV input: {e}")
            logger.error(f"input_fn: Problematic request body that caused error: '{request_body}'")
            raise

    else:
        logger.error(f"input_fn: Unsupported content type: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Makes predictions using the loaded model.
    `input_data` is the pandas DataFrame processed by `input_fn`.
    """
    logger.info(f"predict_fn: Received input_data for prediction. Shape: {input_data.shape}")
    logger.info(f"predict_fn: Input_data dtypes:\n{input_data.dtypes.to_string()}")
    logger.info(f"predict_fn: Input_data values (first 5 rows):\n{input_data.head().to_string()}") # Added for more detail

    try:
        predictions = model.predict(input_data)
        logger.info("predict_fn: Model prediction successful.")
        # Log a snippet of predictions if they are small enough
        logger.info(f"predict_fn: Raw predictions (first 10 values): {predictions[:10] if hasattr(predictions, '__len__') else predictions}")
        return predictions
    except Exception as e:
        # CRITICAL ADDITION: Log the full traceback if model.predict fails
        logger.error(f"predict_fn: Error during model prediction: {e}", exc_info=True)
        # Log the problematic input_data to help debug model-specific issues
        logger.error(f"predict_fn: Problematic input_data head:\n{input_data.head().to_string()}")
        raise

def output_fn(prediction, content_type):
    """
    Converts the model's prediction into the desired output format.
    For 'text/csv', it converts a single prediction (or array of predictions)
    into a newline-separated string.
    """
    logger.info(f"output_fn: Received prediction for output. Content type requested: {content_type}")

    if content_type == "text/csv":
        try:
            if not isinstance(prediction, (list, pd.Series, pd.DataFrame)) and not hasattr(prediction, '__iter__'):
                prediction = [prediction]
            csv_output = pd.DataFrame(prediction).to_csv(header=False, index=False, line_terminator='')
            logger.info(f"output_fn: Outputting CSV string: '{csv_output}'")
            return csv_output
        except Exception as e:
            logger.error(f"output_fn: Error converting prediction to CSV: {e}")
            raise
    else:
        logger.error(f"output_fn: Unsupported accept type: {content_type}")
        raise ValueError(f"Unsupported accept type: {content_type}")
