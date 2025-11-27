# app.py (updated)
import os
import traceback
import json
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from werkzeug.utils import secure_filename
from feature_extraction import image_file_to_feature_vector

# ----------------- CONFIG -----------------
MODEL_CANDIDATES = [
    "model.pkl",
    "brain_tumor_rf_model.pkl",
    "model.pkcls",
    "brain_tumor_rf_model.pkcls",
    "brain_tumor_rf_model.pkl"  # try the original name too
]
SCALER_PATH = "scaler.pkl"
FEATURE_ORDER_PATH = "feature_order.json"
TMP_DIR = "tmp_upload"
LOG_PATH = os.path.join(TMP_DIR, "predict_errors.log")
# ------------------------------------------
CORS(app, origins=["https://wbgit01.github.io"])

app = Flask(__name__, static_folder="static", static_url_path="/")

def find_and_load_model():
    """Find a model file from MODEL_CANDIDATES and load it with joblib."""
    for fname in MODEL_CANDIDATES:
        if os.path.exists(fname):
            try:
                model = joblib.load(fname)
                return model, fname
            except Exception as e:
                # continue trying other candidates but log the issue
                with open(LOG_PATH, "a", encoding="utf-8") as lf:
                    lf.write(f"\n\nFailed to load candidate {fname}:\n")
                    lf.write(traceback.format_exc())
    return None, None

# Ensure tmp dir exists for logs / uploads
os.makedirs(TMP_DIR, exist_ok=True)

# Load scaler
if not os.path.exists(SCALER_PATH):
    print(f"[WARNING] Scaler file not found at {SCALER_PATH}. Make sure you ran compute_and_save_scaler.py")
    scaler = None
else:
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        print("[ERROR] Failed to load scaler.pkl. See log.")
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write("\n\nFailed to load scaler.pkl:\n")
            lf.write(traceback.format_exc())
        scaler = None

# Load feature order
if not os.path.exists(FEATURE_ORDER_PATH):
    print(f"[WARNING] Feature order file not found at {FEATURE_ORDER_PATH}.")
    feature_order = None
else:
    try:
        with open(FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
            feature_order = json.load(f)
    except Exception:
        feature_order = None
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write("\n\nFailed to load feature_order.json:\n")
            lf.write(traceback.format_exc())

# Load model (tries several candidate names)
model, model_used = find_and_load_model()
if model is None:
    print("[ERROR] No model could be loaded. Place model.pkl or brain_tumor_rf_model.pkl in the project folder.")
else:
    print(f"[INFO] Loaded model file: {model_used}")

# Helper to check predict_proba availability
def model_supports_predict_proba(m):
    return callable(getattr(m, "predict_proba", None))

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/info", methods=["GET"])
def info():
    """Return useful info about model and scaler shapes to help debugging."""
    info = {}
    info["model_file"] = model_used if model is not None else None
    try:
        info["model_has_predict_proba"] = model_supports_predict_proba(model) if model is not None else False
    except Exception:
        info["model_has_predict_proba"] = False
    try:
        info["scaler_loaded"] = scaler is not None
        if scaler is not None:
            # scaler.mean_ shape may exist
            info["scaler_mean_shape"] = list(getattr(scaler, "mean_", []).shape)
            info["scaler_var_shape"] = list(getattr(scaler, "var_", []).shape)
    except Exception:
        info["scaler_mean_shape"] = None
        info["scaler_var_shape"] = None

    info["feature_order_len"] = len(feature_order) if feature_order is not None else None
    # model expected feature count (if available)
    try:
        info["model_n_features_in"] = int(getattr(model, "n_features_in_", -1)) if model is not None else None
    except Exception:
        info["model_n_features_in"] = None

    return jsonify(info)

@app.route("/predict", methods=["POST"])
def predict():
    # Ensure JSON response always returned (even on errors)
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    f = request.files["file"]
    filename = secure_filename(f.filename or "upload")
    tmp_path = os.path.join(TMP_DIR, filename)
    try:
        # Save the uploaded file temporarily
        f.save(tmp_path)

        # Basic checks
        if model is None:
            return jsonify({"error": "model_not_loaded", "message": "Model file not found or failed to load on server."}), 500
        if scaler is None:
            return jsonify({"error": "scaler_not_loaded", "message": "Scaler file not found. Run compute_and_save_scaler.py"}), 500

        # Extract features from the uploaded file
        feats = image_file_to_feature_vector(tmp_path)
        if feats is None:
            return jsonify({"error": "feature_extraction_failed", "message": "Feature extractor returned None"}), 500

        X = np.asarray(feats).reshape(1, -1)

        # Sanity: check expected number of features
        expected = None
        if feature_order is not None:
            expected = len(feature_order)
        model_n = None
        try:
            model_n = int(getattr(model, "n_features_in_", -1))
        except Exception:
            model_n = None

        if expected is not None and X.shape[1] != expected:
            return jsonify({
                "error": "feature_length_mismatch",
                "message": f"Extracted feature length {X.shape[1]} does not match feature_order length {expected}."
            }), 500

        if model_n is not None and model_n != -1 and X.shape[1] != model_n:
            return jsonify({
                "error": "model_feature_mismatch",
                "message": f"Extracted feature length {X.shape[1]} does not match model.n_features_in_ {model_n}."
            }), 500

        # Apply scaler
        try:
            Xs = scaler.transform(X)
        except Exception as e:
            # log full traceback
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write("\n\nScaler transform failed:\n")
                lf.write(traceback.format_exc())
            return jsonify({"error": "scaler_transform_failed", "message": str(e)}), 500

        # Ensure model supports predict_proba
        if not model_supports_predict_proba(model):
            # log and return informative error
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write("\n\nModel lacks predict_proba. Model type: {}\n".format(type(model)))
            return jsonify({
                "error": "model_no_predict_proba",
                "message": "Loaded model does not support predict_proba(). Save a scikit-learn RandomForest or LogisticRegression model as model.pkl."
            }), 500

        # Predict
        try:
            proba = float(model.predict_proba(Xs)[0, 1])
            label = int(model.predict(Xs)[0])
        except Exception as e:
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write("\n\nPrediction error:\n")
                lf.write(traceback.format_exc())
            return jsonify({"error": "prediction_failed", "message": str(e)}), 500

        return jsonify({"probability": proba, "label": label}), 200

    except Exception as e:
        # Log exception with traceback so you can inspect tmp_upload/predict_errors.log
        tb = traceback.format_exc()
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write("\n\n--- NEW /predict EXCEPTION ---\n")
            lf.write(tb)
        print(tb)
        return jsonify({"error": "internal_server_error", "message": str(e)}), 500

    finally:
        # Attempt to remove the temporary upload file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    # Run app
    app.run(debug=True, host="0.0.0.0", port=5000)
