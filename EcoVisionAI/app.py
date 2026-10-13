"""EcoVision AI Flask application."""
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from config import Config
from database import init_db, save_prediction, fetch_history, dashboard_stats
from inference import ModelNotReadyError, predict_image, model_status

app = Flask(__name__)
app.config.from_object(Config)
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
Path(app.config["INSTANCE_PATH"]).mkdir(parents=True, exist_ok=True)
init_db(app.config["DATABASE_PATH"])

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
def home():
    return render_template("index.html", model=model_status())

@app.get("/history")
def history_page():
    return render_template("history.html")

@app.get("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@app.get("/about")
def about_page():
    return render_template("about.html", model=model_status())

@app.get("/api/health")
def health():
    status = model_status()
    return jsonify({"status": "ok", "model": status}), 200

@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please choose an image file."}), 400
    image = request.files["image"]
    if not image.filename:
        return jsonify({"error": "The selected file has no name."}), 400
    if not allowed_file(image.filename):
        return jsonify({"error": "Unsupported image type. Use JPG, JPEG, PNG, or WEBP."}), 400
    image_bytes = image.read()
    if not image_bytes:
        return jsonify({"error": "The uploaded file is empty."}), 400
    if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        return jsonify({"error": "The image is too large. Maximum size is 10 MB."}), 413
    try:
        result = predict_image(image_bytes, app.config["MODEL_PATH"], app.config["LABELS_PATH"])
    except ModelNotReadyError as exc:
        return jsonify({"error": str(exc), "code": "model_not_ready"}), 503
    except (ValueError, OSError) as exc:
        return jsonify({"error": f"Could not read this image: {exc}"}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The model could not analyse this image. Check the server logs."}), 500
    record_id = save_prediction(app.config["DATABASE_PATH"], result["category"], result["confidence"])
    result["id"] = record_id
    return jsonify(result)

@app.get("/api/history")
def history_api():
    limit = min(max(request.args.get("limit", default=50, type=int), 1), 200)
    return jsonify({"items": fetch_history(app.config["DATABASE_PATH"], limit)})

@app.get("/api/dashboard")
def dashboard_api():
    return jsonify(dashboard_stats(app.config["DATABASE_PATH"]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
