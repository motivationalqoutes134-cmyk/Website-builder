"""Inference utilities. TensorFlow is imported lazily so the UI can run before training."""
import io, json
from pathlib import Path
from PIL import Image
import numpy as np

class ModelNotReadyError(RuntimeError):
    pass

_model_cache = {}

def model_status(model_path=None, labels_path=None):
    model_path = model_path or str(Path(__file__).parent / "models" / "ecovision_mobilenetv2.keras")
    labels_path = labels_path or str(Path(__file__).parent / "models" / "labels.json")
    ready = Path(model_path).exists() and Path(labels_path).exists()
    return {"ready": ready, "message": "Trained model ready." if ready else "Model not trained yet. Run train.py before analysing images."}

def _load(model_path, labels_path):
    key = (model_path, labels_path)
    if key in _model_cache:
        return _model_cache[key]
    if not Path(model_path).exists() or not Path(labels_path).exists():
        raise ModelNotReadyError("No trained model is available. Follow README.md to download TrashNet and run the training command.")
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise ModelNotReadyError("TensorFlow is not installed. Install requirements.txt and train the model first.") from exc
    with open(labels_path, encoding="utf-8") as fh:
        labels = json.load(fh)
    model = load_model(model_path)
    _model_cache[key] = (model, labels)
    return model, labels

def predict_image(image_bytes, model_path, labels_path, threshold=0.55):
    model, labels = _load(model_path, labels_path)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    category = labels[str(index)] if isinstance(labels, dict) else labels[index]
    uncertain = confidence < threshold
    return {"category": category if not uncertain else "Uncertain", "raw_category": category, "confidence": round(confidence, 4), "uncertain": uncertain, "advice": disposal_advice(category) if not uncertain else "The model is not confident enough. Try a clearer, well-lit image with one item in view."}

def disposal_advice(category):
    advice = {"cardboard": "Flatten clean cardboard and place it in the paper/cardboard recycling stream.", "glass": "Rinse glass and place it in a glass recycling bin; do not include ceramics or bulbs.", "metal": "Empty and rinse metal cans, then use the local metal recycling stream.", "paper": "Keep paper clean and dry, then place it with paper recycling.", "plastic": "Rinse the container and check your local recycling symbol and collection rules.", "trash": "Place contaminated or non-recyclable material in general waste; reduce and reuse where possible."}
    return advice.get(category.lower(), "Check your local council guidance for this material.")
