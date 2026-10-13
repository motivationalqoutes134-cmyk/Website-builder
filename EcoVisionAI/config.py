from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    INSTANCE_PATH = str(BASE_DIR / "instance")
    DATABASE_PATH = str(BASE_DIR / "instance" / "ecovision.db")
    UPLOAD_FOLDER = str(BASE_DIR / "uploads")
    MODEL_PATH = str(BASE_DIR / "models" / "ecovision_mobilenetv2.keras")
    LABELS_PATH = str(BASE_DIR / "models" / "labels.json")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    JSON_SORT_KEYS = False
