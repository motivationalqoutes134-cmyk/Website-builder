import io
import tempfile
import unittest
from pathlib import Path
from app import app

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        app.config.update(TESTING=True, DATABASE_PATH=str(Path(self.tmp.name) / "test.db"), MODEL_PATH=str(Path(self.tmp.name) / "missing.keras"), LABELS_PATH=str(Path(self.tmp.name) / "labels.json"))
        from database import init_db
        init_db(app.config["DATABASE_PATH"])
        self.client = app.test_client()
    def tearDown(self): self.tmp.cleanup()
    def test_health(self): self.assertEqual(self.client.get("/api/health").status_code, 200)
    def test_missing_image_is_rejected(self): self.assertEqual(self.client.post("/api/predict").status_code, 400)
    def test_bad_extension_is_rejected(self):
        response = self.client.post("/api/predict", data={"image": (io.BytesIO(b"data"), "notes.txt")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
    def test_missing_model_is_explicit(self):
        response = self.client.post("/api/predict", data={"image": (io.BytesIO(b"not-image"), "x.jpg")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["code"], "model_not_ready")

if __name__ == "__main__": unittest.main()
