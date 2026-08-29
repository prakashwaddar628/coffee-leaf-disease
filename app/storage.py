import json
from pathlib import Path
from typing import List, Dict
from app.config import PREDICTIONS_FILE
from app.schemas import PredictionResult, HistoryItem

class Storage:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            self._save([])

    def _load(self) -> List[Dict]:
        with open(self.file_path, "r") as f:
            return json.load(f)

    def _save(self, data: List[Dict]):
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)

    def save_prediction(self, prediction: PredictionResult):
        data = self._load()
        data.insert(0, prediction.model_dump()) # Prepend new prediction
        self._save(data)

    def get_prediction(self, prediction_id: str) -> PredictionResult | None:
        data = self._load()
        for item in data:
            if item["id"] == prediction_id:
                return PredictionResult(**item)
        return None

    def get_history(self) -> List[HistoryItem]:
        data = self._load()
        history = []
        for item in data:
            history.append(HistoryItem(**item))
        return history

storage = Storage(PREDICTIONS_FILE)
