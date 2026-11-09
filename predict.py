"""Inference helpers: single-student and batch predictions."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.preprocessing import validate_columns
from src.recommend import generate_recommendations


class StudentPredictor:
    def __init__(self, models_dir=config.MODELS_DIR):
        models_dir = Path(models_dir)
        clf_path = models_dir / "classifier.joblib"
        reg_path = models_dir / "regressor.joblib"
        if not clf_path.exists() or not reg_path.exists():
            raise FileNotFoundError(
                f"Trained models not found in {models_dir}. "
                "Run `python main.py train` first.")
        self.classifier = joblib.load(clf_path)
        self.regressor = joblib.load(reg_path)

    def predict(self, student: dict) -> dict:
        """Predict for one student given a dict of feature values."""
        X = pd.DataFrame([student]).reindex(columns=config.FEATURES)
        score = float(np.clip(self.regressor.predict(X)[0], 0, 100))
        level = str(self.classifier.predict(X)[0])
        proba = self.classifier.predict_proba(X)[0]
        probabilities = {str(c): round(float(p), 4)
                         for c, p in zip(self.classifier.classes_, proba)}
        return {
            "predicted_score": round(score, 1),
            "performance_level": level,
            "probabilities": probabilities,
            "at_risk_probability": probabilities.get("At-Risk", 0.0),
            "risk_flag": level == "At-Risk",
            "recommendations": generate_recommendations(student, level),
        }

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for many students; returns the input plus prediction columns."""
        validate_columns(df, require_targets=False)
        X = df[config.FEATURES]
        out = df.copy()
        out["predicted_score"] = np.clip(self.regressor.predict(X), 0, 100).round(1)
        out["predicted_level"] = self.classifier.predict(X)
        proba = self.classifier.predict_proba(X)
        for i, cls in enumerate(self.classifier.classes_):
            out[f"prob_{cls}"] = proba[:, i].round(4)
        return out


# A ready-made example used by the CLI demo.
DEMO_STUDENT = {
    "age": 20, "gender": "Female", "parental_education": "Bachelor",
    "family_support": "Medium", "internet_access": "Yes",
    "study_hours_per_week": 8, "attendance_pct": 68, "previous_gpa": 2.4,
    "assignment_completion_pct": 62, "internal_marks": 55,
    "participation_score": 3, "sleep_hours": 5.5, "stress_level": 8,
    "extracurricular_activities": 1, "absences": 14,
}
