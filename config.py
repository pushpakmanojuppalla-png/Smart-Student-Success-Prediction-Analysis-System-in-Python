"""Central configuration: paths, feature lists, thresholds."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_FILE = DATA_DIR / "students.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "age",
    "study_hours_per_week",
    "attendance_pct",
    "previous_gpa",
    "assignment_completion_pct",
    "internal_marks",
    "participation_score",
    "sleep_hours",
    "stress_level",
    "extracurricular_activities",
    "absences",
]
CATEGORICAL_FEATURES = [
    "gender",
    "parental_education",
    "family_support",
    "internet_access",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_SCORE = "final_score"          # regression target (0-100)
TARGET_LEVEL = "performance_level"    # classification target

AT_RISK_THRESHOLD = 55   # score < 55  -> At-Risk
HIGH_THRESHOLD = 75      # score >= 75 -> High
LEVELS = ["At-Risk", "Average", "High"]
