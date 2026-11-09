"""Synthetic student dataset generator.

A hidden "ability" factor drives GPA, study habits, attendance and so on, which
makes the features realistically correlated. The final score is then built from
those features plus lifestyle / support adjustments and random noise.

To use real data instead, drop a CSV with the same columns into data/students.csv
(see README for the column list).
"""
import numpy as np
import pandas as pd

from src import config


def score_to_level(score: float) -> str:
    if score >= config.HIGH_THRESHOLD:
        return "High"
    if score >= config.AT_RISK_THRESHOLD:
        return "Average"
    return "At-Risk"


def generate_students(n: int = 1500, seed: int = config.RANDOM_STATE,
                      missing_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(17, 25, n)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.48, 0.49, 0.03])
    parental_education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"], n, p=[0.40, 0.35, 0.18, 0.07])
    family_support = rng.choice(["Low", "Medium", "High"], n, p=[0.20, 0.45, 0.35])
    internet_access = rng.choice(["Yes", "No"], n, p=[0.88, 0.12])

    ability = rng.normal(0, 1, n)  # latent factor

    previous_gpa = np.clip(3.0 + 0.6 * ability + rng.normal(0, 0.15, n), 1.0, 4.0)
    study_hours = np.clip(14 + 5 * ability + rng.normal(0, 5, n), 1, 40)
    attendance = np.clip(78 + 10 * ability + rng.normal(0, 8, n), 35, 100)
    assignment = np.clip(75 + 13 * ability + rng.normal(0, 9, n), 20, 100)
    internal = np.clip(previous_gpa / 4 * 100 * 0.85 + 0.1 * assignment
                       + rng.normal(0, 6, n), 15, 100)
    participation = np.clip(np.round(6 + 1.2 * ability + rng.normal(0, 1.5, n)), 1, 10)
    sleep = np.clip(rng.normal(6.8, 1.1, n), 4, 10)
    stress = np.clip(np.round(5 - 0.5 * ability + rng.normal(0, 2, n)), 1, 10)
    extracurricular = rng.integers(0, 4, n)
    absences = np.round((100 - attendance) * 0.4 + rng.poisson(1, n)).astype(int)

    base = (0.30 * previous_gpa / 4 * 100
            + 0.20 * attendance
            + 0.15 * assignment
            + 0.15 * internal
            + 0.10 * np.minimum(study_hours, 30) / 30 * 100
            + 0.10 * participation * 10)
    adjust = (-1.0 * (stress - 5)
              + np.where((sleep >= 6) & (sleep <= 8.5), 2, -2)
              + np.select([family_support == "High", family_support == "Low"], [3, -3], 0)
              + np.where(internet_access == "No", -3, 0))
    final_score = np.clip(base + adjust + rng.normal(0, 3.5, n), 0, 100)

    df = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "age": age,
        "gender": gender,
        "parental_education": parental_education,
        "family_support": family_support,
        "internet_access": internet_access,
        "study_hours_per_week": study_hours.round(1),
        "attendance_pct": attendance.round(1),
        "previous_gpa": previous_gpa.round(2),
        "assignment_completion_pct": assignment.round(1),
        "internal_marks": internal.round(1),
        "participation_score": participation.astype(int),
        "sleep_hours": sleep.round(1),
        "stress_level": stress.astype(int),
        "extracurricular_activities": extracurricular,
        "absences": absences,
        "final_score": final_score.round(1),
    })
    df["performance_level"] = df["final_score"].apply(score_to_level)

    # Inject a few missing values so the imputation step is exercised.
    if missing_rate > 0:
        for col in ["attendance_pct", "sleep_hours", "family_support"]:
            mask = rng.random(n) < missing_rate
            df.loc[mask, col] = np.nan
    return df


def save_dataset(path=config.DATA_FILE, **kwargs) -> pd.DataFrame:
    df = generate_students(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
