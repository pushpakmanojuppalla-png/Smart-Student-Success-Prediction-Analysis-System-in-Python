"""Loading, validation, splitting and the scikit-learn preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_data(path=config.DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df)
    return df


def validate_columns(df: pd.DataFrame, require_targets: bool = True) -> None:
    required = list(config.FEATURES)
    if require_targets:
        required += [config.TARGET_SCORE, config.TARGET_LEVEL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def split_data(df: pd.DataFrame):
    """Stratified train/test split on the performance level."""
    return train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET_LEVEL],
    )
