"""Preprocessing module for AI4I2020 Predictive Maintenance dataset."""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.data import get_feature_info


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the preprocessing pipeline for features.

    Numeric features: impute median, then scale.
    Categorical features: impute most frequent, then one-hot encode.

    Returns:
        ColumnTransformer ready for use in a sklearn Pipeline.
    """
    info = get_feature_info()
    numeric_features = info["numeric_features"]
    categorical_features = info["categorical_features"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> List[str]:
    """Get feature names after preprocessing transformation.

    Args:
        preprocessor: Fitted ColumnTransformer.

    Returns:
        List of feature names.
    """
    return preprocessor.get_feature_names_out().tolist()