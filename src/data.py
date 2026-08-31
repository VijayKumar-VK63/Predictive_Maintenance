"""Data loading module for AI4I2020 Predictive Maintenance dataset."""

from pathlib import Path
from typing import Tuple
import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).parents[1] / "data" / "ai4i2020.csv"


def load_data(path: str = None) -> pd.DataFrame:
    """Load the AI4I2020 dataset from CSV.

    Args:
        path: Path to the CSV file. Defaults to data/ai4i2020.csv.

    Returns:
        DataFrame with the raw dataset.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    if path is None:
        path = str(DEFAULT_DATA_PATH)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_csv(path)


def get_feature_target(
    df: pd.DataFrame,
    exclude_failure_modes: bool = True,
    exclude_identifiers: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from the dataset.

    Args:
        df: Raw dataset.
        exclude_failure_modes: If True, exclude TWF, HDF, PWF, OSF, RNF columns
            as they represent failure-mode information only known at/after failure.
        exclude_identifiers: If True, exclude UDI and Product ID columns.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    df = df.copy()

    target = df["Machine failure"].copy()

    drop_cols = ["Machine failure"]
    if exclude_failure_modes:
        drop_cols += ["TWF", "HDF", "PWF", "OSF", "RNF"]
    if exclude_identifiers:
        drop_cols += ["UDI", "Product ID"]

    features = df.drop(columns=drop_cols)
    return features, target


def get_feature_info() -> dict:
    """Return metadata about features for documentation/UI."""
    return {
        "numeric_features": [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
        "categorical_features": ["Type"],
        "target": "Machine failure",
        "excluded_failure_modes": ["TWF", "HDF", "PWF", "OSF", "RNF"],
        "excluded_identifiers": ["UDI", "Product ID"],
    }