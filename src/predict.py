"""Prediction module for inference on new data."""

from pathlib import Path
from typing import Dict, Union
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).parents[1] / "models" / "final_model.joblib"
FEATURE_NAMES_PATH = Path(__file__).parents[1] / "results" / "feature_names.json"
THRESHOLD_PATH = Path(__file__).parents[1] / "results" / "test_metrics.json"


class Predictor:
    """Wrapper for loading model and making predictions."""

    def __init__(self, model_path: Union[Path, str] = MODEL_PATH):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Load the trained pipeline."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run training first.")
        self.pipeline = joblib.load(self.model_path)

        # Try to load feature names
        if FEATURE_NAMES_PATH.exists():
            import json
            with open(FEATURE_NAMES_PATH) as f:
                self.feature_names = json.load(f)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict failure probabilities.

        Args:
            X: DataFrame with features (same columns as training data).

        Returns:
            Array of probabilities for positive class (failure).
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict failure labels.

        Args:
            X: DataFrame with features.
            threshold: Probability threshold for classification.

        Returns:
            Array of predicted labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def predict_single(
        self,
        machine_type: str,
        air_temp: float,
        process_temp: float,
        rotational_speed: int,
        torque: float,
        tool_wear: int,
        threshold: float = 0.5,
    ) -> Dict:
        """Make prediction for a single machine.

        Args:
            machine_type: One of 'L', 'M', 'H'.
            air_temp: Air temperature [K].
            process_temp: Process temperature [K].
            rotational_speed: Rotational speed [rpm].
            torque: Torque [Nm].
            tool_wear: Tool wear [min].
            threshold: Classification threshold.

        Returns:
            Dictionary with prediction results.
        """
        input_df = pd.DataFrame(
            {
                "Type": [machine_type],
                "Air temperature [K]": [air_temp],
                "Process temperature [K]": [process_temp],
                "Rotational speed [rpm]": [rotational_speed],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear],
            }
        )

        prob = self.predict_proba(input_df)[0]
        pred = int(prob >= threshold)

        if prob < 0.3:
            risk = "Low Risk"
        elif prob < 0.7:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        return {
            "prediction": "Potential Failure" if pred == 1 else "Normal",
            "failure_probability": float(prob),
            "threshold": threshold,
            "risk_category": risk,
            "prediction_label": pred,
        }

    def predict_batch(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Make predictions for a batch of machines.

        Args:
            df: DataFrame with required feature columns.
            threshold: Classification threshold.

        Returns:
            DataFrame with original columns plus predictions.
        """
        required_cols = [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = df[required_cols].copy()
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)

        result = df.copy()
        result["failure_probability"] = probs
        result["predicted_failure"] = preds
        result["prediction"] = result["predicted_failure"].map({0: "Normal", 1: "Potential Failure"})

        def risk_cat(p):
            if p < 0.3:
                return "Low Risk"
            elif p < 0.7:
                return "Medium Risk"
            return "High Risk"

        result["risk_category"] = result["failure_probability"].apply(risk_cat)

        return result


def get_default_threshold() -> float:
    """Get the recommended threshold from training results."""
    if THRESHOLD_PATH.exists():
        import json
        with open(THRESHOLD_PATH) as f:
            metrics = json.load(f)
        # We'll use the F1-optimal threshold from training
        # For now return a sensible default
        return 0.5
    return 0.5


if __name__ == "__main__":
    # Quick test
    predictor = Predictor()
    result = predictor.predict_single("M", 300.0, 310.0, 1500, 40.0, 100)
    print(result)