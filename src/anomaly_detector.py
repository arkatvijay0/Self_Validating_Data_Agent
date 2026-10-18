
"""
Isolation Forest anomaly detection for the Self-Validating Data Agent.

This module is intentionally independent of the main pipeline so it can
be tested before integration.

An anomaly is an unusual observation, not automatically an error or fraud.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    contamination: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """
    Detect unusual rows using Isolation Forest.

    Parameters
    ----------
    df:
        Input dataset.
    feature_columns:
        Numeric columns to analyze. If omitted, numeric columns are selected
        automatically. Identifier columns should be excluded when possible.
    contamination:
        Expected proportion of anomalies, between 0 and 0.5 (exclusive).
        This is a model setting, not a measured anomaly rate.
    random_state:
        Seed for reproducible results.
    n_estimators:
        Number of trees in the Isolation Forest.

    Returns
    -------
    Dictionary with status, summary, feature information, and a DataFrame
    containing original rows plus anomaly labels and scores.
    """

    if not isinstance(df, pd.DataFrame):
        return {
            "status": "FAIL",
            "reason": "Input must be a pandas DataFrame.",
        }

    if df.empty:
        return {
            "status": "FAIL",
            "reason": "Input dataset has no rows.",
        }

    if not 0 < contamination < 0.5:
        return {
            "status": "FAIL",
            "reason": "contamination must be greater than 0 and less than 0.5.",
        }

    if n_estimators < 1:
        return {
            "status": "FAIL",
            "reason": "n_estimators must be at least 1.",
        }

    if feature_columns is None:
        candidate_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()
    else:
        candidate_columns = list(feature_columns)

    missing_columns = [
        column for column in candidate_columns if column not in df.columns
    ]
    if missing_columns:
        return {
            "status": "FAIL",
            "reason": f"Requested feature columns not found: {missing_columns}",
        }

    if not candidate_columns:
        return {
            "status": "FAIL",
            "reason": "No numeric feature columns were selected.",
        }

    # Convert selected features to numeric values. Non-numeric values become
    # missing values and are handled by the imputer below.
    numeric_features = pd.DataFrame(index=df.index)

    for column in candidate_columns:
        numeric_features[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    numeric_features = numeric_features.replace(
        [np.inf, -np.inf], np.nan
    )

    # A column containing no finite values cannot contribute to detection.
    usable_columns = [
        column for column in numeric_features.columns
        if numeric_features[column].notna().any()
    ]

    if not usable_columns:
        return {
            "status": "FAIL",
            "reason": "Selected features contain no usable numeric values.",
        }

    numeric_features = numeric_features[usable_columns]

    # Isolation Forest requires at least two observations for this task.
    if len(numeric_features) < 2:
        return {
            "status": "FAIL",
            "reason": "At least two rows are required for anomaly detection.",
        }

    # Impute missing values, standardize features, and fit the model.
    # Standardization improves interpretability of preprocessing but is not
    # strictly required by tree-based Isolation Forest.
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    try:
        pipeline.fit(numeric_features)

        model = pipeline.named_steps["model"]
        transformed_features = pipeline[:-1].transform(numeric_features)

        predictions = model.predict(transformed_features)
        decision_scores = model.decision_function(transformed_features)

    except Exception as error:
        return {
            "status": "FAIL",
            "reason": f"Isolation Forest failed: {type(error).__name__}: {error}",
        }

    # scikit-learn returns -1 for anomalies and +1 for inliers.
    anomaly_flags = predictions == -1

    results = df.copy()
    results["anomaly_label"] = np.where(
        anomaly_flags, "anomaly", "normal"
    )
    results["anomaly_score"] = decision_scores.astype(float)

    anomaly_count = int(anomaly_flags.sum())
    total_rows = int(len(df))

    summary = {
        "algorithm": "Isolation Forest",
        "status": "PASS",
        "total_rows": total_rows,
        "anomaly_count": anomaly_count,
        "normal_count": int(total_rows - anomaly_count),
        "detected_anomaly_percentage": float(
            anomaly_count / total_rows * 100
        ),
        "contamination_setting": float(contamination),
        "feature_columns": usable_columns,
        "excluded_all_missing_columns": [
            column for column in candidate_columns
            if column not in usable_columns
        ],
        "missing_cells_imputed": int(
            numeric_features.isna().sum().sum()
        ),
        "random_state": int(random_state),
        "n_estimators": int(n_estimators),
        "score_interpretation": (
            "Lower decision_function scores indicate more anomalous rows."
        ),
        "evidence_note": (
            "Anomaly labels are model-generated flags, not proof of errors "
            "or fraud. Review flagged records in context."
        ),
    }

    # Return the most anomalous records first.
    anomaly_rows = (
        results.loc[results["anomaly_label"] == "anomaly"]
        .sort_values("anomaly_score", ascending=True)
        .copy()
    )

    return {
        "status": "PASS",
        "summary": summary,
        "results": results,
        "anomalies": anomaly_rows,
    }


if __name__ == "__main__":
    # Small reproducible demonstration.
    sample = pd.DataFrame(
        {
            "transaction_amount": [
                95, 102, 98, 100, 97, 103, 99, 101, 96, 5000
            ],
            "transaction_count": [
                2, 3, 2, 2, 3, 2, 2, 3, 2, 80
            ],
        }
    )

    output = detect_anomalies(
        sample,
        feature_columns=["transaction_amount", "transaction_count"],
        contamination=0.10,
        random_state=42,
    )

    if output["status"] != "PASS":
        raise SystemExit(output["reason"])

    print("Isolation Forest demonstration")
    print("-" * 40)
    print(output["summary"])
    print("\nFlagged rows:")
    print(
        output["anomalies"][
            [
                "transaction_amount",
                "transaction_count",
                "anomaly_label",
                "anomaly_score",
            ]
        ].to_string(index=False)
    )
