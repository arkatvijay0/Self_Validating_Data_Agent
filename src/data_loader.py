import pandas as pd
import numpy as np
from scipy import stats


def infer_role(series):
    """
    Infer the analytical role of a column.
    """

    # Check datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "time_index"

    # Check numeric
    if pd.api.types.is_numeric_dtype(series):
        return "continuous"

    # Avoid division by zero
    if len(series) == 0:
        return "unknown"

    # Check categorical
    unique_ratio = series.nunique(dropna=True) / len(series)

    if unique_ratio < 0.05:
        return "categorical"

    return "text_or_id"


def profile_dataset(filepath):
    """
    Load a CSV file and generate a dataset profile.
    """

    df = pd.read_csv(filepath)

    # Try to detect datetime columns
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            if converted.notna().mean() > 0.8:
                df[col] = converted

    profile = {
        "file": filepath,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {}
    }

    for col in df.columns:

        series = df[col]

        col_info = {
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_pct": round(
                series.isnull().mean() * 100,
                2
            ),
            "unique_count": int(series.nunique()),
            "role": infer_role(series)
        }

        # Numeric information
        if pd.api.types.is_numeric_dtype(series):

            clean = series.dropna()

            if len(clean) > 0:

                col_info.update({
                    "mean": round(float(clean.mean()), 2),
                    "median": round(float(clean.median()), 2),
                    "std": round(float(clean.std()), 2),
                    "min": round(float(clean.min()), 2),
                    "max": round(float(clean.max()), 2)
                })

                # Calculate skewness only if enough values exist
                if len(clean) >= 3:
                    col_info["skew"] = round(
                        float(stats.skew(clean)),
                        2
                    )

        # Categorical information
        elif col_info["role"] == "categorical":

            col_info["top_values"] = (
                series.value_counts()
                .head(10)
                .to_dict()
            )

        profile["columns"][col] = col_info

    return profile


if __name__ == "__main__":

    import json

    filepath = "data/raw/sales.csv"

    result = profile_dataset(filepath)

    print(
        json.dumps(
            result,
            indent=4,
            default=str
        )
    )