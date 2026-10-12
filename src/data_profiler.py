import pandas as pd
import numpy as np
from scipy import stats
import json


def infer_role(series):
    """
    Determine the likely role of a column.
    """

    column_name = str(series.name).lower()

    # --------------------------------------------------
    # Identifier detection
    # --------------------------------------------------

    id_keywords = [
        "id",
        "identifier",
        "code",
        "number"
    ]

    if any(
        keyword in column_name
        for keyword in id_keywords
    ):
        return "identifier"

    # --------------------------------------------------
    # Already recognized as datetime
    # --------------------------------------------------

    if pd.api.types.is_datetime64_any_dtype(series):
        return "time_index"

    # --------------------------------------------------
    # Detect date-like object columns
    # --------------------------------------------------

    if series.dtype == "object":

        sample = (
            series
            .dropna()
            .astype(str)
            .head(20)
        )

        if len(sample) > 0:

            converted = pd.to_datetime(
                sample,
                errors="coerce"
            )

            if converted.notna().mean() >= 0.8:
                return "time_index"

    # --------------------------------------------------
    # Numeric columns
    # --------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):
        return "continuous"

    # --------------------------------------------------
    # Categorical columns
    # --------------------------------------------------

    if len(series) > 0:

        unique_ratio = (
            series.nunique(dropna=True) / len(series)
        )

        if unique_ratio < 0.05:
            return "categorical"

    # --------------------------------------------------
    # Remaining text columns
    # --------------------------------------------------

    return "text_or_id"

def profile_dataset(filepath):
    """
    Load a CSV file and generate a detailed dataset profile.
    """

    df = pd.read_csv(filepath)

    profile = {
        "file": filepath,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": {}
    }

    for col in df.columns:

        series = df[col]

        # Basic information
        col_info = {
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_pct": round(
                series.isnull().mean() * 100,
                2
            ),
            "unique_count": int(series.nunique(dropna=True)),
            "role": infer_role(series)
        }

        # Numeric column information
        if col_info["role"] == "continuous":

            clean_series = series.dropna()

            if len(clean_series) > 0:

                col_info.update({
                    "mean": round(float(clean_series.mean()), 2),
                    "median": round(float(clean_series.median()), 2),
                    "std": round(float(clean_series.std()), 2),
                    "min": round(float(clean_series.min()), 2),
                    "max": round(float(clean_series.max()), 2),
                    "skew": round(
                        float(stats.skew(clean_series)),
                        2
                    )
                })

        # Categorical information
        elif col_info["role"] == "categorical":

            value_counts = (
                series
                .value_counts(dropna=False)
                .head(10)
                .to_dict()
            )

            col_info["top_values"] = {
                str(k): int(v)
                for k, v in value_counts.items()
            }

        # Date information
        elif col_info["role"] == "time_index":

            converted = pd.to_datetime(
                series,
                errors="coerce"
            )

            valid_dates = converted.dropna()

            if len(valid_dates) > 0:
                col_info.update({
                    "min_date": str(valid_dates.min()),
                    "max_date": str(valid_dates.max())
                })

        profile["columns"][col] = col_info

    return profile


if __name__ == "__main__":

    # Your current dataset
    filepath = "data/raw/sales.csv"

    result = profile_dataset(filepath)

    print("\n" + "=" * 60)
    print("DATASET PROFILE")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=4,
            default=str
        )
    )

    print("=" * 60)
    print("PROFILING COMPLETED")
    print("=" * 60)