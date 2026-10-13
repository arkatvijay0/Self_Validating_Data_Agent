
from pathlib import Path
import pandas as pd


def calculate_conversion_rate_by_group(
    filepath,
    group_column,
    bins=None,
    labels=None,
):
    """Calculate independently verifiable conversion rates by a requested column."""
    df = pd.read_csv(filepath)

    if group_column not in df.columns:
        raise ValueError(
            f"Column '{group_column}' is not present in the dataset."
        )

    if "conversion" in df.columns:
        outcome = pd.to_numeric(df["conversion"], errors="coerce")
        valid = outcome.isin([0, 1])
        converted = outcome.eq(1).astype(int)
    elif "y" in df.columns:
        outcome = df["y"].astype(str).str.strip().str.lower()
        valid = outcome.isin(["yes", "no"])
        converted = outcome.eq("yes").astype(int)
    else:
        raise ValueError("Dataset needs a 'conversion' or 'y' target column.")

    work = df.loc[
        valid & df[group_column].notna(),
        [group_column],
    ].copy()

    work["converted"] = converted.loc[work.index].astype(int)

    actual_group = group_column

    if bins is not None:
        if labels is None:
            raise ValueError("Provide labels when using balance bins.")

        actual_group = "_analysis_group"
        work[actual_group] = pd.cut(
            pd.to_numeric(work[group_column], errors="coerce"),
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        work = work.loc[work[actual_group].notna()].copy()

    result = work.groupby(actual_group, observed=False).agg(
        customers=("converted", "size"),
        converted=("converted", "sum"),
    )

    result["conversion_rate_percent"] = (
        result["converted"] / result["customers"] * 100
    ).round(2)

    return result
