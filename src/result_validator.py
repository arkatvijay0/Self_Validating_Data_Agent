"""Deterministic validation of specification-driven EDA step results.

This module independently recomputes supported scalar and grouped aggregations
from the original CSV and compares them with the generated result. It also
checks operation coverage, row counts, filter metadata, and result structure.

It is deliberately conservative: unsupported result shapes are reported as
REVIEW rather than being treated as validated.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from anomaly_detector import detect_anomalies, select_anomaly_features

AGGREGATIONS = {"mean", "median", "sum", "count", "min", "max", "std", "variance", "nunique"}
STEP_TERMS = {
    "mean": ("average", "mean", "avg"),
    "median": ("median",), "sum": ("sum", "total"),
    "count": ("count", "number of", "how many"),
    "min": ("minimum", "smallest", "lowest"),
    "max": ("maximum", "largest", "highest"),
    "std": ("standard deviation", "std"), "variance": ("variance",),
    "nunique": ("distinct", "unique"),
    "missing_values": ("missing", "null"),
    "duplicate_count": ("duplicate",),
    "describe": ("describe", "summary", "profile"),
    "correlation": ("correlation", "correlated", "relationship"),
    "outlier_detection": ("outlier", "anomaly"),
    "time_trend": ("trend", "over time", "monthly", "weekly", "yearly"),
    "rank": ("rank", "top ", "bottom "),
}


def _close(actual: Any, expected: Any, rel_tol: float, abs_tol: float) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError):
        return actual == expected
    if math.isnan(a) and math.isnan(e):
        return True
    return math.isclose(a, e, rel_tol=rel_tol, abs_tol=abs_tol)


def _apply_filters(df: pd.DataFrame, filters: List[dict]) -> pd.DataFrame:
    work = df.copy()
    for item in filters:
        col, op, value = item.get("column"), item.get("operator"), item.get("value")
        if col not in work.columns:
            raise KeyError(f"Filter column not found: {col}")
        series = pd.to_numeric(work[col], errors="coerce")
        operations = {
            ">": series > value, ">=": series >= value,
            "<": series < value, "<=": series <= value,
            "==": series == value, "!=": series != value,
        }
        if op not in operations:
            raise ValueError(f"Unsupported filter operator: {op}")
        work = work.loc[operations[op]].copy()
    return work


def _selected_operations(step_text: str, spec: dict) -> List[str]:
    ops = list(spec.get("operations") or [])
    text = str(step_text or "").casefold()
    if text.startswith("apply all explicitly requested filters"):
        return ["filter_only"]
    if "independently recompute" in text:
        return ops
    if len(ops) <= 1:
        return ops
    selected = [op for op in ops if any(term in text for term in STEP_TERMS.get(op, ()))]
    if "rank" in selected and "mean" in ops and spec.get("group_by"):
        selected = ["mean"] + [op for op in selected if op != "mean"]
    return selected


def _calculate(series: pd.Series, operation: str):
    if operation in {"mean", "median", "sum", "std", "variance"}:
        values = pd.to_numeric(series, errors="coerce")
        if operation == "mean": return values.mean()
        if operation == "median": return values.median()
        if operation == "sum": return values.sum()
        if operation == "std": return values.std()
        if operation == "variance": return values.var()
    if operation == "count": return series.count()
    if operation == "min": return series.min()
    if operation == "max": return series.max()
    if operation == "nunique": return series.nunique()
    raise ValueError(f"Cannot independently recompute operation: {operation}")


def _compare_aggregation(actual: Any, expected: Any, rel_tol: float, abs_tol: float) -> Tuple[bool, str]:
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False, "Expected grouped aggregation rows, but result was not a list."
        # Compare rows by group-key tuple, independent of row order.
        key_cols = [k for k in expected[0] if not any("__" in x for x in expected[0]) or "__" not in k] if expected else []
        def key(row): return tuple(str(row.get(k)) for k in key_cols)
        actual_map = {key(row): row for row in actual if isinstance(row, dict)}
        expected_map = {key(row): row for row in expected if isinstance(row, dict)}
        if actual_map.keys() != expected_map.keys():
            return False, "Grouped result keys do not match independently recomputed groups."
        for group_key, expected_row in expected_map.items():
            actual_row = actual_map[group_key]
            for col, value in expected_row.items():
                if col not in actual_row or not _close(actual_row[col], value, rel_tol, abs_tol):
                    return False, f"Grouped value mismatch for {col} in group {group_key}."
        return True, "Grouped aggregation matches independent recomputation."
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False, "Aggregation result has an unexpected structure."
    if actual.keys() != expected.keys():
        return False, "Aggregation metric/operation keys do not match the specification."
    for key, value in expected.items():
        if key not in actual or not _close(actual[key], value, rel_tol, abs_tol):
            return False, f"Aggregation mismatch for {key}."
    return True, "Aggregation values match independent recomputation."



def _nested_close(actual: Any, expected: Any, rel_tol: float, abs_tol: float) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _nested_close(actual[k], v, rel_tol, abs_tol) for k, v in expected.items()
        )
    if isinstance(expected, (int, float)) or expected is None:
        return _close(actual, expected, rel_tol, abs_tol)
    return actual == expected


def _validate_advanced_operation(
    operation: str, filtered: pd.DataFrame, result: dict, spec: dict,
    rel_tol: float, abs_tol: float
) -> Tuple[str, str]:
    """Recompute correlation, IQR outliers, trends, and rankings independently."""
    outputs = result.get("operation_results") or {}
    metrics = list(spec.get("metrics") or [])
    groups = list(spec.get("group_by") or [])

    if operation == "correlation":
        if len(metrics) < 2 or any(c not in filtered.columns for c in metrics):
            return "REVIEW", "Correlation cannot be checked: at least two resolved metric columns are required."
        methods = list(spec.get("statistics") or ["pearson"])
        methods = [m for m in methods if m in {"pearson", "spearman"}]
        if not methods:
            methods = ["pearson"]
        expected = {m: filtered[metrics].corr(method=m).to_dict() for m in methods}
        actual = outputs.get("correlation")
        if not _nested_close(actual, expected, rel_tol, abs_tol):
            return "FAIL", "Correlation matrix does not match independent recomputation."
        return "PASS", "Correlation matrices match independent recomputation."

    if operation == "outlier_detection":
        outlier_method = str(spec.get("outlier_method") or "IQR")
        if outlier_method.replace("_", "").replace("-", "").casefold() == "isolationforest":
            if not isinstance(result, dict):
                return "FAIL", "Isolation Forest result is missing or has an invalid structure."
            if str(result.get("method", "")).replace("_", "").replace("-", "").casefold() != "isolationforest":
                return "FAIL", "The requested Isolation Forest method was not executed."
            expected_features = select_anomaly_features(filtered, spec, None)
            contamination = result.get("contamination", 0.05)
            try:
                contamination = float(contamination)
            except (TypeError, ValueError):
                return "FAIL", "Isolation Forest contamination setting is invalid."
            independent = detect_anomalies(
                filtered,
                feature_columns=expected_features,
                contamination=contamination,
                random_state=42,
                n_estimators=200,
            )
            if not isinstance(independent, dict) or independent.get("status") != "PASS":
                reason = independent.get("reason", "Independent Isolation Forest recomputation failed.") if isinstance(independent, dict) else "Invalid detector output."
                return "REVIEW", str(reason)
            expected_summary = independent.get("summary") or {}
            actual_summary = result.get("summary") or {}
            expected_count = int(expected_summary.get("anomaly_count", -1))
            expected_rows = int(expected_summary.get("total_rows", -1))
            expected_normal = int(expected_summary.get("normal_count", -1))
            expected_percentage = float(expected_summary.get("detected_anomaly_percentage", -1))
            checks = {
                "row_count": result.get("row_count") == expected_rows == len(filtered),
                "anomaly_count": result.get("anomaly_count") == expected_count,
                "normal_count": result.get("normal_count") == expected_normal,
                "feature_columns": result.get("feature_columns") == expected_summary.get("feature_columns"),
                "contamination": _close(result.get("contamination"), contamination, rel_tol, abs_tol),
                "percentage": _close(result.get("anomaly_percentage"), expected_percentage, rel_tol, 0.02),
                "summary_algorithm": actual_summary.get("algorithm") == "Isolation Forest",
                "summary_counts": (
                    actual_summary.get("total_rows") == expected_rows
                    and actual_summary.get("anomaly_count") == expected_count
                    and actual_summary.get("normal_count") == expected_normal
                ),
                "summary_features": actual_summary.get("feature_columns") == expected_summary.get("feature_columns"),
                "summary_contamination": _close(
                    actual_summary.get("contamination_setting"),
                    expected_summary.get("contamination_setting"),
                    rel_tol,
                    abs_tol,
                ),
                "summary_percentage": _close(
                    actual_summary.get("detected_anomaly_percentage"),
                    expected_summary.get("detected_anomaly_percentage"),
                    rel_tol,
                    abs_tol,
                ),
            }
            if not all(checks.values()):
                failed = [name for name, passed in checks.items() if not passed]
                return "FAIL", "Isolation Forest independent recomputation mismatch: " + ", ".join(failed) + "."
            sample = result.get("sample_anomalies")
            if sample is not None:
                if not isinstance(sample, pd.DataFrame):
                    return "FAIL", "Isolation Forest anomaly sample must be a pandas DataFrame."
                recomputed_anomalies = independent.get("anomalies")
                if not isinstance(recomputed_anomalies, pd.DataFrame):
                    return "FAIL", "Independent anomaly table is unavailable."
                expected_sample = recomputed_anomalies.head(len(sample))
                if list(sample.index) != list(expected_sample.index):
                    return "FAIL", "Displayed anomaly sample rows do not match independent Isolation Forest results."
                if "anomaly_score" not in sample.columns or not all(
                    _close(a, b, rel_tol, abs_tol)
                    for a, b in zip(
                        pd.to_numeric(sample["anomaly_score"], errors="coerce").tolist(),
                        pd.to_numeric(expected_sample["anomaly_score"], errors="coerce").tolist(),
                    )
                ):
                    return "FAIL", "Displayed anomaly scores do not match independent Isolation Forest results."
            return "PASS", (
                "Isolation Forest was independently rerun on the filtered source rows; "
                "selected features, anomaly counts, percentage, summary, and sample scores match."
            )

        if not metrics:
            return "REVIEW", "Outlier validation requires at least one resolved metric."
        actual = outputs.get("outlier_detection")
        if not isinstance(actual, dict):
            return "FAIL", "Outlier result is missing or has an invalid structure."
        for metric in metrics:
            if metric not in filtered.columns:
                return "FAIL", f"Outlier metric column is missing: {metric}."
            values = pd.to_numeric(filtered[metric], errors="coerce").dropna()
            if values.empty:
                return "REVIEW", f"No valid numeric values are available for outlier metric {metric}."
            item = actual.get(metric)
            if outlier_method == "ZScore":
                mean = float(values.mean())
                std = float(values.std(ddof=0))
                z_scores = ((values - mean) / std) if std > 0 else pd.Series(0.0, index=values.index)
                expected = {
                    "method": "ZScore",
                    "threshold": 3.0,
                    "mean": mean,
                    "standard_deviation": std,
                    "outlier_count": int((z_scores.abs() > 3.0).sum()),
                    "valid_count": int(values.count()),
                }
                if not _nested_close(item, expected, rel_tol, abs_tol):
                    return "FAIL", f"Z-score outlier result does not match independent recomputation for {metric}."
            else:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                count = int(((values < lower) | (values > upper)).sum())
                expected = {"method": "IQR", "lower_bound": float(lower), "upper_bound": float(upper),
                            "outlier_count": count, "valid_count": int(values.count())}
                if not _nested_close(item, expected, rel_tol, abs_tol):
                    return "FAIL", f"IQR outlier result does not match independent recomputation for {metric}."
        return "PASS", f"{outlier_method} bounds, outlier counts, and valid counts match independent recomputation."

    if operation == "time_trend":
        time_info = spec.get("time") or {}
        column, grain = time_info.get("column"), time_info.get("grain") or "M"
        if not column or column not in filtered.columns or not metrics:
            return "REVIEW", "Trend validation requires a resolved date column and metric."
        dates = pd.to_datetime(filtered[column], errors="coerce")
        temp = filtered.loc[dates.notna()].copy()
        if temp.empty:
            return "REVIEW", "No parseable dates are available for trend validation."
        temp["_period"] = dates.loc[dates.notna()].dt.to_period(grain).astype(str)
        expected_rows = []
        for period, frame in temp.groupby("_period", sort=True):
            row = {"period": period}
            for metric in metrics:
                row[f"{metric}__sum"] = float(pd.to_numeric(frame[metric], errors="coerce").sum())
            expected_rows.append(row)
        actual_rows = outputs.get("time_trend")
        if not isinstance(actual_rows, list) or len(actual_rows) != len(expected_rows):
            return "FAIL", "Trend periods do not match independent recomputation."
        actual_map = {str(row.get("period")): row for row in actual_rows if isinstance(row, dict)}
        expected_map = {row["period"]: row for row in expected_rows}
        if actual_map.keys() != expected_map.keys():
            return "FAIL", "Trend period labels do not match independent recomputation."
        for period, expected in expected_map.items():
            if not _nested_close(actual_map[period], expected, rel_tol, abs_tol):
                return "FAIL", f"Trend values do not match independent recomputation for period {period}."
        return "PASS", "Trend periods and metric totals match independent recomputation."

    if operation == "rank":
        actual_rows = outputs.get("ranking")
        if not isinstance(actual_rows, list):
            return "FAIL", "Ranking result is missing or is not a list."
        if not metrics:
            return "REVIEW", "Ranking validation requires a resolved metric."
        metric = metrics[0]
        direction = (spec.get("sorting") or {}).get("direction")
        ascending = direction == "ascending"
        limit = (spec.get("sorting") or {}).get("limit")
        if groups:
            agg_ops = [op for op in (spec.get("operations") or []) if op in AGGREGATIONS]
            agg_op = agg_ops[0] if agg_ops else "mean"
            if metric not in filtered.columns or any(c not in filtered.columns for c in groups):
                return "REVIEW", "Grouped ranking columns are unavailable."
            expected = []
            for key, frame in filtered.groupby(groups, dropna=False, sort=True):
                key_tuple = key if isinstance(key, tuple) else (key,)
                row = dict(zip(groups, key_tuple))
                value = _calculate(frame[metric], agg_op)
                row[f"{metric}__{agg_op}"] = None if pd.isna(value) else float(value)
                expected.append(row)
            sort_key = f"{metric}__{agg_op}"
            expected.sort(key=lambda r: r[sort_key] if r[sort_key] is not None else float("-inf"), reverse=not ascending)
            if limit:
                expected = expected[:int(limit)]
            if len(actual_rows) != len(expected):
                return "FAIL", "Grouped ranking row count does not match the requested ranking."
            for actual, wanted in zip(actual_rows, expected):
                if not _nested_close(actual, wanted, rel_tol, abs_tol):
                    return "FAIL", "Grouped ranking order or values do not match independent recomputation."
        else:
            if metric not in filtered.columns:
                return "REVIEW", f"Ranking metric is unavailable: {metric}."
            expected_frame = filtered.sort_values(metric, ascending=ascending, na_position="last")
            if limit:
                expected_frame = expected_frame.head(int(limit))
            expected_rows = expected_frame.to_dict(orient="records")
            if len(actual_rows) != len(expected_rows):
                return "FAIL", "Ranking row count does not match the requested ranking."
            for actual, wanted in zip(actual_rows, expected_rows):
                if not isinstance(actual, dict) or not isinstance(wanted, dict):
                    return "FAIL", "Ranking contains a malformed row."
                if actual.keys() != wanted.keys() or any(
                    not _nested_close(actual[k], wanted[k], rel_tol, abs_tol) for k in wanted
                ):
                    return "FAIL", "Ranking order or row values do not match independent recomputation."
        return "PASS", "Ranking order and values match independent recomputation."

    return "REVIEW", f"No advanced validator is registered for {operation}."

def validate_analytical_results(
    filepath: str,
    analytical_spec: dict,
    execution_results: dict,
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-9,
) -> Dict[str, Any]:
    """Validate completed step outputs against the original CSV dataset."""
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []
    if not isinstance(analytical_spec, dict):
        return {"status": "REVIEW", "checks": [], "errors": ["Analytical specification is missing."]}
    try:
        source = pd.read_csv(filepath)
    except Exception as exc:
        return {"status": "FAIL", "checks": [], "errors": [f"Could not reload source dataset: {type(exc).__name__}: {exc}"]}

    steps = execution_results.get("steps", []) if isinstance(execution_results, dict) else []
    if not isinstance(steps, list) or not steps:
        return {"status": "REVIEW", "checks": [], "errors": ["No execution step results were available."]}

    required_ops = set(analytical_spec.get("operations") or [])
    observed_ops = set()
    spec_metrics = list(analytical_spec.get("metrics") or [])
    groups = list(analytical_spec.get("group_by") or [])
    filters = list(analytical_spec.get("filters") or [])

    for step in steps:
        step_text = step.get("step", "")
        result = step.get("final_result")
        if not isinstance(result, dict):
            checks.append({"step": step.get("step_number"), "check": "result_structure", "status": "FAIL", "detail": "Step has no dictionary final_result."})
            errors.append(f"Step {step.get('step_number')} has no usable final result.")
            continue
        status = str(result.get("status", "")).upper()
        if status != "PASS":
            checks.append({"step": step.get("step_number"), "check": "execution_status", "status": "FAIL", "detail": f"Generated result status is {status or 'missing'}."})
            errors.append(f"Step {step.get('step_number')} did not return PASS.")
            continue
        ops = result.get("operations") or []
        observed_ops.update(op for op in ops if op != "filter_only")
        checks.append({"step": step.get("step_number"), "check": "result_structure", "status": "PASS", "detail": "Step returned a PASS result with operation metadata."})

        expected_step_ops = _selected_operations(step_text, analytical_spec)
        if expected_step_ops and not set(expected_step_ops).issubset(set(ops)):
            detail = f"Expected operations {expected_step_ops}, received {ops}."
            checks.append({"step": step.get("step_number"), "check": "operation_coverage", "status": "FAIL", "detail": detail})
            errors.append(f"Step {step.get('step_number')}: {detail}")
        else:
            checks.append({"step": step.get("step_number"), "check": "operation_coverage", "status": "PASS", "detail": "Step operation metadata matches its requested plan step."})

        try:
            filtered = _apply_filters(source, filters)
            if result.get("input_rows") != len(source):
                raise AssertionError("Reported input_rows does not match source dataset.")
            if result.get("analyzed_rows") != len(filtered):
                raise AssertionError("Reported analyzed_rows does not match independently filtered row count.")
            if result.get("filters_applied", []) != filters:
                raise AssertionError("Reported filters do not match analytical specification.")
            checks.append({"step": step.get("step_number"), "check": "row_and_filter_validation", "status": "PASS", "detail": "Input rows, filtered rows, and filter metadata match the source dataset."})
        except Exception as exc:
            checks.append({"step": step.get("step_number"), "check": "row_and_filter_validation", "status": "FAIL", "detail": str(exc)})
            errors.append(f"Step {step.get('step_number')}: {exc}")
            continue

        agg_ops = [op for op in expected_step_ops if op in AGGREGATIONS]
        if agg_ops and spec_metrics:
            try:
                expected: Any
                if groups:
                    grouped = filtered.groupby(groups, dropna=False, sort=True)
                    expected_rows = []
                    for key, frame in grouped:
                        key_tuple = key if isinstance(key, tuple) else (key,)
                        row = dict(zip(groups, key_tuple))
                        for metric in spec_metrics:
                            for op in agg_ops:
                                val = _calculate(frame[metric], op)
                                row[f"{metric}__{op}"] = None if pd.isna(val) else (int(val) if op in {"count", "nunique"} else float(val))
                        expected_rows.append(row)
                    expected = expected_rows
                else:
                    expected = {}
                    for metric in spec_metrics:
                        for op in agg_ops:
                            val = _calculate(filtered[metric], op)
                            expected[f"{metric}__{op}"] = None if pd.isna(val) else (int(val) if op in {"count", "nunique"} else float(val))
                actual = (result.get("operation_results") or {}).get("aggregations")
                valid, detail = _compare_aggregation(actual, expected, rel_tol, abs_tol)
                checks.append({"step": step.get("step_number"), "check": "independent_aggregation", "status": "PASS" if valid else "FAIL", "detail": detail})
                if not valid: errors.append(f"Step {step.get('step_number')}: {detail}")
            except Exception as exc:
                checks.append({"step": step.get("step_number"), "check": "independent_aggregation", "status": "REVIEW", "detail": str(exc)})
                errors.append(f"Step {step.get('step_number')} aggregation could not be independently validated: {exc}")

        advanced_ops = [op for op in expected_step_ops if op in {"correlation", "outlier_detection", "time_trend", "rank"}]
        for operation in advanced_ops:
            try:
                validation_status, detail = _validate_advanced_operation(
                    operation, filtered, result, analytical_spec, rel_tol, abs_tol
                )
                checks.append({"step": step.get("step_number"), "check": f"independent_{operation}",
                               "status": validation_status, "detail": detail})
                if validation_status == "FAIL":
                    errors.append(f"Step {step.get('step_number')}: {detail}")
            except Exception as exc:
                checks.append({"step": step.get("step_number"), "check": f"independent_{operation}",
                               "status": "REVIEW", "detail": str(exc)})
                errors.append(f"Step {step.get('step_number')} {operation} could not be independently validated: {exc}")

    missing_ops = sorted(required_ops - observed_ops)
    if missing_ops:
        errors.append("Requested operations were not observed in step results: " + ", ".join(missing_ops))
        checks.append({"check": "overall_operation_coverage", "status": "FAIL", "detail": errors[-1]})
    else:
        checks.append({"check": "overall_operation_coverage", "status": "PASS", "detail": "All requested operations appear in executed step results."})

    statuses = {item["status"] for item in checks}
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "REVIEW" in statuses:
        overall = "REVIEW"
    else:
        overall = "PASS"
    return {
        "status": overall,
        "checks": checks,
        "errors": errors,
        "requested_operations": sorted(required_ops),
        "observed_operations": sorted(observed_ops),
        "source_rows": int(len(source)),
    }



def validate_single_step_analytical_result(
    filepath: str,
    analytical_spec: dict,
    step_text: str,
    step_number: int,
    final_result: dict,
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-9,
) -> Dict[str, Any]:
    """
    Independently validate one step inside the self-correction loop.

    Only operations explicitly selected for this plan step are checked.
    Operations without an independent implementation in this validator are
    reported as skipped, not falsely marked as validated.
    """
    if not isinstance(analytical_spec, dict):
        return {
            "status": "REVIEW",
            "step_number": step_number,
            "step": step_text,
            "validated_operations": [],
            "skipped_operations": [],
            "checks": [],
            "errors": ["Analytical specification is missing."],
        }

    selected = _selected_operations(step_text, analytical_spec)
    independently_supported = [
        operation for operation in selected
        if operation in AGGREGATIONS
        or operation in {"correlation", "outlier_detection", "time_trend", "rank"}
    ]
    skipped = [
        operation for operation in selected
        if operation not in independently_supported
    ]

    if not independently_supported:
        return {
            "status": "SKIPPED",
            "step_number": step_number,
            "step": step_text,
            "validated_operations": [],
            "skipped_operations": skipped,
            "checks": [],
            "errors": [],
            "reason": (
                "This step has no operation with an independent recomputation "
                "rule in the current result validator."
            ),
        }

    step_spec = dict(analytical_spec)
    step_spec["operations"] = selected

    wrapped_execution = {
        "steps": [{
            "step_number": step_number,
            "step": step_text,
            "final_result": final_result,
        }]
    }

    validation = validate_analytical_results(
        filepath=filepath,
        analytical_spec=step_spec,
        execution_results=wrapped_execution,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

    return {
        "status": validation.get("status", "REVIEW"),
        "step_number": step_number,
        "step": step_text,
        "validated_operations": independently_supported,
        "skipped_operations": skipped,
        "checks": validation.get("checks", []),
        "errors": validation.get("errors", []),
        "reason": (
            "Independent per-step validation completed."
            if validation.get("status") == "PASS"
            else "Independent per-step validation found issues or requires review."
        ),
    }



def merge_step_validation(
    validation_result: dict,
    independent_validation: dict,
) -> dict:
    """
    Merge independent analytical checks into the existing step validator result.

    FAIL/REVIEW remain non-PASS, allowing the caller's bounded retry loop to
    receive the exact failure feedback. SKIPPED does not alter legacy checks.
    """
    if not isinstance(validation_result, dict):
        validation_result = {"overall": "REVIEW", "checks": {}}
    if not isinstance(independent_validation, dict):
        return validation_result

    status = str(independent_validation.get("status", "REVIEW")).upper()
    if status == "SKIPPED":
        validation_result["independent_analytical_validation"] = independent_validation
        return validation_result

    checks = validation_result.setdefault("checks", {})
    if not isinstance(checks, dict):
        checks = {}
        validation_result["checks"] = checks

    errors = independent_validation.get("errors") or []
    reason = "; ".join(str(item) for item in errors)
    if not reason:
        reason = independent_validation.get("reason", "No detail supplied.")

    checks["independent_analytical_result"] = {
        "pass": status == "PASS",
        "status": status,
        "reason": reason,
        "validated_operations": independent_validation.get("validated_operations", []),
        "skipped_operations": independent_validation.get("skipped_operations", []),
        "details": independent_validation.get("checks", []),
    }
    validation_result["independent_analytical_validation"] = independent_validation

    current = str(validation_result.get("overall", "REVIEW")).upper()
    if status == "FAIL":
        validation_result["overall"] = "FAIL"
    elif status == "REVIEW" and current == "PASS":
        validation_result["overall"] = "REVIEW"
    elif status == "PASS" and current not in {"FAIL", "REVIEW"}:
        validation_result["overall"] = "PASS"

    return validation_result
