import pandas as pd
import numpy as np
import re


# ============================================================
# HELPER - CONVERT TO SERIES
# ============================================================

def _to_series(result):
    if isinstance(result, pd.Series):
        return result
    if isinstance(result, dict):
        if "Region" in result and "Total Profit" in result:
            return pd.Series({result["Region"]: result["Total Profit"]})
        try:
            return pd.Series(result)
        except Exception:
            return None
    if isinstance(result, pd.DataFrame):
        if result.shape[1] >= 2:
            return pd.Series(
                result.iloc[:, 1].values,
                index=result.iloc[:, 0].values
            )
    return None


# ============================================================
# HELPER - SAFE NUMBER
# ============================================================

def _safe_number(value):
    try:
        return float(value)
    except Exception:
        return None


# ============================================================
# GROUPED EVIDENCE
# ============================================================

def create_grouped_evidence(
    result,
    category,
    metric,
    operation
):
    series = _to_series(result)

    if series is None:
        return {
            "status": "FAIL",
            "analysis_type": "grouped_analysis",
            "category": category,
            "metric": metric,
            "operation": operation,
            "reason": "Result could not be converted to a pandas Series."
        }

    if len(series) == 0:
        return {
            "status": "FAIL",
            "analysis_type": "grouped_analysis",
            "category": category,
            "metric": metric,
            "operation": operation,
            "reason": "Grouped result is empty."
        }

    series = series.sort_values(ascending=False)
    top_category = str(series.index[0])
    top_value = _safe_number(series.iloc[0])

    second_category = None
    second_value = None

    if len(series) > 1:
        second_category = str(series.index[1])
        second_value = _safe_number(series.iloc[1])

    supporting_result = {
        str(key): _safe_number(value)
        for key, value in series.items()
    }

    if second_category is not None:
        evidence_summary = (
            f"The {operation} of {metric} "
            f"was calculated for each {category}. "
            f"{top_category} had the highest value "
            f"at {top_value:,.2f}. "
            f"{second_category} ranked second "
            f"with {second_value:,.2f}."
        )
    else:
        evidence_summary = (
            f"The {operation} of {metric} "
            f"was calculated for each {category}. "
            f"{top_category} had the highest value "
            f"at {top_value:,.2f}."
        )

    return {
        "status": "PASS",
        "analysis_type": "grouped_analysis",
        "category": category,
        "metric": metric,
        "operation": operation,
        "evidence": {
            "top_category": top_category,
            "top_value": top_value,
            "second_category": second_category,
            "second_value": second_value
        },
        "supporting_result": supporting_result,
        "evidence_summary": evidence_summary
    }


# ============================================================
# CORRELATION EVIDENCE
# ============================================================

def create_correlation_evidence(
    result,
    x_column="Variable 1",
    y_column="Variable 2",
    method="Pearson"
):
    try:
        try:
            from insight_generator import _extract_correlation_data
        except ImportError:
            from .insight_generator import _extract_correlation_data

        correlation, x_var, y_var = _extract_correlation_data(result)
        if x_var:
            x_column = x_var
        if y_var:
            y_column = y_var
    except Exception:
        correlation = _safe_number(result)

    correlation = _safe_number(correlation)
    if correlation is None:
        return {
            "status": "FAIL",
            "analysis_type": "correlation_analysis",
            "reason": "Correlation result is not numeric."
        }

    direction = "positive" if correlation > 0 else ("negative" if correlation < 0 else "neutral")
    absolute_value = abs(correlation)

    if absolute_value < 0.20:
        strength = "very weak"
    elif absolute_value < 0.40:
        strength = "weak"
    elif absolute_value < 0.60:
        strength = "moderate"
    elif absolute_value < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    return {
        "status": "PASS",
        "analysis_type": "correlation_analysis",
        "x_variable": x_column,
        "y_variable": y_column,
        "method": method,
        "correlation": correlation,
        "direction": direction,
        "strength": strength,
        "evidence_summary": (
            f"{method} correlation between {x_column} and {y_column} "
            f"is {correlation:.4f}, indicating a {strength} {direction} relationship."
        )
    }


# ============================================================
# TREND EVIDENCE
# ============================================================

def create_trend_evidence(
    result,
    metric="Total Revenue",
    time_label="Order Date"
):
    try:
        if isinstance(result, pd.DataFrame):
            frame = result.copy()
        elif isinstance(result, pd.Series):
            frame = result.rename(metric).reset_index()
            frame.columns = [time_label, metric]
        elif isinstance(result, dict):
            frame = pd.DataFrame({
                time_label: list(result.keys()),
                metric: list(result.values())
            })
        elif isinstance(result, list):
            frame = pd.DataFrame(result)
        else:
            frame = pd.DataFrame()

        if frame.empty:
            return {
                "status": "FAIL",
                "analysis_type": "trend_analysis",
                "metric": metric,
                "time_label": time_label,
                "reason": "Trend result is empty or has an unsupported format."
            }

        time_col = time_label if time_label in frame.columns else None
        metric_col = metric if metric in frame.columns else None

        if metric_col is None:
            numeric_cols = [
                col for col in frame.columns
                if pd.api.types.is_numeric_dtype(frame[col])
            ]
            if len(numeric_cols) == 1:
                metric_col = numeric_cols[0]

        if time_col is None:
            non_metric_cols = [
                col for col in frame.columns if col != metric_col
            ]
            if non_metric_cols:
                time_col = non_metric_cols[0]

        if metric_col is None:
            return {
                "status": "FAIL",
                "analysis_type": "trend_analysis",
                "metric": metric,
                "time_label": time_label,
                "reason": "No numeric metric column was found in the trend result."
            }

        frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
        frame = frame.dropna(subset=[metric_col])

        if frame.empty:
            return {
                "status": "FAIL",
                "analysis_type": "trend_analysis",
                "metric": metric,
                "time_label": time_label,
                "reason": "Trend result contains no valid numeric metric values."
            }

        values = frame[metric_col].astype(float).tolist()
        first_value = float(values[0])
        last_value = float(values[-1])

        if last_value > first_value:
            direction = "increasing"
        elif last_value < first_value:
            direction = "decreasing"
        else:
            direction = "unchanged"

        supporting_result = {
            "time_label": str(time_col or time_label),
            "metric": str(metric_col),
            "observation_count": int(len(values)),
            "first_value": first_value,
            "last_value": last_value,
            "minimum": float(min(values)),
            "maximum": float(max(values)),
            "direction_first_to_last": direction,
            "records": [
                {
                    "time": str(row[time_col]) if time_col is not None else str(i),
                    "value": float(row[metric_col])
                }
                for i, (_, row) in enumerate(frame.iterrows())
            ]
        }

        return {
            "status": "PASS",
            "analysis_type": "trend_analysis",
            "metric": str(metric_col),
            "time_label": str(time_col or time_label),
            "evidence": supporting_result,
            "supporting_result": supporting_result,
            "evidence_summary": (
                f"{metric_col} has {len(values)} valid observations. "
                f"The first observed value is {first_value:,.2f}, the last is "
                f"{last_value:,.2f}, and the first-to-last direction is {direction}."
            )
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "analysis_type": "trend_analysis",
            "metric": metric,
            "time_label": time_label,
            "reason": f"Trend evidence could not be created: {exc}"
        }


# ============================================================
# EVIDENCE SUMMARY & MAPPING HELPERS
# ============================================================

def create_evidence_summary(evidence):
    if not isinstance(evidence, dict) or evidence.get("status") != "PASS":
        return "Evidence validation failed."
    return evidence.get("evidence_summary", "Evidence summary unavailable.")


def _extract_claimed_category(insight, evidence):
    if not isinstance(insight, str):
        return None
    evidence_details = evidence.get("evidence", {})
    top_category = evidence_details.get("top_category")
    second_category = evidence_details.get("second_category")
    candidates = []
    if top_category is not None:
        candidates.append(str(top_category))
    if second_category is not None:
        candidates.append(str(second_category))
    supporting_result = evidence.get("supporting_result")
    if isinstance(supporting_result, dict):
        candidates.extend([str(key) for key in supporting_result.keys()])
    candidates = sorted(set(candidates), key=len, reverse=True)
    normalized_insight = insight.lower()
    for candidate in candidates:
        if candidate.lower() in normalized_insight:
            return candidate
    return None


def _validate_grouped_insight(insight, evidence):
    evidence_details = evidence.get("evidence", {})
    expected_category = evidence_details.get("top_category")
    expected_value = evidence_details.get("top_value")
    claimed_category = _extract_claimed_category(insight, evidence)

    if expected_category is None:
        return {"status": "FAIL", "reason": "Evidence does not contain a top category."}
    if claimed_category is None:
        return {"status": "FAIL", "reason": "The insight does not contain a category supported by the evidence."}

    if claimed_category.lower() != str(expected_category).lower():
        return {
            "status": "FAIL",
            "reason": f"Insight contradicts the evidence. Expected top category: {expected_category}. Insight claims: {claimed_category}.",
            "expected_category": expected_category,
            "claimed_category": claimed_category,
            "expected_value": expected_value
        }
    return {
        "status": "PASS",
        "reason": None,
        "expected_category": expected_category,
        "claimed_category": claimed_category,
        "expected_value": expected_value
    }


def _validate_correlation_insight(insight, evidence):
    if not isinstance(insight, str):
        return {"status": "FAIL", "reason": "Insight is not text."}
    correlation = evidence.get("correlation")
    if correlation is None:
        return {"status": "FAIL", "reason": "Correlation evidence is missing."}
    
    method = evidence.get("method", "Pearson")
    x_variable = evidence.get("x_variable")
    y_variable = evidence.get("y_variable")
    normalized = insight.lower()

    if x_variable and x_variable.lower() not in normalized:
        return {"status": "FAIL", "reason": f"Insight does not mention {x_variable}."}
    if y_variable and y_variable.lower() not in normalized:
        return {"status": "FAIL", "reason": f"Insight does not mention {y_variable}."}

    expected_direction = evidence.get("direction")
    expected_strength = evidence.get("strength")

    if expected_direction and expected_direction.lower() not in normalized:
        return {"status": "FAIL", "reason": "Insight direction does not match the evidence."}
    if expected_strength and expected_strength.lower() not in normalized:
        return {"status": "FAIL", "reason": "Insight strength does not match the evidence."}

    return {
        "status": "PASS",
        "reason": None,
        "method": method,
        "correlation": correlation,
        "direction": expected_direction,
        "strength": expected_strength
    }


def _parse_claim_number(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def map_insight_to_evidence(insight, evidence):
    if not isinstance(evidence, dict):
        return {"status": "FAIL", "analysis_type": "unknown", "insight": insight, "reason": "Evidence must be a dictionary."}
    if evidence.get("status") != "PASS":
        return {"status": "FAIL", "analysis_type": evidence.get("analysis_type", "unknown"), "insight": insight, "reason": evidence.get("reason", "Evidence is invalid.")}

    analysis_type = evidence.get("analysis_type")

    if analysis_type == "correlation_analysis":
        validation = _validate_correlation_insight(insight, evidence)
        res = {
            "status": validation.get("status"),
            "analysis_type": analysis_type,
            "insight": insight,
            "evidence_summary": create_evidence_summary(evidence),
            "x_variable": evidence.get("x_variable"),
            "y_variable": evidence.get("y_variable"),
            "method": evidence.get("method"),
            "correlation": evidence.get("correlation"),
            "direction": evidence.get("direction"),
            "strength": evidence.get("strength")
        }
        if validation.get("status") == "FAIL":
            res["reason"] = validation.get("reason")
        return res

    if analysis_type == "grouped_analysis":
        validation = _validate_grouped_insight(insight, evidence)
        res = {
            "status": validation.get("status"),
            "analysis_type": analysis_type,
            "insight": insight,
            "evidence_summary": create_evidence_summary(evidence),
            "top_category": evidence.get("evidence", {}).get("top_category"),
            "top_value": evidence.get("evidence", {}).get("top_value"),
            "supporting_result": evidence.get("supporting_result")
        }
        if validation.get("status") == "FAIL":
            res["reason"] = validation.get("reason")
        return res

    return {
        "status": "PASS",
        "analysis_type": analysis_type or "generic",
        "insight": insight,
        "evidence_summary": create_evidence_summary(evidence),
        "supporting_result": evidence.get("supporting_result")
    }