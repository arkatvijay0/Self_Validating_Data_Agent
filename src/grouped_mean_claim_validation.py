"""Group-aware and metric-aware numerical evidence checks."""

import re


_NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?")


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _close(actual, expected):
    return abs(actual - expected) <= max(0.011, abs(expected) * 0.00001)


def _group_patterns(group):
    label = str(group).strip()
    normalized = label.casefold()

    if normalized == "no":
        return [
            re.compile(r"\bnon[- ]subscribers?\b", re.I),
            re.compile(r"\bnon[- ]subscriber group\b", re.I),
            re.compile(r"\bnot[- ]subscribed\b", re.I),
            re.compile(r"\b(?:customers? )?who did not subscribe\b", re.I),
            re.compile(r"\bdidn't subscribe\b", re.I),
            re.compile(r"\bno\b", re.I),
        ]

    if normalized == "yes":
        return [
            re.compile(r"(?<!non )(?<!non-)\bsubscribers?\b", re.I),
            re.compile(r"\bsubscribed customers?\b", re.I),
            re.compile(r"(?<!not )(?<!non-)\bsubscribed\b", re.I),
            re.compile(r"\byes\b", re.I),
        ]

    if not label:
        return []

    left = r"(?<!\w)" if label[0].isalnum() else ""
    right = r"(?!\w)" if label[-1].isalnum() else ""

    return [
        re.compile(left + re.escape(label) + right, re.I)
    ]


def group_label_is_present(text, group):
    return any(
        pattern.search(str(text or ""))
        for pattern in _group_patterns(group)
    )


def _number_near_phrase(
    text,
    patterns,
    expected,
    max_distance=100,
    split_and=False
):
    expected = _finite_float(expected)

    if expected is None:
        return False

    source = str(text or "")

    separators = (
        r"[,;]+|(?<!\d)[.!?](?!\d)|\n+|"
        r"\b(?:while|whereas|but|compared with|compared to|versus|vs\.?)\b"
    )

    if split_and:
        separators += r"|\band\b"

    # Each clause should contain the group label and its own value.
    # This prevents a following group's value from being borrowed
    # by the preceding group.
    for segment in re.split(separators, source, flags=re.I):

        number_matches = []

        for match in _NUMBER.finditer(segment):
            try:
                number_matches.append(
                    (
                        match,
                        float(match.group(0).replace(",", ""))
                    )
                )
            except ValueError:
                continue

        if not number_matches:
            continue

        for pattern in patterns:
            for phrase in pattern.finditer(segment):

                candidates = [
                    (number_match, value)
                    for number_match, value in number_matches
                    if abs(
                        number_match.start() - phrase.end()
                    ) <= max_distance
                ]

                if not candidates:
                    continue

                _, nearest = min(
                    candidates,
                    key=lambda item: abs(
                        item[0].start() - phrase.end()
                    ),
                )

                if _close(nearest, expected):
                    return True

    return False


def number_is_associated_with_group(text, group, expected):
    return _number_near_phrase(
        text,
        _group_patterns(group),
        expected,
        max_distance=100,
        split_and=" and " not in f" {str(group).strip().casefold()} ",
    )


def number_is_associated_with_phrase(
    text,
    phrase_pattern,
    expected
):
    return _number_near_phrase(
        text,
        [re.compile(phrase_pattern, re.I)],
        expected,
        max_distance=80
    )


def extract_metric_statement(insight, metric, all_metrics):
    """Return the metric's own '<metric>: ...' statement, or None."""

    source = str(insight or "")
    matches = []

    for name in all_metrics:
        name = str(name).strip()

        if not name:
            continue

        match = re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*:",
            source,
            re.I,
        )

        if match:
            matches.append(
                (match.start(), match.end(), name)
            )

    target = str(metric).strip().casefold()

    selected = next(
        (
            item for item in matches
            if item[2].casefold() == target
        ),
        None,
    )

    if selected is None:
        return None

    later = [
        item[0]
        for item in matches
        if item[0] > selected[0]
    ]

    end = min(later) if later else len(source)

    return source[selected[1]:end].strip()


def metric_group_mean_checks(
    insight,
    metric,
    result,
    all_metrics
):
    """Check both group means and the difference inside one metric statement."""

    block = extract_metric_statement(
        insight,
        metric,
        all_metrics
    )

    if not isinstance(result, dict):
        return {
            "statement_present": block is not None,
            "non_subscriber_mean_present": False,
            "subscriber_mean_present": False,
            "difference_present": False,
            "pass": False,
            "reason": "A valid grouped-mean result is missing.",
        }

    groups = result.get("group_results")

    if not isinstance(groups, dict):
        groups = {}

    no_group = groups.get("no", {})
    yes_group = groups.get("yes", {})

    no_mean = (
        no_group.get("mean_value", no_group.get("mean"))
        if isinstance(no_group, dict)
        else None
    )

    yes_mean = (
        yes_group.get("mean_value", yes_group.get("mean"))
        if isinstance(yes_group, dict)
        else None
    )

    difference = result.get(
        "mean_difference_yes_minus_no"
    )

    no_ok = (
        block is not None
        and number_is_associated_with_group(
            block, "no", no_mean
        )
    )

    yes_ok = (
        block is not None
        and number_is_associated_with_group(
            block, "yes", yes_mean
        )
    )

    difference_ok = (
        block is not None
        and number_is_associated_with_phrase(
            block,
            r"subscriber[- ]minus[- ]non[- ]subscriber difference",
            difference,
        )
    )

    passed = bool(
        block is not None
        and no_ok
        and yes_ok
        and difference_ok
    )

    return {
        "statement_present": block is not None,
        "non_subscriber_mean_present": no_ok,
        "subscriber_mean_present": yes_ok,
        "difference_present": difference_ok,
        "pass": passed,
        "reason": None if passed else (
            f"The statement for '{metric}' does not correctly associate "
            "both group means and the difference with this metric."
        ),
    }