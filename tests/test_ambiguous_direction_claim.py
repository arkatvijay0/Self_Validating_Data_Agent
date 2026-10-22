
import pandas as pd

import src.main as main_module


def test_contradictory_subscriber_claim_stops_before_planning(
    tmp_path, monkeypatch, capsys
):
    """Contradictory direction claims must stop before planning or execution."""

    dataset_path = tmp_path / "test_bank_marketing.csv"
    pd.DataFrame(
        {
            "balance": [100, 200, 300, 400],
            "y": ["no", "yes", "no", "yes"],
            "conversion": ["no", "yes", "no", "yes"],
        }
    ).to_csv(dataset_path, index=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "EDA planning or execution must not run for an ambiguous claim."
        )

    monkeypatch.setattr(main_module, "generate_plan_mock", fail_if_called)
    monkeypatch.setattr(main_module, "execute_plan_steps", fail_if_called)

    result = main_module.main(
        dataset_path=str(dataset_path),
        user_question=(
            "Subscribers have a higher and lower average balance "
            "than non-subscribers."
        ),
    )

    output = capsys.readouterr().out

    assert result is None
    assert "CLARIFICATION REQUIRED" in output
    assert "higher" in output.lower()
    assert "lower" in output.lower()
    assert "No EDA plan was generated" in output
