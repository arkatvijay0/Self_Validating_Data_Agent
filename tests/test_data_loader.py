
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import profile_dataset


def test_profile_dataset_reports_basic_statistics(tmp_path):
    csv_path = tmp_path / "test_data.csv"

    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["A", "A", "B"],
        }
    )
    df.to_csv(csv_path, index=False)

    result = profile_dataset(csv_path)

    assert result["n_rows"] == 3
    assert result["n_columns"] == 2
    assert "age" in result["columns"]
    assert result["columns"]["age"]["mean"] == 30.0
    assert result["columns"]["age"]["min"] == 20.0
    assert result["columns"]["age"]["max"] == 40.0
    assert result["columns"]["city"]["role"] == "text_or_id"
