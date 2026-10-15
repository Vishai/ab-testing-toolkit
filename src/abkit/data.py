"""Loading and normalising experiment data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .scenarios import Scenario, get

REQUIRED_COLUMNS = (
    "SessionID",
    "AB_Group",
    "Device",
    "UserType",
    "PageViewed",
    "Conversion",
    "TimeOnPage_sec",
    "Bounce",
    "Revenue",
)

GROUP_MAP = {"A_Control": "A", "B_Variation": "B"}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load(scenario: str | Scenario, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a scenario's sessions and add analysis-friendly columns.

    Adds:
      group     - "A" or "B"
      treatment - 0 for control, 1 for variation
    """
    sc = get(scenario) if isinstance(scenario, str) else scenario
    path = Path(data_dir or DEFAULT_DATA_DIR) / sc.file
    df = pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    unknown = set(df["AB_Group"].unique()) - set(GROUP_MAP)
    if unknown:
        raise ValueError(f"Unexpected AB_Group labels: {sorted(unknown)}")

    df = df.copy()
    df["group"] = df["AB_Group"].map(GROUP_MAP)
    df["treatment"] = (df["group"] == "B").astype(int)
    df["Revenue"] = df["Revenue"].astype(float)
    return df
