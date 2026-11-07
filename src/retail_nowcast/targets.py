from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Sequence, Mapping
from dataclasses import dataclass
import numpy as np
import pandas as pd


def category_delta_targets(msrs: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    """Derive exact-calendar category delta-YoY without bridging missing months."""
    frame = msrs.loc[
        ~msrs["state"].eq("USA") & msrs["category"].isin(categories),
        ["state", "category", "date", "value"],
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    if frame.duplicated(["state", "category", "date"]).any():
        raise ValueError("Duplicate MSRS category target key")
    lag = frame.rename(columns={"value": "lag_yoy"}).copy()
    lag["date"] = lag["date"] + pd.DateOffset(months=1)
    frame = frame.merge(
        lag[["state", "category", "date", "lag_yoy"]],
        on=["state", "category", "date"], how="left", validate="one_to_one",
    )
    frame["z"] = frame["value"] - frame["lag_yoy"]
    prior = frame[["state", "category", "date", "z"]].rename(columns={"z": "lag_delta"})
    prior["date"] = prior["date"] + pd.DateOffset(months=1)
    frame = frame.merge(
        prior, on=["state", "category", "date"], how="left", validate="one_to_one"
    )
    return frame.sort_values(["category", "date", "state"]).reset_index(drop=True)
