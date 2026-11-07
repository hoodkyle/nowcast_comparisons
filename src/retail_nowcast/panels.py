from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Sequence, Mapping
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .factors import expanding_block_scores


SCOPES = (
    "own_state", "pca_state", "pca_industry", "pca_both", "all_state",
    "related_industry", "all_industry", "private_only", "inclusive",
)

def validate_v1_panel(panel: pd.DataFrame, *, mrts_lag: int = 2) -> None:
    if mrts_lag not in {1, 2}:
        raise ValueError("MRTS lag must be one or two months")
    required = {"source", "date", "reference_month", "timing_variant"}
    if missing := required - set(panel):
        raise ValueError(f"V1 panel is missing columns: {sorted(missing)}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M")
    frame["reference_month"] = pd.to_datetime(frame["reference_month"]).dt.to_period("M")
    if (frame["reference_month"] > frame["date"]).any():
        raise ValueError("Future information found in V1 panel")
    lag = (frame["date"] - frame["reference_month"]).apply(lambda value: value.n)
    expected = {"marts": 1, "mrts": mrts_lag, "msrs": 1}
    for source, months in expected.items():
        selected = frame["source"].eq(source)
        if selected.any() and not lag.loc[selected].eq(months).all():
            raise ValueError(f"{source} violates the frozen V1 lag of {months} month(s)")
    private = frame["source"].isin({"fiserv", "gamma"})
    if private.any() and not lag.loc[private].eq(0).all():
        raise ValueError("Private V1 signals must be contemporaneous where observed")
    forbidden = frame["timing_variant"].astype(str).str.contains("current_official", case=False)
    if forbidden.any():
        raise ValueError("Current official target information is forbidden in V1")

def build_feature_store(target: pd.DataFrame, state_v1: pd.DataFrame, industry_v1: pd.DataFrame, *, mrts_lag: int = 2) -> pd.DataFrame:
    """Construct early-information predictors and availability indicators."""
    validate_v1_panel(state_v1, mrts_lag=mrts_lag)
    validate_v1_panel(industry_v1, mrts_lag=mrts_lag)
    base = target[["state", "date", "z", "lag_yoy", "lag_delta"]].copy()
    base["date"] = pd.to_datetime(base["date"]).dt.to_period("M").dt.to_timestamp()
    state = state_v1.copy()
    state["feature"] = "state__" + state["source"].astype(str) + "__" + state["concept"].astype(str)
    sw = state.pivot_table(index=["state", "date"], columns="feature", values="delta_yoy", aggfunc="first")
    state["cross_feature"] = (
        "crossstate__" + state["source"].astype(str) + "__"
        + state["concept"].astype(str) + "__" + state["state"].astype(str)
    )
    cw = state.pivot_table(index="date", columns="cross_feature", values="delta_yoy", aggfunc="first")
    iw = industry_v1.copy()
    iw["feature"] = "industry__" + iw["source"].astype(str) + "__" + iw["concept"].astype(str)
    iw = iw.pivot_table(index="date", columns="feature", values="delta_yoy", aggfunc="first")
    state_blocks = {
        f"{source}_{concept}": group.pivot(index="date", columns="state", values="delta_yoy")
        for (source, concept), group in state.groupby(["source", "concept"])
        if not group.duplicated(["date", "state"]).any()
    }
    industry_blocks = {
        str(source): group.pivot(index="date", columns="concept", values="delta_yoy")
        for source, group in industry_v1.groupby("source")
        if not group.duplicated(["date", "concept"]).any()
    }
    state_pca = expanding_block_scores(state_blocks, "pca_state").rename_axis("date")
    industry_pca = expanding_block_scores(industry_blocks, "pca_industry").rename_axis("date")
    result = base.merge(sw.reset_index(), on=["state", "date"], how="left", validate="one_to_one")
    result = result.merge(cw.reset_index(), on="date", how="left", validate="many_to_one")
    result = result.merge(iw.reset_index(), on="date", how="left", validate="many_to_one")
    result = result.merge(state_pca.reset_index(), on="date", how="left", validate="many_to_one")
    result = result.merge(industry_pca.reset_index(), on="date", how="left", validate="many_to_one")
    value_columns = [column for column in result if column.startswith(("state__", "crossstate__", "industry__"))]
    masks = pd.DataFrame(
        {f"available__{column}": result[column].notna().astype("int8") for column in value_columns},
        index=result.index,
    )
    result = pd.concat([result, masks], axis=1)
    return result.sort_values(["date", "state"]).reset_index(drop=True)

def scope_columns(store: pd.DataFrame, scope: str) -> list[str]:
    if scope not in SCOPES:
        raise ValueError(f"Unknown feature scope: {scope}")
    state = [c for c in store if c.startswith("state__") or c.startswith("available__state__")]
    crossstate = [c for c in store if c.startswith("crossstate__") or c.startswith("available__crossstate__")]
    industry = [c for c in store if c.startswith("industry__") or c.startswith("available__industry__")]
    private = [c for c in state + crossstate + industry if any(f"__{s}__" in c for s in ("fiserv", "gamma"))]
    related = [c for c in industry if any(token in c for token in ("TOTAL", "RSG_no_454", "mapped_retail_total"))]
    pca_state = [c for c in store if c.startswith("pca_state_")]
    pca_industry = [c for c in store if c.startswith("pca_industry_")]
    mapping = {
        "own_state": state, "pca_state": pca_state, "pca_industry": pca_industry,
        "pca_both": pca_state + pca_industry, "all_state": crossstate,
        "related_industry": related, "all_industry": industry,
        "private_only": private, "inclusive": state + crossstate + industry,
    }
    return ["lag_yoy", "lag_delta", *mapping[scope]]
