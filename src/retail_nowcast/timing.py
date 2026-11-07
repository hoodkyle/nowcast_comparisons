from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Sequence, Mapping
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .factors import expanding_block_scores, source_wide


def _allowed_official(vintage: str) -> set[tuple[str, str]]:
    base = {("marts", "lag1_official"), ("mrts", "lag2_official")}
    base.add(("marts", "current_official"))
    if vintage == "v3_full_national":
        base.add(("mrts", "current_official"))
    return base

def validate_vintage_panels(state: pd.DataFrame, industry: pd.DataFrame, vintage: str) -> None:
    if vintage not in {"v2_advance", "v3_full_national"}:
        raise ValueError("Unknown information stage")
    required = {"source", "date", "reference_month", "concept", "timing_variant", "delta_yoy"}
    for name, panel in (("state", state), ("industry", industry)):
        if missing := required - set(panel):
            raise ValueError(f"{name} panel missing fields: {sorted(missing)}")
        dates = pd.to_datetime(panel["date"]).dt.to_period("M")
        references = pd.to_datetime(panel["reference_month"]).dt.to_period("M")
        if (references > dates).any():
            raise ValueError(f"{name} panel contains future reference months")
        if panel.duplicated(["source", "concept", "timing_variant", *( ["state"] if "state" in panel else []), "date"]).any():
            raise ValueError(f"{name} panel contains duplicate signal keys")
        available_column = "available_v2" if vintage == "v2_advance" else "available_v3"
        if available_column in panel and not panel[available_column].all():
            raise ValueError(f"{name} panel contains unavailable {vintage} rows")
        current_msrs = panel["source"].eq("msrs") & references.eq(dates)
        if current_msrs.any():
            raise ValueError("Current MSRS is forbidden")
    observed_official = set(
        industry.loc[industry["source"].isin({"marts", "mrts"}), ["source", "timing_variant"]]
        .drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_official != _allowed_official(vintage):
        raise ValueError(f"Official timing set differs from {vintage} contract")
    periods = pd.to_datetime(industry["date"]).dt.to_period("M")
    references = pd.to_datetime(industry["reference_month"]).dt.to_period("M")
    lag = pd.Series([(date - reference).n for date, reference in zip(periods, references, strict=True)], index=industry.index)
    expected = {
        ("marts", "current_official"): 0, ("marts", "lag1_official"): 1,
        ("mrts", "current_official"): 0, ("mrts", "lag2_official"): 2,
    }
    for key, months in expected.items():
        selected = industry["source"].eq(key[0]) & industry["timing_variant"].eq(key[1])
        if selected.any() and not lag.loc[selected].eq(months).all():
            raise ValueError(f"{key} reference-month lag mismatch")

def _state_blocks(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        f"{source}_{concept}": group.pivot(index="date", columns="state", values="delta_yoy")
        for (source, concept), group in panel.groupby(["source", "concept"])
    }

def _industry_blocks(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    blocks = {}
    for source, group in panel.groupby("source"):
        selected = group.copy()
        if source in {"marts", "mrts"}:
            selected["concept"] = selected["timing_variant"] + ":" + selected["concept"].astype(str)
        blocks[str(source)] = source_wide(selected, prefix=f"{source}:")
    return blocks

def build_vintage_feature_store(
    target: pd.DataFrame, state_panel: pd.DataFrame, industry_panel: pd.DataFrame, vintage: str,
) -> pd.DataFrame:
    """Construct later information sets from their corresponding indicator panels."""
    validate_vintage_panels(state_panel, industry_panel, vintage)
    base = target[["state", "date", "z", "lag_yoy", "lag_delta"]].copy()
    base["date"] = pd.to_datetime(base["date"]).dt.to_period("M").dt.to_timestamp()
    state = state_panel.copy()
    state["feature"] = "state__" + state["source"].astype(str) + "__" + state["concept"].astype(str)
    state_wide = state.pivot_table(index=["state", "date"], columns="feature", values="delta_yoy", aggfunc="first")
    state["cross_feature"] = (
        "crossstate__" + state["source"].astype(str) + "__" + state["concept"].astype(str)
        + "__" + state["state"].astype(str)
    )
    cross_wide = state.pivot_table(index="date", columns="cross_feature", values="delta_yoy", aggfunc="first")
    industry = industry_panel.copy()
    concept = industry["concept"].astype(str)
    official = industry["source"].isin({"marts", "mrts"})
    concept = concept.where(~official, industry["timing_variant"].astype(str) + ":" + concept)
    industry["feature"] = "industry__" + industry["source"].astype(str) + "__" + concept
    industry_wide = industry.pivot(index="date", columns="feature", values="delta_yoy")
    state_pca = expanding_block_scores(_state_blocks(state), "pca_state").rename_axis("date")
    industry_pca = expanding_block_scores(_industry_blocks(industry_panel), "pca_industry").rename_axis("date")
    result = base.merge(state_wide.reset_index(), on=["state", "date"], how="left", validate="one_to_one")
    result = result.merge(cross_wide.reset_index(), on="date", how="left", validate="many_to_one")
    result = result.merge(industry_wide.reset_index(), on="date", how="left", validate="many_to_one")
    result = result.merge(state_pca.reset_index(), on="date", how="left", validate="many_to_one")
    result = result.merge(industry_pca.reset_index(), on="date", how="left", validate="many_to_one")
    value_columns = [column for column in result if column.startswith(("state__", "crossstate__", "industry__"))]
    masks = pd.DataFrame(
        {f"available__{column}": result[column].notna().astype("int8") for column in value_columns},
        index=result.index,
    )
    return pd.concat([result, masks], axis=1).sort_values(["date", "state"]).reset_index(drop=True)
