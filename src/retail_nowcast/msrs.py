"""Parser for official Census MSRS CSV files."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .states import normalize_state, state_to_fips

MSRS_SOURCE = "msrs"
TOTAL_CATEGORY = "TOTAL"
CATEGORY_LABELS = {
    "TOTAL": "Total Retail Sales excluding Nonstore Retailers",
    "441": "Motor Vehicle and Parts Dealers",
    "442": "Furniture and Home Furnishings Stores",
    "443": "Electronics and Appliance Stores",
    "444": "Building Material and Garden Equipment and Supplies Dealers",
    "445": "Food and Beverage Stores",
    "446": "Health and Personal Care Stores",
    "447": "Gasoline Stations",
    "448": "Clothing and Clothing Accessories Stores",
    "451": "Sporting Goods, Hobby, Musical Instrument, and Book Stores",
    "452": "General Merchandise Stores",
    "453": "Miscellaneous Store Retailers",
}
ID_COLUMNS = ("fips", "stateabbr", "naics")


def _read_wide(path: str | Path, prefix: str, value_name: str, numeric: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    missing = set(ID_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"MSRS file is missing identifier columns: {sorted(missing)}")
    duplicates = frame.duplicated(list(ID_COLUMNS), keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, list(ID_COLUMNS)].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate MSRS state/category rows: {keys}")
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{6}})$")
    month_columns = [column for column in frame.columns if pattern.fullmatch(column)]
    unknown = set(frame.columns) - set(ID_COLUMNS) - set(month_columns)
    if unknown:
        raise ValueError(f"Unexpected MSRS columns: {sorted(unknown)}")
    if not month_columns:
        raise ValueError(f"No {prefix}<YYYYMM> columns found")
    long = frame.melt(id_vars=list(ID_COLUMNS), value_vars=month_columns, var_name="period", value_name="_raw")
    periods = long["period"].str.extract(pattern, expand=False)
    try:
        long["date"] = pd.to_datetime(periods, format="%Y%m")
    except ValueError as exc:
        raise ValueError("Invalid YYYYMM month in MSRS columns") from exc
    long[f"{value_name}_suppressed"] = long["_raw"].eq("S")
    if numeric:
        values = pd.to_numeric(long["_raw"].mask(long[f"{value_name}_suppressed"]), errors="coerce")
        invalid = long["_raw"].notna() & ~long[f"{value_name}_suppressed"] & values.isna()
        if invalid.any():
            tokens = sorted(long.loc[invalid, "_raw"].astype(str).unique().tolist())
            raise ValueError(f"Unexpected nonnumeric MSRS values in {value_name}: {tokens}")
        long[value_name] = values.astype(float)
    else:
        long[value_name] = long["_raw"].mask(long[f"{value_name}_suppressed"])
    return long.drop(columns=["period", "_raw"])


def load_msrs(
    yoy_path: str | Path,
    standard_errors_path: str | Path | None = None,
    coverage_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load official wide MSRS files into a canonical monthly panel.

    Published YoY values are used directly. Suppressions remain missing and are
    explicitly flagged; values are never interpolated or forward-filled.
    """
    panel = _read_wide(yoy_path, "yy", "value", numeric=True).rename(
        columns={"value_suppressed": "suppressed"}
    )
    join_keys = [*ID_COLUMNS, "date"]
    if standard_errors_path is not None:
        errors = _read_wide(standard_errors_path, "se", "standard_error", numeric=True)
        panel = panel.merge(errors, on=join_keys, how="left", validate="one_to_one")
    if coverage_path is not None:
        coverage = _read_wide(coverage_path, "cov", "coverage", numeric=False)
        panel = panel.merge(coverage, on=join_keys, how="left", validate="one_to_one")
    panel = panel.rename(columns={"stateabbr": "state", "naics": "category"})
    panel.insert(0, "source", MSRS_SOURCE)
    panel["state_fips"] = panel["fips"]
    state_rows = ~panel["state"].eq("USA")
    panel.loc[state_rows, "state"] = panel.loc[state_rows, "state"].map(normalize_state)
    expected_fips = panel.loc[state_rows, "state"].map(state_to_fips)
    if not expected_fips.eq(panel.loc[state_rows, "state_fips"]).all():
        raise ValueError("MSRS state abbreviations and FIPS codes disagree with canonical crosswalk")
    panel["category_label"] = panel["category"].map(CATEGORY_LABELS).fillna("Unknown MSRS category")
    keys = ["source", "state", "category", "date"]
    if panel.duplicated(keys, keep=False).any():
        raise ValueError("Duplicate source/state/category/month observations after parsing")
    return panel.sort_values(keys).reset_index(drop=True)
