"""Version-controlled state identity and stable embedding IDs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import pandas as pd

STATE_CROSSWALK_VERSION = "us_states_dc_v1"


@lru_cache(maxsize=1)
def _crosswalk() -> pd.DataFrame:
    frame = pd.read_csv(
        Path(__file__).with_name("state_crosswalk.csv"),
        dtype={"state_id": int, "state": str, "state_fips": str, "state_name": str},
    )
    if len(frame) != 51 or frame[["state_id", "state", "state_fips", "state_name"]].nunique().ne(51).any():
        raise RuntimeError("State crosswalk must contain 51 unique modeling geographies")
    if frame["state_id"].tolist() != list(range(51)):
        raise RuntimeError("State IDs must be the contiguous, versioned range 0..50")
    if frame["state"].tolist() != sorted(frame["state"]):
        raise RuntimeError("State IDs are frozen by alphabetical postal abbreviation")
    return frame


def state_crosswalk() -> pd.DataFrame:
    """Return a defensive copy of the 50-state-plus-DC crosswalk."""
    return _crosswalk().copy()


def normalize_state(identifier: str) -> str:
    """Return a postal abbreviation from a postal code, 2-digit FIPS, or full name.

    Matching is case-insensitive for postal abbreviations and names. FIPS must
    remain two-digit text; malformed or unrecognized identifiers fail.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("State identifier must be nonempty text")
    token = identifier.strip()
    frame = _crosswalk()
    if re.fullmatch(r"\d{2}", token):
        matches = frame.loc[frame["state_fips"].eq(token), "state"]
    elif re.fullmatch(r"[A-Za-z]{2}", token):
        matches = frame.loc[frame["state"].eq(token.upper()), "state"]
    else:
        matches = frame.loc[frame["state_name"].str.casefold().eq(token.casefold()), "state"]
    if len(matches) != 1:
        raise ValueError(f"Unrecognized or ambiguous state identifier: {identifier!r}")
    return str(matches.iloc[0])


def state_to_fips(identifier: str) -> str:
    state = normalize_state(identifier)
    return str(_crosswalk().set_index("state").at[state, "state_fips"])


def fips_to_state(fips: str) -> str:
    if not isinstance(fips, str) or not re.fullmatch(r"\d{2}", fips.strip()):
        raise ValueError("FIPS identifier must be two-digit text")
    return normalize_state(fips.strip())


def state_to_id(identifier: str) -> int:
    state = normalize_state(identifier)
    return int(_crosswalk().set_index("state").at[state, "state_id"])
