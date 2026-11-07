"""Calendar-safe two-token features for the main comparison."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping
import numpy as np
import pandas as pd
from .panels import scope_columns

KEYS = ["state", "category", "date"]
FORBIDDEN = frozenset({"target", "value", "z"})

def month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.to_period("M").dt.to_timestamp()

def validate_keys(frame: pd.DataFrame, keys: Iterable[str] = KEYS) -> None:
    keys = list(keys)
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f"missing feature keys: {sorted(missing)}")
    if frame.duplicated(keys).any():
        raise ValueError("duplicate feature keys")

def calendar_lag_features(frame: pd.DataFrame, value_columns: list[str], lags: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    source = frame.copy()
    source["date"] = month_start(source["date"])
    validate_keys(source)
    if not value_columns or set(value_columns) - set(source.columns):
        raise ValueError("feature columns are missing")
    result = source[KEYS].copy()
    for lag in lags:
        if lag < 0:
            raise ValueError("lags must be nonnegative calendar offsets")
        lookup = source[KEYS + value_columns].copy()
        lookup["date"] = lookup["date"] + pd.DateOffset(months=lag)
        lookup = lookup.rename(columns={c: f"{c}_lag{lag}" for c in value_columns})
        result = result.merge(lookup, on=KEYS, how="left", validate="one_to_one")
    return result

@dataclass(frozen=True)
class OriginFrames:
    train: pd.DataFrame
    score: pd.DataFrame
    feature_columns: tuple[str, ...]

@dataclass(frozen=True)
class OriginDataset:
    train: Mapping[str, object]
    score: Mapping[str, object]
    channels: tuple[str, ...]
    training_start: pd.Timestamp | None
    n_train: int
    n_score: int
    @property
    def x(self) -> np.ndarray:
        return self.train["x"]  # type: ignore[return-value]
    @property
    def y(self) -> np.ndarray:
        return self.train["y"]  # type: ignore[return-value]

def select_training_columns(frame: pd.DataFrame, candidates: list[str]) -> tuple[str, ...]:
    chosen: list[str] = []
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if len(finite):
            chosen.append(column)
    return tuple(chosen)

def split_origin(frame: pd.DataFrame, origin: str | pd.Timestamp, candidates: list[str], target: str = "target") -> OriginFrames:
    source = frame.copy()
    source["date"] = month_start(source["date"])
    validate_keys(source)
    cutoff = pd.Timestamp(origin).to_period("M").to_timestamp()
    if target not in source:
        raise ValueError(f"missing target column {target}")
    train = source.loc[source["date"] < cutoff].copy()
    score = source.loc[source["date"].eq(cutoff)].copy()
    features = select_training_columns(train, [c for c in candidates if c not in FORBIDDEN])
    return OriginFrames(train=train, score=score, feature_columns=features)

def _two_tokens(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    base = frame[KEYS + columns].copy()
    base["date"] = month_start(base["date"])
    validate_keys(base)
    prior = base.copy()
    prior["date"] = prior["date"] + pd.DateOffset(months=1)
    prior = prior.rename(columns={c: f"__prior__{c}" for c in columns})
    joined = base[KEYS].merge(prior, on=KEYS, how="left", validate="one_to_one")
    current = base[KEYS + columns].rename(columns={c: f"__current__{c}" for c in columns})
    joined = joined.merge(current, on=KEYS, how="left", validate="one_to_one")
    prior_x = joined[[f"__prior__{c}" for c in columns]].to_numpy(float)
    current_x = joined[[f"__current__{c}" for c in columns]].to_numpy(float)
    return joined[KEYS], np.stack([prior_x, current_x], axis=1)

def build_origin_dataset(frame: pd.DataFrame, origin: str | pd.Timestamp, category: str, *, scope: str = "all_industry", target: str = "target") -> OriginDataset:
    source = frame.copy()
    source["date"] = month_start(source["date"])
    validate_keys(source)
    if category not in set(source["category"].astype(str)):
        raise ValueError(f"category {category!r} is absent")
    if target not in source:
        raise ValueError(f"missing feature target column {target}")
    source = source.loc[source["category"].astype(str).eq(category)].copy()
    candidates = [c for c in scope_columns(source, scope) if c in source.columns and c not in FORBIDDEN]
    if not candidates:
        raise ValueError("no canonical predictor columns available")
    keys, raw_x = _two_tokens(source, candidates)
    cutoff = pd.Timestamp(origin).to_period("M").to_timestamp()
    train_ix = np.flatnonzero((keys["date"] < cutoff).to_numpy())
    score_ix = np.flatnonzero((keys["date"] == cutoff).to_numpy())
    channels: list[str] = []
    for j, column in enumerate(candidates):
        keep = True
        for token in (0, 1):
            values = raw_x[train_ix, token, j]
            finite = values[np.isfinite(values)]
            keep = keep and len(finite) > 0 and np.ptp(finite) > 0
        if keep:
            channels.append(column)
    channel_ix = [candidates.index(c) for c in channels]
    indexed = source.set_index(KEYS)[target]
    def targets(ix: np.ndarray) -> np.ndarray:
        if not len(ix):
            return np.empty(0, float)
        return indexed.reindex(pd.MultiIndex.from_frame(keys.iloc[ix][KEYS])).to_numpy(float)
    train_target = targets(train_ix)
    train_x_all = raw_x[train_ix][:, :, channel_ix]
    score_x = raw_x[score_ix][:, :, channel_ix]
    common = np.isfinite(train_x_all).all(axis=(1, 2)) & np.isfinite(train_target)
    train_x = train_x_all[common]
    train_y = train_target[common]
    score_eligible = np.isfinite(score_x).all(axis=(1, 2))
    score_target = targets(score_ix)
    if not channels:
        raise ValueError("no training-selected predictor channels")
    if not len(train_y):
        raise ValueError("no usable training rows")
    if not len(score_ix):
        raise ValueError("no score keys at origin")
    train_keys = keys.iloc[train_ix].loc[common].reset_index(drop=True)
    score_keys = keys.iloc[score_ix].reset_index(drop=True)
    training_start = train_keys["date"].min() if len(train_keys) else None
    return OriginDataset(
        train={"keys": train_keys, "x": train_x, "y": train_y, "mask": common, "states": train_keys["state"].to_numpy()},
        score={"keys": score_keys, "x": score_x, "actual": score_target, "eligible": score_eligible, "states": score_keys["state"].to_numpy()},
        channels=tuple(channels), training_start=training_start, n_train=len(train_y), n_score=len(score_keys),
    )

build_features = build_origin_dataset
