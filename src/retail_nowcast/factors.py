from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Sequence, Mapping
from dataclasses import dataclass
import numpy as np
import pandas as pd


def source_wide(frame: pd.DataFrame, *, value: str = "delta_yoy", prefix: str = "") -> pd.DataFrame:
    data = frame.loc[np.isfinite(frame[value])].copy()
    wide = data.pivot(index="date", columns="concept", values=value).sort_index()
    wide.columns = [f"{prefix}{column}" for column in wide.columns]
    return wide

def expanding_first_factor(wide: pd.DataFrame, *, minimum_dates: int = 12) -> pd.Series:
    if wide.empty:
        raise ValueError("Factor block is empty")
    values = []
    for date in wide.index.sort_values():
        current = wide.loc[date]
        history = wide.loc[wide.index < date]
        candidates = [
            column for column in wide.columns
            if pd.notna(current[column]) and history[column].notna().sum() >= minimum_dates
        ]
        if not candidates:
            continue
        block = history[candidates].dropna(axis=0, how="any")
        if len(block) < minimum_dates:
            continue
        x = block.to_numpy(float)
        mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
        keep = scale > 0
        if not keep.any():
            continue
        z = (x[:, keep] - mean[keep]) / scale[keep]
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        anchor = int(np.argmax(np.abs(vt[0])))
        sign = -1.0 if vt[0, anchor] < 0 else 1.0
        projected = ((current[candidates].to_numpy(float)[keep] - mean[keep]) / scale[keep]) @ vt[0]
        values.append((date, float(sign * projected)))
    return pd.Series(dict(values), name="F1").sort_index()

def expanding_block_scores(blocks: dict[str, pd.DataFrame], prefix: str) -> pd.DataFrame:
    scores = []
    for name, wide in sorted(blocks.items()):
        try:
            score = expanding_first_factor(wide)
        except ValueError:
            continue
        score.name = f"{prefix}_{name}_F1"
        scores.append(score)
    if not scores:
        raise ValueError("No factor block supplied an expanding score")
    return pd.concat(scores, axis=1).sort_index()

def _ridge_panel_predict(
    train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], *, ridge: float = 1.0
) -> np.ndarray:
    states = sorted(train["state"].unique())
    baseline = states[0]
    dummies_train = np.column_stack([(train["state"] == state).to_numpy(float) for state in states[1:]])
    dummies_test = np.column_stack([(test["state"] == state).to_numpy(float) for state in states[1:]])
    raw_train = train[list(features)].to_numpy(float) if features else np.empty((len(train), 0))
    raw_test = test[list(features)].to_numpy(float) if features else np.empty((len(test), 0))
    if features:
        mean, scale = raw_train.mean(axis=0), raw_train.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        raw_train, raw_test = (raw_train - mean) / scale, (raw_test - mean) / scale
    x = np.column_stack([np.ones(len(train)), dummies_train, raw_train])
    xt = np.column_stack([np.ones(len(test)), dummies_test, raw_test])
    penalty = np.zeros(x.shape[1])
    penalty[1 + len(states) - 1 :] = ridge
    beta = np.linalg.pinv(x.T @ x + np.diag(penalty)) @ x.T @ train["z"].to_numpy(float)
    del baseline
    return xt @ beta
