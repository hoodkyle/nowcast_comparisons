"""Forecasting specifications using common two-month estimation support."""
from __future__ import annotations

from typing import Any
import warnings
import numpy as np
import pandas as pd

from .features import OriginDataset




def _state_effects(states: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.column_stack([states == level for level in levels[1:]]).astype(float) if len(levels) > 1 else np.empty((len(states), 0))


def _masked(ds: OriginDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tx = np.asarray(ds.train["x"], float)
    y = np.asarray(ds.train["y"], float)
    qx = np.asarray(ds.score["x"], float)
    eligible = np.asarray(ds.score["eligible"], bool)
    if tx.ndim != 3 or qx.ndim != 3 or len(tx) != len(y):
        raise ValueError("malformed shared dataset tensors")
    if not np.isfinite(tx).all() or not np.isfinite(y).all():
        raise ValueError("training dataset contains nonfinite values after shared mask")
    return tx, y, qx, eligible


def _standardize(x: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x - mean) / scale, (q - mean) / scale, mean, scale


def _ridge_unpenalized(x: np.ndarray, y: np.ndarray, q: np.ndarray, alpha: float,
                       penalized: int) -> np.ndarray:
    """Ridge with an explicit unpenalized intercept/state block."""
    penalty = np.eye(x.shape[1]) * alpha
    penalty[penalized:, penalized:] = 0.0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return q @ coef


def _state_aug(ds: OriginDataset, x: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_states = np.asarray(ds.train["states"], str)
    score_states = np.asarray(ds.score["states"], str)
    levels = np.unique(train_states)
    return (np.column_stack([x, _state_effects(train_states, levels), np.ones(len(x))]),
            np.column_stack([q, _state_effects(score_states, levels), np.ones(len(q))]), levels)


def _random_walk(ds: OriginDataset) -> np.ndarray:
    # RW0 is the frozen zero-delta benchmark.  It has no fitted parameters.
    out = np.zeros(len(ds.score["states"]), dtype=float)
    out[~np.asarray(ds.score["eligible"], bool)] = np.nan
    return out


def _tree_predict(ds: OriginDataset, spec: dict[str, Any], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    tx, y, qx, eligible = _masked(ds)
    x = tx.reshape(len(tx), -1)
    q = qx.reshape(len(qx), -1)
    x, q, _, _ = _standardize(x, q)
    x, q, _ = _state_aug(ds, x, q)
    family = spec["family"]
    if family == "ols":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif family in {"ridge", "pca_ridge"}:
        from sklearn.linear_model import Ridge
        if family == "pca_ridge":
            from sklearn.decomposition import PCA
            numeric_width = 2 * tx.shape[2]
            pca = PCA(n_components=min(int(spec.get("components", 2)), numeric_width), random_state=seed).fit(x[:, :numeric_width])
            x = np.column_stack([pca.transform(x[:, :2 * tx.shape[2]]), x[:, 2 * tx.shape[2]:]])
            q_pca = np.full((len(q), pca.n_components_), np.nan)
            if eligible.any(): q_pca[eligible] = pca.transform(q[eligible, :numeric_width])
            q = np.column_stack([q_pca, q[:, numeric_width:]])
            model = None
        if family == "pca_ridge":
            pred = np.full(len(q), np.nan)
            if eligible.any(): pred[eligible] = _ridge_unpenalized(x, y, q[eligible], float(spec.get("alpha", 1.0)), pca.n_components_)
            return pred, {"rows": len(y), "features": int(x.shape[1]), "components": int(pca.n_components_)}
        model = Ridge(alpha=float(spec.get("alpha", 1.0)))
    elif family == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=float(spec["alpha"]), max_iter=100000, random_state=seed)
    elif family == "elastic_net":
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=float(spec["alpha"]), l1_ratio=float(spec["l1_ratio"]), max_iter=100000, random_state=seed)
    elif family == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=int(spec["n_estimators"]), max_depth=int(spec["max_depth"]), min_samples_leaf=int(spec["min_samples_leaf"]), max_features=spec["max_features"], n_jobs=1, random_state=seed)
    elif family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=int(spec["max_iter"]), max_leaf_nodes=int(spec["max_leaf_nodes"]), l2_regularization=float(spec["l2_regularization"]), random_state=seed)
    elif family == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=int(spec["iterations"]), depth=int(spec["depth"]), learning_rate=float(spec["learning_rate"]), l2_leaf_reg=float(spec["l2_leaf_reg"]), random_seed=seed, verbose=False, task_type="GPU", gpu_ram_part=float(spec.get("gpu_ram_part", .5)), devices=str(spec.get("devices", "0")), allow_writing_files=False)
    elif family == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=int(spec["n_estimators"]), max_depth=int(spec["max_depth"]), learning_rate=float(spec["learning_rate"]), min_child_weight=float(spec["min_child_weight"]), reg_lambda=float(spec["reg_lambda"]), subsample=float(spec["subsample"]), colsample_bytree=float(spec["colsample_bytree"]), random_state=seed, n_jobs=1, tree_method="hist", device=str(spec.get("device", "cuda")))
    else:
        raise ValueError(f"unsupported conventional family: {family}")
    solver_attempts = []
    if family in {"lasso", "elastic_net"}:
        from sklearn.exceptions import ConvergenceWarning
        tolerance = float(getattr(model, "tol", 1e-4))
        gap_limit = tolerance * float(np.var(y))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        first_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
        first_n_iter = int(np.max(np.asarray(model.n_iter_)))
        first_gap = float(np.max(np.asarray(model.dual_gap_)))
        first_finite = bool(np.isfinite(np.asarray(model.coef_, dtype=float)).all()
                            and np.isfinite(np.asarray(model.intercept_, dtype=float)).all()
                            and np.isfinite(first_gap))
        solver_attempts.append({"max_iter": 100000, "n_iter": first_n_iter,
                                "dual_gap": first_gap, "gap_limit": gap_limit,
                                "warning": first_warning, "finite": first_finite})
        if first_warning or not first_finite or first_gap > gap_limit:
            # Continue the already fitted coordinate-descent path.  Keeping the
            # first call unchanged preserves predictions for all fits
            # that converged under the 100,000-iteration limit.
            model.set_params(max_iter=1_000_000, warm_start=True, precompute=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x, y)
            retry_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            retry_n_iter = int(np.max(np.asarray(model.n_iter_)))
            retry_gap = float(np.max(np.asarray(model.dual_gap_)))
            retry_finite = bool(np.isfinite(np.asarray(model.coef_, dtype=float)).all()
                                and np.isfinite(np.asarray(model.intercept_, dtype=float)).all()
                                and np.isfinite(retry_gap))
            solver_attempts.append({"max_iter": 1_000_000, "n_iter": retry_n_iter,
                                    "dual_gap": retry_gap, "gap_limit": gap_limit,
                                    "warning": retry_warning, "finite": retry_finite,
                                    "warm_start": True, "precompute": True})
            if retry_warning or not retry_finite or retry_gap > gap_limit:
                raise RuntimeError(
                    f"{family} solver did not converge after 100000 and 1000000 iterations "
                    f"(dual_gap={retry_gap!r}, limit={gap_limit!r})"
                )
    else:
        model.fit(x, y)
    actual_device = "cuda" if family == "catboost" else "cpu"
    if family == "xgboost":
        requested = str(spec.get("device", "cuda"))
        if requested != "cuda":
            raise ValueError("XGBoost configuration must request CUDA")
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("XGBoost CUDA configuration requested but CUDA is unavailable")
            config = model.get_booster().save_config()
            if '"device":"cuda"' not in config and '"device":"cuda:0"' not in config:
                raise RuntimeError("XGBoost did not save a CUDA device configuration")
            actual_device = "cuda"
        except AttributeError as exc:
            raise RuntimeError("XGBoost booster configuration unavailable") from exc
    pred = np.full(len(q), np.nan)
    if eligible.any():
        pred[eligible] = model.predict(q[eligible])
    metadata = {"rows": len(y), "features": int(x.shape[1]), "device": actual_device}
    if family in {"lasso", "elastic_net"}:
        metadata.update({"solver_max_iter": int(solver_attempts[-1]["max_iter"]),
                         "solver_n_iter": int(np.max(np.asarray(model.n_iter_))),
                         "solver_dual_gap": float(np.max(np.asarray(model.dual_gap_))),
                         "solver_attempts": solver_attempts})
    return pred, metadata


def _nn_predict(ds: OriginDataset, spec: dict[str, Any], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    tx, y, qx, eligible = _masked(ds)
    flat = tx.reshape(len(tx), -1)
    qflat = qx.reshape(len(qx), -1)
    flat, qflat, xm, xs = _standardize(flat, qflat)
    ym, ys = y.mean(), max(float(y.std()), 1e-12)
    yt = torch.tensor(((y - ym) / ys)[:, None], dtype=torch.float32)
    levels = np.unique(np.asarray(ds.train["states"], str))
    tr_code = {s: i for i, s in enumerate(levels)}
    tc = torch.tensor([tr_code[s] for s in np.asarray(ds.train["states"], str)], dtype=torch.long)
    qc = torch.tensor([tr_code.get(s, -1) for s in np.asarray(ds.score["states"], str)], dtype=torch.long)
    class Net(torch.nn.Module):
        def __init__(self, c, nstates, transformer):
            super().__init__(); self.transformer = transformer; self.shared = torch.nn.Linear(c, 8); self.state = torch.nn.Embedding(max(nstates, 1), 8)
            if transformer:
                layer = torch.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
                self.encoder = torch.nn.TransformerEncoder(layer, 1); self.position = torch.nn.Parameter(torch.zeros(2, 8))
            if not transformer: self.hidden = torch.nn.Linear(16, 8)
            self.head = torch.nn.Linear(16, 1)
        def forward(self, v, state):
            projected = self.shared(v.reshape(-1, 2, v.shape[-1]))
            tok = projected if self.transformer else projected.relu()
            if self.transformer: tok = self.encoder(tok + self.position)
            h = tok.mean(dim=1) if self.transformer else self.hidden(tok.reshape(len(v), -1)).relu()
            h = torch.cat([h, self.state(state.clamp_min(0))], dim=1)
            return self.head(h)
    requested_device = str(spec.get("device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production neural models")
    device = torch.device(requested_device)
    model = Net(tx.shape[2], len(levels), spec["family"] == "temporal_transformer").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(spec.get("learning_rate", .001)), weight_decay=float(spec.get("weight_decay", .001)))
    epochs = int(spec.get("epochs", 25)); batch = int(spec.get("batch_size", 256))
    xt = torch.tensor(flat.reshape(len(flat), 2, -1), dtype=torch.float32, device=device)
    yt, tc, qc = yt.to(device), tc.to(device), qc.to(device)
    for _ in range(epochs):
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), batch):
            idx = order[start:start + batch]
            opt.zero_grad(); loss = ((model(xt[idx], tc[idx]) - yt[idx]) ** 2).mean(); loss.backward(); opt.step()
    pred = np.full(len(qflat), np.nan)
    if eligible.any():
        if np.any(qc[eligible].detach().cpu().numpy() < 0):
            raise ValueError("score state is absent from training embedding")
        model.eval()
        with torch.no_grad(): pred[eligible] = (model(torch.tensor(qflat[eligible].reshape(eligible.sum(), 2, -1), dtype=torch.float32, device=device), qc[eligible]).cpu().numpy().ravel() * ys + ym)
    return pred, {"rows": len(y), "epochs": epochs, "batch_size": batch, "parameter_count": sum(p.numel() for p in model.parameters()), "backend": "torch", "device": str(device)}


def fit_predict(dataset: OriginDataset, specification: dict[str, Any], *, seed: int = 0):
    if not isinstance(dataset, OriginDataset):
        raise TypeError("fit_predict requires an OriginDataset")
    ds, spec, actual_seed = dataset, specification, int(seed)
    name = spec["name"]
    if name == "RW0":
        pred = _random_walk(ds); meta = {"model": name, "rows": int(ds.n_train), "features": []}
    elif spec["family"] in {"mlp", "temporal_transformer"}:
        pred, meta = _nn_predict(ds, spec, actual_seed)
    elif spec["family"] == "scalar_ar1_state_space":
        from .state_space import fit_state_space
        pred, meta = fit_state_space(ds, spec, actual_seed)
    elif name == "local_lag_ridge":
        tx, y, qx, eligible = _masked(ds)
        if "lag_delta" not in ds.channels:
            raise ValueError("local_lag_ridge requires lag_delta")
        lag = ds.channels.index("lag_delta")
        x = tx[:, 1, lag:lag + 1]; q = qx[:, 1, lag:lag + 1]
        x, q, _, _ = _standardize(x, q); x, q, _ = _state_aug(ds, x, q)
        pred = np.full(len(q), np.nan)
        if eligible.any(): pred[eligible] = _ridge_unpenalized(x, y, q[eligible], float(spec.get("alpha", 1.0)), 1)
        meta = {"model": name, "rows": len(y), "features": ["lag_delta", "state_effects"]}
    else:
        pred, meta = _tree_predict(ds, spec, actual_seed)
    if np.any(~np.isfinite(pred[np.asarray(ds.score["eligible"], bool)])):
        raise ValueError(f"nonfinite prediction on eligible score row for {name}")
    meta.update({"model": name, "seed": actual_seed, "finite_predictions": int(np.isfinite(pred).sum())})
    return pred, meta
