"""M3 (part 2) — LSTM trajectory-kinematics risk model.

Layer-2, model #2 of the agentic architecture. An **anticipatory** risk model: given the
past `KIN_HISTORY` steps of an agent's kinematics, predict the derived kinematic risk over
the next `KIN_HORIZON` steps. Framing it as forecasting (past window -> *future* risk)
makes the temporal model earn its keep — it learns to see a hard-braking / swerving event
coming rather than merely re-describing the current window.

Supervision is the `kinematics.kinematic_risk` heuristic (no crash labels exist on these
datasets — a documented limitation). Trained on Argoverse 2; the physical, rate-normalized
features let it transfer to nuScenes.

Pipeline:
    scenes -> build_windows -> (X:[N,H,4], y:[N]) -> standardize -> LSTM -> sigmoid risk
Artifacts (state_dict + feature mean/std + hyperparams) are saved to KIN_MODEL_PATH.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from sdiq import config
from sdiq.data_loader import DrivingScene
from sdiq.kinematics import N_FEATURES, extract_kinematics, kinematic_risk

log = logging.getLogger("sdiq.kinematic")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class KinematicRiskLSTM(nn.Module):
    """2-layer LSTM over (B, H, 4) kinematics -> scalar risk logit per sequence."""

    def __init__(self, n_features: int = N_FEATURES,
                 hidden: int = config.KIN_HIDDEN, layers: int = config.KIN_LAYERS) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)             # (B, H, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)   # logit per sequence


# ---------------------------------------------------------------------------
# Windowed, anticipatory dataset construction
# ---------------------------------------------------------------------------
def build_windows(scenes: Iterable[DrivingScene],
                  history: int = config.KIN_HISTORY,
                  horizon: int = config.KIN_HORIZON,
                  stride: int = 5,
                  neg_per_pos: float = 4.0,
                  pos_threshold: float = 0.25,
                  rng_seed: int = 0,
                  max_windows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Slide windows over every agent track: X = past `history` kinematic steps,
    y = max derived risk over the next `horizon` steps.

    To counter the heavy class imbalance (~3% risky), negatives (y < pos_threshold) are
    subsampled to ~`neg_per_pos` per positive. Returns (X[N,H,4], y[N]).
    """
    rng = np.random.default_rng(rng_seed)
    pos_X, pos_y, neg_X, neg_y = [], [], [], []
    for scene in scenes:
        for track in scene.agents:
            feats = extract_kinematics(track)
            T = feats.shape[0]
            if T < history + horizon:
                continue
            step_risk = kinematic_risk(feats)
            for i in range(0, T - history - horizon + 1, stride):
                window = feats[i:i + history]
                future = step_risk[i + history:i + history + horizon]
                label = float(future.max())
                if label >= pos_threshold:
                    pos_X.append(window); pos_y.append(label)
                else:
                    neg_X.append(window); neg_y.append(label)

    n_keep_neg = int(len(pos_X) * neg_per_pos) if pos_X else min(len(neg_X), 2000)
    if len(neg_X) > n_keep_neg:
        idx = rng.choice(len(neg_X), n_keep_neg, replace=False)
        neg_X = [neg_X[i] for i in idx]; neg_y = [neg_y[i] for i in idx]

    X = np.asarray(pos_X + neg_X, dtype=np.float32)
    y = np.asarray(pos_y + neg_y, dtype=np.float32)
    if len(X) == 0:
        return X.reshape(0, history, N_FEATURES), y
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    if max_windows and len(X) > max_windows:
        X, y = X[:max_windows], y[:max_windows]
    return X, y


def feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over all timesteps (for standardization)."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainReport:
    n_train: int
    n_val: int
    val_mse: float
    val_auc: float          # ranking: do high-risk windows score above low-risk?
    pos_fraction: float
    epochs: int


def _auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC without sklearn dependency (rank statistic)."""
    pos = y_score[y_true_bin == 1]
    neg = y_score[y_true_bin == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score)); ranks[order] = np.arange(1, len(y_score) + 1)
    r_pos = ranks[y_true_bin == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train(scenes: Iterable[DrivingScene] | None = None,
          scenario_limit: int = 2000,
          epochs: int = 8,
          batch_size: int = 256,
          lr: float = 1e-3,
          val_frac: float = 0.2,
          device: str | None = None,
          save_path: Path | None = None,
          seed: int = 0,
          **window_kwargs) -> tuple[KinematicRiskLSTM, TrainReport, dict]:
    """Build the dataset, train the LSTM, evaluate, and (optionally) save the artifact."""
    from sdiq.data_loader import iter_av2_scenarios

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if scenes is None:
        scenes = iter_av2_scenarios(limit=scenario_limit)

    log.info("building windows ...")
    X, y = build_windows(scenes, rng_seed=seed, **window_kwargs)
    if len(X) < 10:
        raise RuntimeError(f"too few windows ({len(X)}) — increase scenario_limit")
    mean, std = feature_stats(X)
    Xn = (X - mean) / std

    n_val = max(1, int(len(Xn) * val_frac))
    Xtr, ytr = Xn[n_val:], y[n_val:]
    Xva, yva = Xn[:n_val], y[:n_val]
    pos_frac = float((y >= 0.5).mean())

    model = KinematicRiskLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Soft-label BCE. build_windows already balances the set by subsampling negatives, so
    # only a mild residual up-weighting is applied (capped) to avoid over-predicting risk.
    pos_weight = torch.tensor([min(5.0, max(1.0, (1 - pos_frac) / max(pos_frac, 1e-3)))],
                              device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.from_numpy(Xtr).to(device); ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device); yva_t = torch.from_numpy(yva).to(device)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for b in range(0, len(Xtr_t), batch_size):
            bi = perm[b:b + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[bi]), ytr_t[bi])
            loss.backward()
            opt.step()
        if (ep + 1) % max(1, epochs // 4) == 0:
            log.info("epoch %d/%d  train_loss=%.4f", ep + 1, epochs, float(loss.detach()))

    model.eval()
    with torch.no_grad():
        val_score = torch.sigmoid(model(Xva_t)).cpu().numpy()
    val_mse = float(np.mean((val_score - yva) ** 2))
    val_auc = _auc((yva >= 0.5).astype(int), val_score)
    report = TrainReport(len(Xtr), len(Xva), val_mse, val_auc, pos_frac, epochs)

    meta = {"feat_mean": mean, "feat_std": std,
            "hidden": config.KIN_HIDDEN, "layers": config.KIN_LAYERS,
            "history": config.KIN_HISTORY, "horizon": config.KIN_HORIZON,
            "n_features": N_FEATURES}
    save_path = Path(save_path or config.KIN_MODEL_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, save_path)
    log.info("saved model -> %s", save_path)
    return model, report, meta


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class KinematicRiskModel:
    """Loads a trained LSTM and scores tracks / scenes."""

    def __init__(self, path: Path | None = None, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(Path(path or config.KIN_MODEL_PATH),
                          map_location=self.device, weights_only=False)
        self.mean = ckpt["feat_mean"]; self.std = ckpt["feat_std"]
        self.history = ckpt["history"]
        self.model = KinematicRiskLSTM(ckpt["n_features"], ckpt["hidden"], ckpt["layers"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def from_model(cls, model: KinematicRiskLSTM, meta: dict,
                   device: str | None = None) -> "KinematicRiskModel":
        obj = cls.__new__(cls)
        obj.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.mean = meta["feat_mean"]; obj.std = meta["feat_std"]
        obj.history = meta["history"]
        obj.model = model.to(obj.device).eval()
        return obj

    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        Xn = (windows - self.mean) / self.std
        with torch.no_grad():
            t = torch.from_numpy(Xn.astype(np.float32)).to(self.device)
            return torch.sigmoid(self.model(t)).cpu().numpy()

    def score_track(self, track, reducer: str = "max") -> float:
        """Anticipatory risk for one track: score each history window, reduce."""
        feats = extract_kinematics(track)
        T = feats.shape[0]
        if T < self.history:
            if T == 0:
                return 0.0
            pad = np.repeat(feats[:1], self.history - T, axis=0)
            windows = np.concatenate([pad, feats])[None, :, :]
        else:
            windows = np.stack([feats[i:i + self.history]
                                for i in range(0, T - self.history + 1, max(1, self.history // 2))])
        scores = self._score_windows(windows)
        return float(scores.max() if reducer == "max" else scores.mean())

    def score_scene(self, scene: DrivingScene) -> dict:
        """Per-agent risk + scene aggregates (max = worst agent, mean = overall)."""
        per_agent = {a.track_id: self.score_track(a) for a in scene.agents}
        vals = list(per_agent.values()) or [0.0]
        return {"per_agent": per_agent,
                "max_risk": float(max(vals)), "mean_risk": float(np.mean(vals))}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os
    limit = int(os.environ.get("SDIQ_TRAIN_SCENARIOS", "2000"))
    _, rep, _ = train(scenario_limit=limit)
    print(f"\nTrainReport: train={rep.n_train} val={rep.n_val} "
          f"val_mse={rep.val_mse:.4f} val_auc={rep.val_auc:.3f} "
          f"pos_frac={rep.pos_fraction:.3f} epochs={rep.epochs}")
