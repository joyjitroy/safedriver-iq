"""M4 (part 2) — LSTM correction for VRU trajectory forecasting + combined risk model.

Layer-2, model #3 of the agentic architecture: a HYBRID of physics (Social Force Model,
`social_force.py`) and learning. The SFM gives a reaction-aware VRU forecast; a small LSTM
learns the *residual* between that forecast and the observed motion, sharpening short-horizon
predictions. Forecasting happens in a VRU-centric canonical frame (translate to the VRU,
rotate to its heading) so the model is invariant to map orientation and transfers across
cities / datasets.

Two outputs:
  * trajectory forecast (SFM + LSTM correction) — evaluated by ADE/FDE vs SFM-only.
  * ego-VRU interaction risk — the conservative constant-velocity collision-course metric
    from `social_force.assess_scene_vru` (the safety signal), with the SFM+LSTM realistic
    risk reported alongside.

Trained on Argoverse 2 @10 Hz; nuScenes (2 Hz) uses the SFM physics directly.
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
from sdiq.social_force import (
    _resample, _scene_grid, assess_scene_vru, constant_velocity_rollout, sfm_rollout,
)

log = logging.getLogger("sdiq.vru")

# history feature layout (canonical frame): vru rel pos, vru vel, ego rel pos, ego vel
N_VRU_FEATURES = 8


# ---------------------------------------------------------------------------
# canonical frame helpers
# ---------------------------------------------------------------------------
def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _canon(theta: float):
    """Return (R, Rinv) for world<->canonical (heading-aligned) transforms."""
    R = _rotation(theta)
    return R, R.T


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class VRUForecastLSTM(nn.Module):
    """Encode VRU history -> predict residual offsets (horizon x 2) over the SFM forecast."""

    def __init__(self, n_features: int = N_VRU_FEATURES, hidden: int = config.VRU_HIDDEN,
                 layers: int = config.VRU_LAYERS, horizon: int = config.VRU_HORIZON) -> None:
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, horizon * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).view(-1, self.horizon, 2)   # (B, H, 2) residual


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------
@dataclass
class VRUSamples:
    X: np.ndarray         # (N, Hh, 8) canonical history features
    sfm: np.ndarray       # (N, Hf, 2) SFM forecast offsets (canonical)
    actual: np.ndarray    # (N, Hf, 2) observed future offsets (canonical)


def build_vru_dataset(scenes: Iterable[DrivingScene],
                      history: int = config.VRU_HISTORY,
                      horizon: int = config.VRU_HORIZON,
                      stride: int = 5,
                      max_neighbors: int = 12,
                      max_samples: int | None = None,
                      seed: int = 0) -> VRUSamples:
    """Slide windows over every VRU track; for each, build canonical-frame history
    features, the SFM forecast, and the observed future (all as offsets from the VRU at
    the reference step)."""
    rng = np.random.default_rng(seed)
    Xs, sfms, acts = [], [], []
    for scene in scenes:
        g = _scene_grid(scene)
        if g is None:
            continue
        grid, epos, evel = g
        G = len(grid)
        if G < history + horizon + 1:
            continue
        dt = float(np.median(np.diff(grid)))
        others = scene.agents
        resampled = {a.track_id: _resample(a, grid) for a in others}

        for vru in scene.vru_agents:
            vpos, vvel, vvalid = resampled[vru.track_id]
            for k in range(0, G - history - horizon, stride):
                r = k + history - 1                       # reference (end of history)
                fut = slice(r + 1, r + 1 + horizon)
                if not (vvalid[k] and vvalid[r] and vvalid[r + horizon]):
                    continue
                speed = np.linalg.norm(vvel[r])
                if speed < 0.3:                            # skip near-stationary VRUs
                    continue
                theta = float(np.arctan2(vvel[r][1], vvel[r][0]))
                R, Rinv = _canon(theta)
                origin = vpos[r]

                # neighbours at reference (other agents), nearest few
                npos, nvel = [], []
                for a in others:
                    if a.track_id == vru.track_id:
                        continue
                    p, v, val = resampled[a.track_id]
                    if val[r]:
                        npos.append(p[r]); nvel.append(v[r])
                npos = np.array(npos) if npos else np.zeros((0, 2))
                nvel = np.array(nvel) if nvel else np.zeros((0, 2))
                if len(npos) > max_neighbors:
                    d = np.linalg.norm(npos - origin, axis=1)
                    keep = np.argsort(d)[:max_neighbors]
                    npos, nvel = npos[keep], nvel[keep]

                ego_future = epos[r + 1:r + 1 + horizon]
                sfm_world = sfm_rollout(vpos[r], vvel[r], ego_future, npos, nvel, horizon, dt)
                sfm_off = (sfm_world - origin) @ Rinv.T              # canonical offsets
                actual_off = (vpos[fut] - origin) @ Rinv.T

                # history features in canonical frame
                hist = slice(k, k + history)
                vru_rel = (vpos[hist] - origin) @ Rinv.T
                vru_v = vvel[hist] @ Rinv.T
                ego_rel = (epos[hist] - origin) @ Rinv.T
                ego_v = evel[hist] @ Rinv.T
                feat = np.concatenate([vru_rel, vru_v, ego_rel, ego_v], axis=1)  # (Hh, 8)

                Xs.append(feat.astype(np.float32))
                sfms.append(sfm_off.astype(np.float32))
                acts.append(actual_off.astype(np.float32))

    if not Xs:
        return VRUSamples(np.zeros((0, history, N_VRU_FEATURES), np.float32),
                          np.zeros((0, horizon, 2), np.float32),
                          np.zeros((0, horizon, 2), np.float32))
    X = np.stack(Xs); sfm = np.stack(sfms); act = np.stack(acts)
    perm = rng.permutation(len(X))
    X, sfm, act = X[perm], sfm[perm], act[perm]
    if max_samples and len(X) > max_samples:
        X, sfm, act = X[:max_samples], sfm[:max_samples], act[:max_samples]
    return VRUSamples(X, sfm, act)


def _ade_fde(pred: np.ndarray, actual: np.ndarray):
    """Average & final displacement error between (N,H,2) trajectories."""
    d = np.linalg.norm(pred - actual, axis=2)        # (N, H)
    return float(d.mean()), float(d[:, -1].mean())


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@dataclass
class VRUTrainReport:
    n_train: int
    n_val: int
    sfm_ade: float
    sfm_fde: float
    corrected_ade: float
    corrected_fde: float
    epochs: int

    @property
    def ade_improvement_pct(self) -> float:
        return 100.0 * (self.sfm_ade - self.corrected_ade) / max(self.sfm_ade, 1e-9)


def train(scenes: Iterable[DrivingScene] | None = None,
          scenario_limit: int = 1200,
          epochs: int = 12,
          batch_size: int = 256,
          lr: float = 1e-3,
          val_frac: float = 0.2,
          device: str | None = None,
          save_path: Path | None = None,
          seed: int = 0,
          **ds_kwargs) -> tuple[VRUForecastLSTM, VRUTrainReport, dict]:
    """Train the residual LSTM and report ADE/FDE for SFM vs SFM+LSTM."""
    from sdiq.data_loader import iter_av2_scenarios

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if scenes is None:
        scenes = iter_av2_scenarios(limit=scenario_limit)

    log.info("building VRU windows ...")
    ds = build_vru_dataset(scenes, seed=seed, **ds_kwargs)
    if len(ds.X) < 10:
        raise RuntimeError(f"too few VRU windows ({len(ds.X)}) — increase scenario_limit")

    mean = ds.X.reshape(-1, N_VRU_FEATURES).mean(0)
    std = ds.X.reshape(-1, N_VRU_FEATURES).std(0); std[std < 1e-6] = 1.0
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    Xn = (ds.X - mean) / std
    residual = ds.actual - ds.sfm                    # what the LSTM must predict

    n_val = max(1, int(len(Xn) * val_frac))
    sl = slice(n_val, None); vl = slice(0, n_val)
    horizon = ds.actual.shape[1]
    model = VRUForecastLSTM(horizon=horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(Xn[sl]).to(device); rtr = torch.from_numpy(residual[sl]).to(device)
    Xva = torch.from_numpy(Xn[vl]).to(device)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for b in range(0, len(Xtr), batch_size):
            bi = perm[b:b + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[bi]), rtr[bi])
            loss.backward(); opt.step()
        if (ep + 1) % max(1, epochs // 4) == 0:
            log.info("epoch %d/%d  mse=%.4f", ep + 1, epochs, float(loss.detach()))

    model.eval()
    with torch.no_grad():
        pred_resid = model(Xva).cpu().numpy()
    sfm_pred = ds.sfm[vl]
    corrected = sfm_pred + pred_resid
    actual = ds.actual[vl]
    sfm_ade, sfm_fde = _ade_fde(sfm_pred, actual)
    cor_ade, cor_fde = _ade_fde(corrected, actual)
    report = VRUTrainReport(len(Xtr), n_val, sfm_ade, sfm_fde, cor_ade, cor_fde, epochs)

    meta = {"feat_mean": mean, "feat_std": std, "horizon": horizon,
            "history": config.VRU_HISTORY, "hidden": config.VRU_HIDDEN,
            "layers": config.VRU_LAYERS, "n_features": N_VRU_FEATURES}
    save_path = Path(save_path or config.VRU_MODEL_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, save_path)
    log.info("saved VRU model -> %s", save_path)
    return model, report, meta


# ---------------------------------------------------------------------------
# inference: combined SFM + LSTM risk model
# ---------------------------------------------------------------------------
class VRURiskModel:
    """SFM + LSTM-corrected VRU forecasting and ego-VRU interaction risk."""

    def __init__(self, path: Path | None = None, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(Path(path or config.VRU_MODEL_PATH),
                          map_location=self.device, weights_only=False)
        self.mean = ckpt["feat_mean"]; self.std = ckpt["feat_std"]
        self.history = ckpt["history"]; self.horizon = ckpt["horizon"]
        self.model = VRUForecastLSTM(ckpt["n_features"], ckpt["hidden"], ckpt["layers"],
                                     ckpt["horizon"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def from_model(cls, model, meta, device=None):
        obj = cls.__new__(cls)
        obj.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.mean = meta["feat_mean"]; obj.std = meta["feat_std"]
        obj.history = meta["history"]; obj.horizon = meta["horizon"]
        obj.model = model.to(obj.device).eval()
        return obj

    def assess_scene(self, scene: DrivingScene) -> dict:
        """Conservative (const-velocity) interaction risk — the safety signal — plus the
        SFM realistic risk for comparison. Returns aggregates, near-miss count, and the
        closest-approach distance/TTC of the worst (highest-risk) VRU."""
        conservative = assess_scene_vru(scene, forecaster=constant_velocity_rollout)
        realistic = assess_scene_vru(scene, forecaster=sfm_rollout)
        worst = max(conservative.interactions, key=lambda i: i.risk, default=None)
        return {
            "max_risk": conservative.max_risk,
            "mean_risk": conservative.mean_risk,
            "near_misses": conservative.near_misses,
            "realistic_max_risk": realistic.max_risk,
            "min_distance": worst.min_distance if worst else float("inf"),
            "ttc": worst.ttc if worst else float("inf"),
            "per_agent": {i.track_id: i.risk for i in conservative.interactions},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os
    limit = int(os.environ.get("SDIQ_TRAIN_SCENARIOS", "1200"))
    _, rep, _ = train(scenario_limit=limit)
    print(f"\nVRUTrainReport: train={rep.n_train} val={rep.n_val} epochs={rep.epochs}")
    print(f"  SFM        ADE={rep.sfm_ade:.3f} FDE={rep.sfm_fde:.3f}")
    print(f"  SFM+LSTM   ADE={rep.corrected_ade:.3f} FDE={rep.corrected_fde:.3f}")
    print(f"  ADE improvement: {rep.ade_improvement_pct:.1f}%")
