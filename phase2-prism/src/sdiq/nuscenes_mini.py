"""Lightweight, dependency-free reader for the nuScenes *mini* JSON tables.

Why this exists instead of `nuscenes-devkit`: the devkit hard-pins ``numpy<2.0``,
which is irreconcilable with the modern ``torch>=2`` + ``shap>=0.50`` stack the rest
of the POC needs (both require ``numpy>=2``). The mini split is just a set of JSON
tables joined by token, so we read them directly and expose the small slice of the
devkit API the POC actually uses:

    nusc = NuScenesMini(dataroot, version="v1.0-mini")
    nusc.scene            # list[dict]
    nusc.sample           # list[dict]
    nusc.sample_annotation
    nusc.get("sample", token)               # token lookup, O(1)
    sample["data"]["CAM_FRONT"]             # synthesized channel -> sample_data token
    nusc.get("sample_data", token)["filename"]

It is intentionally a *reader*, not a full SDK (no geometry/box/render helpers).
M1's data_loader builds the richer ``AgentTrack``/``EgoState`` objects on top of this.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# The mini tables we load. (Others exist but the POC does not need them yet.)
_TABLES = (
    "category",
    "attribute",
    "visibility",
    "instance",
    "sensor",
    "calibrated_sensor",
    "ego_pose",
    "log",
    "scene",
    "sample",
    "sample_data",
    "sample_annotation",
    "map",
)


class NuScenesMini:
    def __init__(self, dataroot: str | Path, version: str = "v1.0-mini",
                 verbose: bool = False) -> None:
        self.dataroot = Path(dataroot)
        self.version = version
        self.table_root = self.dataroot / version
        if not self.table_root.is_dir():
            raise FileNotFoundError(
                f"nuScenes tables not found at {self.table_root}. "
                f"dataroot must contain the '{version}/' folder alongside "
                f"samples/ sweeps/ maps/."
            )

        # Load every table as a list[dict].
        self._tables: dict[str, list[dict[str, Any]]] = {}
        for name in _TABLES:
            path = self.table_root / f"{name}.json"
            self._tables[name] = json.loads(path.read_text()) if path.exists() else []

        # token -> record index, per table, for O(1) get().
        self._index: dict[str, dict[str, dict]] = {}
        for name, rows in self._tables.items():
            self._index[name] = {r["token"]: r for r in rows}

        self._build_sample_data_shortcut()

        if verbose:
            print(f"Loaded nuScenes {version} from {self.table_root}: "
                  f"{len(self.scene)} scenes, {len(self.sample)} samples, "
                  f"{len(self.sample_annotation)} annotations.")

    # -- devkit-style table attributes -------------------------------------
    def __getattr__(self, name: str) -> list[dict]:
        # Exposes self.scene, self.sample, self.sample_annotation, etc.
        tables = self.__dict__.get("_tables", {})
        if name in tables:
            return tables[name]
        raise AttributeError(name)

    def get(self, table: str, token: str) -> dict:
        """O(1) token lookup, mirroring nuscenes-devkit's NuScenes.get()."""
        try:
            return self._index[table][token]
        except KeyError as exc:
            raise KeyError(f"token {token!r} not found in table {table!r}") from exc

    # -- synthesize sample['data'] and sample['anns'] ----------------------
    def _build_sample_data_shortcut(self) -> None:
        """Reproduce the devkit's reverse index.

        For every *key-frame* sample_data record, resolve its sensor channel via
        calibrated_sensor -> sensor, then attach ``sample['data'][channel] = sd_token``.
        Also attach ``sample['anns']`` = list of annotation tokens for the sample.
        """
        chan_of_calib = {}
        for cs in self.calibrated_sensor:
            sensor = self._index["sensor"].get(cs["sensor_token"], {})
            chan_of_calib[cs["token"]] = sensor.get("channel")

        for sample in self.sample:
            sample.setdefault("data", {})
            sample.setdefault("anns", [])

        for sd in self.sample_data:
            if not sd.get("is_key_frame"):
                continue
            channel = chan_of_calib.get(sd["calibrated_sensor_token"])
            if channel:
                self._index["sample"][sd["sample_token"]]["data"][channel] = sd["token"]

        for ann in self.sample_annotation:
            samp = self._index["sample"].get(ann["sample_token"])
            if samp is not None:
                samp["anns"].append(ann["token"])

    # -- small conveniences the POC will reuse -----------------------------
    def field2token(self, table: str, field: str, value: Any) -> list[str]:
        return [r["token"] for r in self._tables[table] if r.get(field) == value]

    def get_sample_data_path(self, sd_token: str) -> Path:
        return self.dataroot / self.get("sample_data", sd_token)["filename"]


if __name__ == "__main__":
    from sdiq import config

    nusc = NuScenesMini(config.NUSCENES_DATAROOT, config.NUSCENES_VERSION, verbose=True)
    s = nusc.scene[0]
    print("first scene:", s["name"], "—", s["description"])
    samp = nusc.get("sample", s["first_sample_token"])
    print("channels:", list(samp["data"].keys()))
    print("CAM_FRONT:", nusc.get("sample_data", samp["data"]["CAM_FRONT"])["filename"])
