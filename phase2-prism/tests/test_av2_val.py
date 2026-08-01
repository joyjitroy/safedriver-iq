"""M0 sanity test — Argoverse 2 motion-forecasting val split loads via the av2 devkit.

Config-driven path (no hard-coded ~/av-safety-poc). If the av2 devkit is unavailable,
falls back to reading the scenario parquet directly with pyarrow so the data itself is
still validated (the devkit is a convenience, not a hard requirement for the POC).
"""
from __future__ import annotations

from collections import Counter

from sdiq import config


def _first_parquet():
    p = next(config.AV2_VAL_ROOT.rglob("*.parquet"), None)
    assert p is not None, f"no scenario parquet found under {config.AV2_VAL_ROOT}"
    return p


def test_av2_loads_via_devkit():
    try:
        from av2.datasets.motion_forecasting.scenario_serialization import (
            load_argoverse_scenario_parquet,
        )
    except Exception as exc:  # devkit missing/broken -> use the fallback test
        import pytest
        pytest.skip(f"av2 devkit unavailable ({exc}); see test_av2_loads_via_pyarrow")
        return

    path = _first_parquet()
    scenario = load_argoverse_scenario_parquet(path)
    assert scenario.scenario_id
    assert scenario.city_name
    assert len(scenario.timestamps_ns) > 0
    assert len(scenario.tracks) > 0

    counts = Counter(t.object_type.value for t in scenario.tracks)
    vru = [t for t in scenario.tracks
           if t.object_type.value in ("pedestrian", "cyclist")]
    print("Scenario:", scenario.scenario_id, "| city:", scenario.city_name)
    print("tracks:", len(scenario.tracks), "| object types:", dict(counts))
    print("VRU tracks:", len(vru))


def test_av2_loads_via_pyarrow():
    """Devkit-free validation: the parquet is readable and has the expected columns."""
    import pyarrow.parquet as pq

    path = _first_parquet()
    table = pq.read_table(path)
    cols = set(table.column_names)
    # AV2 motion-forecasting schema staples.
    for required in ("track_id", "object_type", "timestep", "position_x", "position_y"):
        assert required in cols, f"missing column {required!r} in {path.name} ({cols})"
    assert table.num_rows > 0
    print(f"pyarrow OK: {path.name} | {table.num_rows} rows | cols={sorted(cols)}")


def test_av2_val_count():
    n = sum(1 for d in config.AV2_VAL_ROOT.iterdir() if d.is_dir())
    assert n == 24988, f"expected 24988 AV2 scenarios, got {n}"
    print(f"AV2 scenario folders: {n}")


if __name__ == "__main__":
    test_av2_val_count()
    test_av2_loads_via_pyarrow()
    try:
        test_av2_loads_via_devkit()
        print("\nPASS: av2 sanity test (devkit)")
    except SystemExit:
        print("\nPASS: av2 sanity test (pyarrow fallback; devkit skipped)")
