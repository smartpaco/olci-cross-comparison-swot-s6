from pathlib import Path

import pandas as pd
import pytest
from shapely import LineString

from satmatch.cli import resolve_s3_platform
from satmatch.geometry import _polygon_parts
from satmatch.orf import OrbitInterpolator, pass_windows, read_orf


def test_read_and_interpolate_small_orf(tmp_path: Path):
    path = tmp_path / "mini.orf"
    path.write_text(
        "entete\n"
        "2023/07/20 20:07:49.838 000 00574 02630 239.05  77.68\n"
        "2023/07/20 20:33:31.216 000 00574 02630 322.60   0.00\n"
        "2023/07/20 20:59:16.622 000 00575 02630  46.14 -77.67\n"
        "2023/07/20 21:25:02.163 000 00575 02631 129.64   0.00\n"
        "2023/07/20 21:50:43.500 000 00576 02631 213.18  77.68\n",
        encoding="ascii",
    )
    events = read_orf(path)
    windows = pass_windows(events)
    assert len(events) == 5
    assert len(windows) == 2
    assert windows[0].ascending is False
    sampled = OrbitInterpolator(events).sample(windows[0], 60.0)
    assert len(sampled) > 50
    assert sampled["lat"].between(-90.0, 90.0).all()

    selected = pass_windows(
        events,
        start=pd.Timestamp("2023-07-20T21:00:00Z"),
        end=pd.Timestamp("2023-07-20T22:00:00Z"),
    )
    assert len(selected) == 1
    assert selected[0].pass_number == 575


def test_tangent_intersection_has_no_surface():
    assert _polygon_parts(LineString([(0.0, 0.0), (1.0, 1.0)])) == []


def test_resolve_sentinel3_platform_from_orf_name() -> None:
    assert resolve_s3_platform("S3A_ORF_example") == "S3A"
    assert resolve_s3_platform("path/to/S3B_ORF_example") == "S3B"


def test_explicit_sentinel3_platform_for_generic_orf_name() -> None:
    assert resolve_s3_platform("orbit_table.txt", "S3B") == "S3B"


def test_reject_platform_orf_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        resolve_s3_platform("S3A_ORF_example", "S3B")
