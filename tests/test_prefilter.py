import pandas as pd

from satmatch.geometry import GeometryOptions, passes_may_overlap


def track(lon: list[float], lat: list[float], start: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=len(lon), freq="60s", tz="UTC"),
            "lon": lon,
            "lat": lat,
        }
    )


def test_pass_prefilter_requires_space_and_time() -> None:
    options = GeometryOptions(max_dt_minutes=30.0)
    s3 = track([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], "2026-01-01T00:00:00")
    near = track([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], "2026-01-01T00:10:00")
    far_in_space = track([100.0, 101.0, 102.0], [0.0, 0.0, 0.0], "2026-01-01T00:10:00")
    far_in_time = track([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], "2026-01-01T02:00:00")

    assert passes_may_overlap(s3, near, options, sample_seconds=60.0)
    assert not passes_may_overlap(s3, far_in_space, options, sample_seconds=60.0)
    assert not passes_may_overlap(s3, far_in_time, options, sample_seconds=60.0)


def test_pass_prefilter_rejects_polar_only_crossing() -> None:
    s3 = track([0.0, 1.0, 2.0], [70.0, 70.0, 70.0], "2026-01-01T00:00:00")
    swot = track([0.0, 1.0, 2.0], [70.5, 70.5, 70.5], "2026-01-01T00:10:00")

    assert not passes_may_overlap(s3, swot, GeometryOptions(), sample_seconds=60.0)
    assert passes_may_overlap(
        s3,
        swot,
        GeometryOptions(min_latitude=-80.0, max_latitude=80.0),
        sample_seconds=60.0,
    )
