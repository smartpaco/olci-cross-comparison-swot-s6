import pandas as pd
import pytest

from satmatch.geometry import GeometryOptions, make_vignettes
from satmatch.orf import PassWindow


class AllOcean:
    def percentage(self, geometry, lon0, lat0, to_local) -> float:
        return 100.0


def orbit_track(start: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=21, freq="10s", tz="UTC"),
            "lon": [0.0] * 21,
            "lat": [value / 10.0 for value in range(-10, 11)],
        }
    )


def pass_window(uid: int, start: str) -> PassWindow:
    beginning = pd.Timestamp(start)
    return PassWindow(
        uid=uid,
        cycle=1,
        pass_number=2,
        revolution=3,
        start=beginning,
        equator=beginning + pd.Timedelta(seconds=100),
        end=beginning + pd.Timedelta(seconds=200),
        ascending=True,
    )


def test_make_vignettes_returns_one_scene_with_both_swaths() -> None:
    swot_start = "2026-01-01T00:00:00Z"
    s3_start = "2026-01-01T00:05:00Z"
    records = make_vignettes(
        orbit_track(s3_start),
        orbit_track(swot_start),
        pass_window(1, s3_start),
        pass_window(2, swot_start),
        GeometryOptions(
            max_along_track_km=50.0,
            max_dt_minutes=10.0,
            min_area_km2=0.0,
            min_ocean_percent=0.0,
        ),
        AllOcean(),
    )

    assert records
    for scene in records:
        assert scene["swot_side"] == "both"
        assert scene["left_geometry"].area > 0.0
        assert scene["right_geometry"].area > 0.0
        assert scene["geometry"].geom_type == "MultiPolygon"
        assert scene["area_km2"] == pytest.approx(
            scene["left_area_km2"] + scene["right_area_km2"],
            rel=1e-6,
        )
        assert scene["ocean_percent"] == 100.0
