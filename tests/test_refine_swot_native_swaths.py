import numpy as np
import pytest

from scripts.refine_swot_native_swaths import (
    native_footprint,
    representative_spacing_m,
)


def test_native_footprint_uses_full_selected_swath() -> None:
    longitude = np.tile(np.linspace(-1.0, 1.0, 5), (4, 1))
    latitude = np.tile(np.linspace(10.0, 10.3, 4)[:, None], (1, 5))
    selected = np.zeros((4, 5), dtype=bool)
    selected[:, 1:4] = True

    footprint = native_footprint(longitude, latitude, selected)

    assert footprint.bounds == pytest.approx((-0.5, 10.0, 0.5, 10.3))
    assert footprint.area == pytest.approx(0.3)


def test_representative_spacing_ignores_missing_edge_columns() -> None:
    longitude = np.tile(np.array([0.0, 0.1, 0.2]), (3, 1))
    latitude = np.tile(np.array([0.0, 0.01, 0.02])[:, None], (1, 3))
    cross_track = np.tile(np.array([-20_000.0, -10_000.0, np.nan]), (3, 1))
    longitude[:, 2] = np.nan
    latitude[:, 2] = np.nan

    spacing = representative_spacing_m(longitude, latitude, cross_track)

    assert spacing == pytest.approx(1_105.7, rel=0.01)
