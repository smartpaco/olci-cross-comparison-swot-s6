"""Conversion between total-column water vapour and zenith wet delay."""

from __future__ import annotations

from typing import TypeVar

import numpy as np


# Bennartz et al. (2017), Appendix A, Eq. A15.
# delay [m] = (A + B / Tm) * TCWV [kg m-2]
WET_DELAY_A_M_PER_KG_M2 = -2.95077e-5
WET_DELAY_B_M_K_PER_KG_M2 = 1.73276
DEFAULT_MEAN_ATMOSPHERIC_TEMPERATURE_K = 270.0

ArrayLike = TypeVar("ArrayLike")


def wet_delay_factor(
    mean_atmospheric_temperature_k: ArrayLike = DEFAULT_MEAN_ATMOSPHERIC_TEMPERATURE_K,
) -> ArrayLike:
    """Return the TCWV-to-zenith-wet-delay factor in m / (kg m-2).

    ``mean_atmospheric_temperature_k`` is the water-vapour-weighted mean
    atmospheric temperature, usually called Tm. It may be a scalar, NumPy
    array, or xarray DataArray. Missing values are propagated.
    """
    temperature = np.asarray(mean_atmospheric_temperature_k, dtype=float)
    finite = np.isfinite(temperature)
    if np.any(temperature[finite] <= 0.0):
        raise ValueError("Mean atmospheric temperature must be positive in kelvin")
    return (
        WET_DELAY_A_M_PER_KG_M2
        + WET_DELAY_B_M_K_PER_KG_M2 / mean_atmospheric_temperature_k
    )


def tcwv_to_zenith_wet_delay(
    tcwv_kg_m2: ArrayLike,
    mean_atmospheric_temperature_k: ArrayLike = DEFAULT_MEAN_ATMOSPHERIC_TEMPERATURE_K,
) -> ArrayLike:
    """Convert TCWV [kg m-2] to positive one-way zenith wet delay [m]."""
    factor = wet_delay_factor(mean_atmospheric_temperature_k)
    return tcwv_kg_m2 * factor
