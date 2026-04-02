"""Volume estimation methods extracted and simplified from new_volume.ipynb.

For now, the primary production method is a simple cylinder model
using DBH and tree height.
"""

from typing import Optional

import numpy as np


def estimate_volume_cylinder(dbh_cm: float,
                             height_m: float,
                             wood_density_kg_m3: Optional[float] = None) -> dict:
    """Estimate trunk volume assuming a simple cylinder.

    Parameters
    ----------
    dbh_cm : float
        Diameter at breast height in centimeters.
    height_m : float
        Trunk height in meters.
    wood_density_kg_m3 : float, optional
        If given, an approximate dry mass is computed.

    Returns
    -------
    info : dict with keys:
        volume_m3, volume_liters, dbh_cm, height_m, mass_kg (optional)
    """
    dbh_m = dbh_cm / 100.0
    radius_m = dbh_m / 2.0

    volume_m3 = float(np.pi * radius_m ** 2 * height_m)
    volume_liters = volume_m3 * 1000.0

    info = {
        "dbh_cm": float(dbh_cm),
        "height_m": float(height_m),
        "volume_m3": volume_m3,
        "volume_liters": volume_liters,
    }

    if wood_density_kg_m3 is not None:
        info["mass_kg"] = float(volume_m3 * wood_density_kg_m3)

    return info


def estimate_volume(points,
                    dbh_cm: Optional[float] = None,
                    height_m: Optional[float] = None,
                    method: str = "cylinder",
                    **kwargs) -> dict:
    """High-level volume estimator.

    Currently supports only the cylinder method, which is also
    the recommended default based on prior experiments.

    Parameters
    ----------
    points : np.ndarray or None
        Trunk points; currently unused for cylinder but kept
        for future methods.
    dbh_cm : float, optional
    height_m : float, optional
    method : str
        Only "cylinder" is implemented.

    Returns
    -------
    dict with volume information.
    """
    if method != "cylinder":
        raise ValueError("Only 'cylinder' volume method is implemented at the moment.")

    if dbh_cm is None or height_m is None:
        raise ValueError("dbh_cm and height_m are required for cylinder volume.")

    return estimate_volume_cylinder(dbh_cm, height_m, **kwargs)
