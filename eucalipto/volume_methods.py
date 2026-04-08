"""Volume estimation methods extracted and simplified from new_volume.ipynb.

Métodos atualmente implementados:

- Cilindro simples (DAP + altura);
- Integração da curva de afilamento (taper) ajustada por polinômio;
- Frustum (cone truncado) com raios estimados na base e topo;
- Volume a partir de QSM ("qsm"), delegando o ajuste do modelo a
    bibliotecas externas como PyTLidar / TreeQSM.

O método de cilindro continua sendo o padrão mais estável.
"""

from typing import Optional, Tuple, List

import numpy as np
from scipy.integrate import quad
import pyransac3d as pyrsc


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


def _slice_points(points: np.ndarray,
                  z_center: float,
                  thickness: float) -> np.ndarray:
    z = points[:, 2]
    half = thickness / 2.0
    mask = (z >= z_center - half) & (z <= z_center + half)
    return points[mask]


def _radius_ransac_on_slice(points_slice: np.ndarray,
                            thresh: float = 0.03,
                            radius_min: float = 0.01,
                            radius_max: float = 1.0) -> Optional[Tuple[float, int]]:
    if points_slice.shape[0] < 3:
        return None

    circle = pyrsc.Circle()
    center, normal, radius, inliers = circle.fit(points_slice, thresh=thresh)

    if radius <= 0 or not (radius_min <= radius <= radius_max):
        return None

    return float(radius), int(len(inliers))


def estimate_taper_radii(points: np.ndarray,
                         n_height_samples: int = 15,
                         slice_thickness: float = 0.2,
                         ransac_thresh: float = 0.03,
                         radius_min: float = 0.01,
                         radius_max: float = 1.0,
                         min_inliers: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate radius(h) profile along the trunk height using RANSAC slices.

    Returns
    -------
    heights_rel : np.ndarray
        Alturas relativas (m) em relação à base (z_min).
    radii : np.ndarray
        Raios estimados (m) em cada altura.
    trunk_height : float
        Altura total aproximada do tronco (m).
    """
    if points.shape[0] < 3:
        return np.array([]), np.array([]), 0.0

    z = points[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    trunk_height = z_max - z_min

    if trunk_height <= 0:
        return np.array([]), np.array([]), 0.0

    # Amostras ao longo da altura, evitando extremidades para estabilidade
    z_samples = np.linspace(z_min + 0.1, z_max - 0.1, n_height_samples)

    heights: List[float] = []
    radii: List[float] = []

    for z_target in z_samples:
        slice_pts = _slice_points(points, z_center=z_target, thickness=slice_thickness)
        if slice_pts.shape[0] < 3:
            continue

        res = _radius_ransac_on_slice(slice_pts,
                                      thresh=ransac_thresh,
                                      radius_min=radius_min,
                                      radius_max=radius_max)
        if res is None:
            continue
        radius, n_inliers = res
        if n_inliers < min_inliers:
            continue

        heights.append(z_target - z_min)
        radii.append(radius)

    if len(radii) < 2:
        return np.array([]), np.array([]), trunk_height

    return np.asarray(heights), np.asarray(radii), trunk_height


def estimate_volume_taper(points: np.ndarray,
                          wood_density_kg_m3: Optional[float] = None) -> dict:
    """Estimate volume by integrating a taper curve r(h).

    A curva r(h) é ajustada com um polinômio (grau até 3) sobre os
    raios estimados em diferentes alturas usando RANSAC em fatias.
    """
    heights_rel, radii, trunk_height = estimate_taper_radii(points)
    if heights_rel.size < 3:
        raise ValueError("Insuficientes amostras de raio para o método taper.")

    # Ajuste polinomial r(h)
    degree = int(min(3, len(radii) - 1))
    coeffs = np.polyfit(heights_rel, radii, degree)
    poly = np.poly1d(coeffs)

    def integrand(h: float) -> float:
        r = float(poly(h))
        if r <= 0:
            return 0.0
        return float(np.pi * r * r)

    volume_m3, _ = quad(integrand, 0.0, trunk_height)
    volume_liters = volume_m3 * 1000.0

    info = {
        "dbh_cm": None,
        "height_m": float(trunk_height),
        "volume_m3": float(volume_m3),
        "volume_liters": float(volume_liters),
        "taper_sample_count": int(len(radii)),
        "taper_poly_coeffs": coeffs.tolist(),
    }

    if wood_density_kg_m3 is not None:
        info["mass_kg"] = float(volume_m3 * wood_density_kg_m3)

    return info


def estimate_volume_frustum(points: np.ndarray,
                            wood_density_kg_m3: Optional[float] = None) -> dict:
    """Estimate volume assuming a frustum (cone truncado).

    Os raios na base e no topo são obtidos a partir da primeira e da
    última amostra da curva de taper.
    """
    heights_rel, radii, trunk_height = estimate_taper_radii(points)
    if radii.size < 2:
        raise ValueError("Insuficientes amostras de raio para o método frustum.")

    radius_base = float(radii[0])
    radius_top = float(radii[-1])

    volume_m3 = (1.0 / 3.0) * np.pi * trunk_height * (
        radius_base ** 2 + radius_base * radius_top + radius_top ** 2
    )
    volume_liters = float(volume_m3 * 1000.0)

    info = {
        "dbh_cm": None,
        "height_m": float(trunk_height),
        "volume_m3": float(volume_m3),
        "volume_liters": volume_liters,
        "radius_base_m": radius_base,
        "radius_top_m": radius_top,
    }

    if wood_density_kg_m3 is not None:
        info["mass_kg"] = float(volume_m3 * wood_density_kg_m3)

    return info


def estimate_volume_qsm(points: np.ndarray,
                        wood_density_kg_m3: Optional[float] = None,
                        qsm_volume_func=None,
                        **kwargs) -> dict:
    """Estimate volume using a QSM-based approach (e.g. via PyTLidar).

    Parameters
    ----------
    points : np.ndarray
        Nuvem de pontos (idealmente tronco + galhos principais) em metros.
    wood_density_kg_m3 : float, optional
        Densidade da madeira para estimar massa (se desejado).
    qsm_volume_func : callable
        Função externa responsável por ajustar o QSM e retornar o volume.
        A assinatura esperada é::

            def qsm_volume_func(points: np.ndarray, **kwargs) -> float | tuple:
                ""Retorna volume_m3 ou (volume_m3, info_dict).""

        Isso permite integrar PyTLidar (TreeQSM) ou outras bibliotecas sem
        acoplar diretamente o pacote aqui.

    Returns
    -------
    info : dict
        Dicionário com volume, massa (opcional) e metadados do QSM.
    """
    if qsm_volume_func is None:
        raise ValueError(
            "Para usar o método de volume 'qsm' é necessário fornecer "
            "um 'qsm_volume_func' que ajuste o modelo (por exemplo, "
            "via PyTLidar/TreeQSM) e retorne o volume em m³."
        )

    res = qsm_volume_func(points, **kwargs)

    qsm_info = None
    if isinstance(res, tuple) and len(res) == 2:
        volume_m3, qsm_info = res
    else:
        volume_m3 = res

    volume_m3 = float(volume_m3)
    volume_liters = float(volume_m3 * 1000.0)

    info = {
        "dbh_cm": None,
        "height_m": None,
        "volume_m3": volume_m3,
        "volume_liters": volume_liters,
        "qsm_info": qsm_info,
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

        Actualmente suporta quatro métodos principais:

        - "cylinder": modelo de cilindro usando DAP + altura;
        - "taper": integração da curva de afilamento r(h) estimada por fatias;
        - "frustum": modelo de tronco como frustum (cone truncado) com raios
            estimados na base e topo;
        - "qsm": volume a partir de um modelo QSM ajustado externamente
            (ex.: PyTLidar / TreeQSM) via ``qsm_volume_func``.

    Parameters
    ----------
    points : np.ndarray or None
        Trunk points; currently unused for cylinder but kept
        for future methods.
    dbh_cm : float, optional
    height_m : float, optional
    method : str
        Um de {"cylinder", "taper", "frustum", "qsm"}.

    Returns
    -------
    dict with volume information.
    """
    method = method.lower()

    if method == "cylinder":
        if dbh_cm is None or height_m is None:
            raise ValueError("dbh_cm e height_m são necessários para volume 'cylinder'.")
        return estimate_volume_cylinder(dbh_cm, height_m, **kwargs)

    if points is None:
        raise ValueError("O método de volume selecionado requer os pontos do tronco.")

    if method == "taper":
        return estimate_volume_taper(points, **kwargs)

    if method == "frustum":
        return estimate_volume_frustum(points, **kwargs)

    if method == "qsm":
        return estimate_volume_qsm(points, **kwargs)

    raise ValueError(f"Método de volume não suportado: {method}")
