"""Volume estimation methods extracted and simplified from new_volume.ipynb.

Métodos atualmente implementados:

- Cilindro simples (DAP + altura);
- Voxelização do tronco por ocupação de voxels;
- Integração da curva de afilamento (taper) ajustada por polinômio;
- Frustum (cone truncado) com raios estimados na base e topo;
- Perfil radial robusto por fatias (reconstrução geométrica com filtro de distância);
- Volume a partir de QSM ("qsm"), delegando o ajuste do modelo a
    bibliotecas externas como PyTLidar / TreeQSM.

O método de cilindro continua sendo o padrão mais estável.
"""

from typing import Optional, Tuple, List

import numpy as np
from scipy.integrate import quad
import pyransac3d as pyrsc


def estimate_volume_axis_profile(points: np.ndarray,
                                 n_slices: int = 20,
                                 slice_thickness: float = 0.30,
                                 radius_percentile: float = 85.0,
                                 min_points_per_slice: int = 20,
                                 wood_density_kg_m3: Optional[float] = None) -> dict:
    """Estimate volume by reconstructing a robust radial profile along height.

    The profile is built from horizontal slices. For each slice, the center is
    estimated with median XY and the radius is taken from a robust percentile of
    radial distances. This approach tolerates sparse holes in trunk predictions.
    """
    if points is None or points.shape[0] < 10:
        raise ValueError("Insufficient points for 'axis_profile' volume estimation.")

    z = points[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    height_m = z_max - z_min
    if height_m <= 0:
        raise ValueError("Invalid trunk height for 'axis_profile' volume estimation.")

    centers = np.linspace(z_min, z_max, n_slices)
    hs: List[float] = []
    rs: List[float] = []

    for zc in centers:
        sl = _slice_points(points, z_center=float(zc), thickness=slice_thickness)
        if sl.shape[0] < min_points_per_slice:
            continue

        ctr = np.median(sl[:, :2], axis=0)
        d = np.linalg.norm(sl[:, :2] - ctr, axis=1)
        r = float(np.percentile(d, radius_percentile))
        if r <= 0:
            continue

        hs.append(float(zc - z_min))
        rs.append(r)

    if len(rs) < 2:
        raise ValueError("Insufficient valid slices for 'axis_profile' volume estimation.")

    h_arr = np.asarray(hs)
    r_arr = np.asarray(rs)

    order = np.argsort(h_arr)
    h_arr = h_arr[order]
    r_arr = r_arr[order]

    # Numerical integration of area profile A(h)=pi*r(h)^2 using trapezoids.
    area = np.pi * (r_arr ** 2)
    volume_m3 = float(np.trapz(area, h_arr))
    volume_liters = float(volume_m3 * 1000.0)

    info = {
        "dbh_cm": None,
        "height_m": float(height_m),
        "volume_m3": volume_m3,
        "volume_liters": volume_liters,
        "axis_profile_slices_used": int(len(r_arr)),
        "axis_profile_radius_percentile": float(radius_percentile),
    }

    if wood_density_kg_m3 is not None:
        info["mass_kg"] = float(volume_m3 * wood_density_kg_m3)

    return info


def estimate_volume_voxel(points: np.ndarray,
                          voxel_size: float = 0.05,
                          wood_density_kg_m3: Optional[float] = None) -> dict:
    """Estimate volume by counting occupied voxels in the trunk point cloud.

    The trunk points are quantized into a regular voxel grid anchored at the
    local minimum XYZ of the cloud. The estimated volume is the number of
    occupied voxels multiplied by the voxel volume.
    """
    if points is None or points.shape[0] < 3:
        raise ValueError("Insufficient points for 'voxel' volume estimation.")

    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0 for 'voxel' volume estimation.")

    pts = np.asarray(points, dtype=np.float64)
    origin = pts.min(axis=0)
    voxel_indices = np.floor((pts - origin) / float(voxel_size)).astype(np.int64)

    unique_voxels = np.unique(voxel_indices, axis=0)
    occupied_voxels = int(unique_voxels.shape[0])
    voxel_volume = float(voxel_size ** 3)
    volume_m3 = float(occupied_voxels * voxel_volume)
    volume_liters = float(volume_m3 * 1000.0)

    bbox_extent = pts.max(axis=0) - origin
    bbox_voxels = int(np.prod(np.maximum(np.ceil(bbox_extent / voxel_size).astype(np.int64), 1)))
    fill_ratio = float(occupied_voxels / bbox_voxels) if bbox_voxels > 0 else 0.0

    info = {
        "dbh_cm": None,
        "height_m": float(pts[:, 2].max() - pts[:, 2].min()),
        "volume_m3": volume_m3,
        "volume_liters": volume_liters,
        "voxel_size_m": float(voxel_size),
        "voxel_occupied_count": occupied_voxels,
        "voxel_bbox_count_estimate": bbox_voxels,
        "voxel_fill_ratio": fill_ratio,
        "voxel_origin_xyz": origin.tolist(),
    }

    if wood_density_kg_m3 is not None:
        info["mass_kg"] = float(volume_m3 * wood_density_kg_m3)

    return info


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
        - "voxel": ocupação de voxels no tronco;
        - "taper": integração da curva de afilamento r(h) estimada por fatias;
        - "frustum": modelo de tronco como frustum (cone truncado) com raios
            estimados na base e topo;
        - "axis_profile": reconstrução por perfil radial robusto em fatias;
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
        Um de {"cylinder", "voxel", "taper", "frustum", "axis_profile", "qsm"}.

    Returns
    -------
    dict with volume information.
    """
    method = method.lower()

    if method == "cylinder":
        if dbh_cm is None or height_m is None:
            raise ValueError("dbh_cm e height_m são necessários para volume 'cylinder'.")
        return estimate_volume_cylinder(dbh_cm, height_m, **kwargs)

    if method == "voxel":
        return estimate_volume_voxel(points, **kwargs)

    if points is None:
        raise ValueError("O método de volume selecionado requer os pontos do tronco.")

    if method == "taper":
        return estimate_volume_taper(points, **kwargs)

    if method == "frustum":
        return estimate_volume_frustum(points, **kwargs)

    if method == "axis_profile":
        return estimate_volume_axis_profile(points, **kwargs)

    if method == "qsm":
        return estimate_volume_qsm(points, **kwargs)

    raise ValueError(f"Método de volume não suportado: {method}")
