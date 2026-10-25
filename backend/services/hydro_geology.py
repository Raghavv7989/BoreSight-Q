"""
BoreSight-Q: Hydrogeological Analysis Engine.

Implements the complete parcel-scale hydro-geology pipeline:
  1. Hydrologically conditioned DEM (skimage.morphology.reconstruction pit filling),
     D8 flow direction, flow accumulation (As), Strahler stream ordering (1 to 4),
     smooth vector LineStrings (drainage_geojson in EPSG:4326), Slope (beta),
     Plan & Profile Curvature, and Topographic Wetness Index:
     TWI = ln((As + 10) / tan(max(beta, 0.001)))
  2. Multi-azimuth SAR + DEM structural lineament extraction (hillshades at
     azimuths 0°, 45°, 90°, 135° at 30° altitude fused with log10(VH/VV) gradient),
     Canny + Probabilistic Hough (cv2.HoughLinesP), collinear fracture merging (10°),
     NDVI agricultural field edge masking, lineaments_geojson (EPSG:4326),
     10m lineament_density raster (km/km^2 within 100m circular kernel),
     and Euclidean distance rasters dist_to_drainage and dist_to_intersection.
  3. Subsurface fracture-drainage intersection pinpointing (lineament x drainage
     and conjugate lineament-lineament fractures inside the parcel polygon) and
     assembly of the aligned 12-band float32 feature stack:
     [twi, dist_to_drainage, lineament_density, dist_to_intersection, s1_vv_db,
      s1_vh_vv_ratio, s2_ndmi, s2_ndwi, curvature, slope, k_sat, insar_velocity]
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Guard optional heavy dependencies
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import skimage
    import skimage.morphology as ski_morph
except ImportError:
    skimage = None  # type: ignore
    ski_morph = None  # type: ignore

try:
    from scipy.ndimage import distance_transform_edt, uniform_filter
except ImportError:
    distance_transform_edt = None  # type: ignore
    uniform_filter = None  # type: ignore

try:
    import pyproj
except ImportError:
    pyproj = None  # type: ignore

try:
    from shapely.geometry import (
        LineString as ShapelyLineString,
        Point as ShapelyPoint,
        Polygon as ShapelyPolygon,
        shape as shapely_shape,
        mapping as shapely_mapping,
    )
    from shapely.ops import linemerge
except ImportError:
    ShapelyLineString = None  # type: ignore
    ShapelyPoint = None  # type: ignore
    ShapelyPolygon = None  # type: ignore
    shapely_shape = None  # type: ignore

from backend.services.satellite_fetcher import SatelliteDataStack

logger = logging.getLogger("boresight.hydro_geology")
logger.setLevel(logging.INFO)

# D8 direction codes (power-of-2, ArcGIS standard)
#   64  128  1
#   32   X   2
#   16   8   4
_D8_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
_D8_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
_D8_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_D8_DIST_FACTOR = np.array(
    [1.0, math.sqrt(2), 1.0, math.sqrt(2),
     1.0, math.sqrt(2), 1.0, math.sqrt(2)],
    dtype=np.float64,
)
_D8_REVERSE = np.array([16, 32, 64, 128, 1, 2, 4, 8], dtype=np.uint8)


# ======================================================================
# 1. Hydrological Conditioning & Pit Filling (skimage.morphology)
# ======================================================================

def fill_depressions(
    dem: np.ndarray,
    epsilon: float = 0.001,
    max_iterations: int = 500,
) -> np.ndarray:
    """
    Hydrologically condition DEM by filling sinks/depressions.
    Uses skimage.morphology.reconstruction (dual reconstruction by erosion)
    with a fast vectorized Planchon-Darboux fallback.
    """
    # 1. Preferred method: skimage morphological reconstruction
    if ski_morph is not None:
        try:
            mask = dem.astype(np.float64)
            seed = mask.copy()
            # Boundary cells retain their elevation and act as drainage outlets;
            # interior cells are set to a high ceiling.
            seed[1:-1, 1:-1] = mask.max() + 10.0
            filled = ski_morph.reconstruction(seed, mask, method="erosion")
            logger.info("Depression filling completed with skimage.morphology.reconstruction")
            return filled.astype(np.float32)
        except Exception as err:
            logger.warning("skimage reconstruction fallback to vectorized Planchon-Darboux: %s", err)

    # 2. Vectorized Planchon & Darboux (2001) iterative fallback
    H, W = dem.shape
    INF = 1e30
    filled = np.full_like(dem, INF, dtype=np.float64)
    dem64 = dem.astype(np.float64)

    # Boundary cells preserve original elevation
    filled[0, :] = dem64[0, :]
    filled[-1, :] = dem64[-1, :]
    filled[:, 0] = dem64[:, 0]
    filled[:, -1] = dem64[:, -1]

    # Pre-pad buffers for fast 8-neighbor shifting
    iteration = 0
    changed = True
    while changed and iteration < max_iterations:
        iteration += 1
        old_filled = filled.copy()

        # Compute minimum over 8 neighbors
        n_min = np.full((H, W), INF, dtype=np.float64)
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]:
            shifted = np.roll(np.roll(filled, -dr, axis=0), -dc, axis=1)
            # Clip boundary roll artifacts
            if dr == -1:
                shifted[-1, :] = INF
            elif dr == 1:
                shifted[0, :] = INF
            if dc == -1:
                shifted[:, -1] = INF
            elif dc == 1:
                shifted[:, 0] = INF
            n_min = np.minimum(n_min, shifted)

        ws = n_min + epsilon
        target = np.maximum(dem64, ws)
        # Only interior cells with filled > dem can update
        mask_update = (filled[1:-1, 1:-1] > dem64[1:-1, 1:-1]) & (target[1:-1, 1:-1] < filled[1:-1, 1:-1])
        filled[1:-1, 1:-1][mask_update] = target[1:-1, 1:-1][mask_update]

        max_diff = np.max(np.abs(old_filled - filled))
        if max_diff < 1e-4:
            changed = False

    logger.info("Vectorized depression filling converged in %d iterations", iteration)
    return filled.astype(np.float32)


# ======================================================================
# 2. D8 Flow Direction & Flow Accumulation
# ======================================================================

def compute_d8_flow_direction(
    filled_dem: np.ndarray,
    cell_size: float = 10.0,
) -> np.ndarray:
    """
    D8 flow direction: each cell drains to its steepest downslope neighbour.
    Codes: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE. Flat/pit cells -> 0.
    """
    H, W = filled_dem.shape
    flow_dir = np.zeros((H, W), dtype=np.uint8)
    dem64 = filled_dem.astype(np.float64)

    for r in range(H):
        for c in range(W):
            max_slope = 0.0
            best_code = 0
            for k in range(8):
                nr = r + _D8_DR[k]
                nc = c + _D8_DC[k]
                if 0 <= nr < H and 0 <= nc < W:
                    drop = dem64[r, c] - dem64[nr, nc]
                    dist = cell_size * _D8_DIST_FACTOR[k]
                    s = drop / dist
                    if s > max_slope:
                        max_slope = s
                        best_code = int(_D8_CODES[k])
            flow_dir[r, c] = best_code
    return flow_dir


def compute_flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """
    Compute contributing-area count (As) via topological sort.
    Each cell's value = number of upstream cells that drain through it.
    """
    H, W = flow_dir.shape
    acc = np.ones((H, W), dtype=np.float64)
    in_degree = np.zeros((H, W), dtype=np.int32)

    downstream_r = np.full((H, W), -1, dtype=np.int32)
    downstream_c = np.full((H, W), -1, dtype=np.int32)

    for r in range(H):
        for c in range(W):
            code = flow_dir[r, c]
            if code == 0:
                continue
            for k in range(8):
                if _D8_CODES[k] == code:
                    nr = r + _D8_DR[k]
                    nc = c + _D8_DC[k]
                    if 0 <= nr < H and 0 <= nc < W:
                        downstream_r[r, c] = nr
                        downstream_c[r, c] = nc
                        in_degree[nr, nc] += 1
                    break

    queue: deque = deque()
    for r in range(H):
        for c in range(W):
            if in_degree[r, c] == 0:
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        dr = downstream_r[r, c]
        dc = downstream_c[r, c]
        if dr >= 0:
            acc[dr, dc] += acc[r, c]
            in_degree[dr, dc] -= 1
            if in_degree[dr, dc] == 0:
                queue.append((dr, dc))

    return acc


# ======================================================================
# 3. Strahler Stream Extraction & LineString Smoothing
# ======================================================================

def _downstream_cell(r: int, c: int, flow_dir: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return downstream (row, col) or None."""
    H, W = flow_dir.shape
    code = flow_dir[r, c]
    for k in range(8):
        if _D8_CODES[k] == code:
            nr = r + _D8_DR[k]
            nc = c + _D8_DC[k]
            if 0 <= nr < H and 0 <= nc < W:
                return nr, nc
    return None


def _upstream_stream_cells(r: int, c: int, flow_dir: np.ndarray, is_stream: np.ndarray) -> List[Tuple[int, int]]:
    """Return upstream cells that drain into (r, c)."""
    H, W = flow_dir.shape
    ups = []
    for k in range(8):
        nr = r + _D8_DR[k]
        nc = c + _D8_DC[k]
        if 0 <= nr < H and 0 <= nc < W and is_stream[nr, nc]:
            rev_code = _D8_REVERSE[k]
            if flow_dir[nr, nc] == rev_code:
                ups.append((nr, nc))
    return ups


def _smooth_line_coordinates(coords: List[List[float]]) -> List[List[float]]:
    """
    Applies Gaussian/Chaikin-style 3-point weighted moving average smoothing
    to eliminate raster 90-degree stair-stepping on stream LineStrings.
    Preserves first and last endpoints.
    """
    if len(coords) < 3:
        return coords

    smoothed = [coords[0]]
    for i in range(1, len(coords) - 1):
        prev_p = coords[i - 1]
        curr_p = coords[i]
        next_p = coords[i + 1]
        sm_lon = 0.25 * prev_p[0] + 0.50 * curr_p[0] + 0.25 * next_p[0]
        sm_lat = 0.25 * prev_p[1] + 0.50 * curr_p[1] + 0.25 * next_p[1]
        smoothed.append([round(sm_lon, 7), round(sm_lat, 7)])
    smoothed.append(coords[-1])
    return smoothed


def extract_strahler_streams(
    flow_dir: np.ndarray,
    flow_acc: np.ndarray,
    cell_size: float = 10.0,
    threshold: int = 100,
    utm_transform: Any = None,
    utm_epsg: int = 32643,
) -> Dict[str, Any]:
    """
    Extract stream network with Strahler stream orders (1 to 4).
    Vectorizes stream pixels into smooth LineString features in EPSG:4326.
    """
    H, W = flow_dir.shape
    is_stream = flow_acc >= threshold

    order = np.zeros((H, W), dtype=np.int32)
    in_deg_s = np.zeros((H, W), dtype=np.int32)

    for r in range(H):
        for c in range(W):
            if not is_stream[r, c]:
                continue
            ups = _upstream_stream_cells(r, c, flow_dir, is_stream)
            in_deg_s[r, c] = len(ups)

    queue: deque = deque()
    for r in range(H):
        for c in range(W):
            if is_stream[r, c] and in_deg_s[r, c] == 0:
                order[r, c] = 1  # headwater stream segment
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        ds = _downstream_cell(r, c, flow_dir)
        if ds is None or not is_stream[ds[0], ds[1]]:
            continue
        dr, dc = ds
        in_deg_s[dr, dc] -= 1

        if in_deg_s[dr, dc] == 0:
            ups = _upstream_stream_cells(dr, dc, flow_dir, is_stream)
            resolved = [order[u[0], u[1]] for u in ups if order[u[0], u[1]] > 0]
            if len(resolved) == 0:
                order[dr, dc] = 1
            else:
                max_ord = max(resolved)
                count_max = resolved.count(max_ord)
                order[dr, dc] = max_ord + 1 if count_max >= 2 else max_ord
            queue.append((dr, dc))

    visited = np.zeros((H, W), dtype=bool)
    features: List[Dict[str, Any]] = []
    seg_id = 0

    def pixel_to_utm(r_px: int, c_px: int) -> Tuple[float, float]:
        if utm_transform is not None:
            try:
                x = utm_transform.c + c_px * utm_transform.a + 0.5 * utm_transform.a
                y = utm_transform.f + r_px * utm_transform.e + 0.5 * utm_transform.e
                return x, y
            except AttributeError:
                pass
            if isinstance(utm_transform, (list, tuple)) and len(utm_transform) >= 6:
                a, b, c_val, d, e, f = utm_transform[:6]
                return c_val + c_px * a + 0.5 * a, f + r_px * e + 0.5 * e
        return float(c_px * cell_size), float((H - r_px) * cell_size)

    def utm_to_wgs84(x: float, y: float) -> Tuple[float, float]:
        if pyproj is not None:
            tr = pyproj.Transformer.from_crs(utm_epsg, 4326, always_xy=True)
            return tr.transform(x, y)
        from backend.services.satellite_fetcher import _utm_to_wgs84
        return _utm_to_wgs84(x, y, utm_epsg)

    for r in range(H):
        for c in range(W):
            if not is_stream[r, c] or visited[r, c] or order[r, c] == 0:
                continue
            seg_order = int(order[r, c])
            if seg_order < 1 or seg_order > 4:
                continue

            path_px = [(r, c)]
            visited[r, c] = True
            cr, cc = r, c
            while True:
                ds = _downstream_cell(cr, cc, flow_dir)
                if ds is None or not is_stream[ds[0], ds[1]]:
                    break
                dr, dc = ds
                if visited[dr, dc] and order[dr, dc] == seg_order:
                    break
                if order[dr, dc] != seg_order:
                    path_px.append((dr, dc))
                    break
                visited[dr, dc] = True
                path_px.append((dr, dc))
                cr, cc = dr, dc

            if len(path_px) < 2:
                continue

            raw_coords_wgs = []
            for pr, pc_val in path_px:
                ux, uy = pixel_to_utm(pr, pc_val)
                lon, lat = utm_to_wgs84(ux, uy)
                raw_coords_wgs.append([round(lon, 7), round(lat, 7)])

            # Smooth LineString geometry
            coords_wgs = _smooth_line_coordinates(raw_coords_wgs)

            length_m = 0.0
            for i in range(1, len(path_px)):
                dr_px = path_px[i][0] - path_px[i - 1][0]
                dc_px = path_px[i][1] - path_px[i - 1][1]
                length_m += math.sqrt(dr_px ** 2 + dc_px ** 2) * cell_size

            seg_id += 1
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords_wgs},
                "properties": {
                    "segment_id": seg_id,
                    "stream_order": seg_order,
                    "strahler_order": seg_order,
                    "length_m": round(length_m, 1),
                },
            })

    logger.info("Extracted %d smooth stream lineaments (orders 1-4)", len(features))
    return {"type": "FeatureCollection", "features": features}


# ======================================================================
# 4. Terrain Derivatives & Topographic Wetness Index (TWI)
# ======================================================================

def compute_terrain_derivatives(
    dem: np.ndarray,
    flow_acc: np.ndarray,
    cell_size: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Slope (beta), Plan Curvature, Profile Curvature, and
    Topographic Wetness Index (TWI):
      TWI = ln((As + 10) / tan(max(beta, 0.001)))
    """
    dem64 = dem.astype(np.float64)
    H, W = dem64.shape
    cs = float(cell_size)
    cs2 = cs * cs

    dz_dy, dz_dx = np.gradient(dem64, cs)
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)).astype(np.float32)

    dem_pad = np.pad(dem64, 1, mode="edge")
    z1 = dem_pad[0:-2, 0:-2]
    z2 = dem_pad[0:-2, 1:-1]
    z3 = dem_pad[0:-2, 2:]
    z4 = dem_pad[1:-1, 0:-2]
    z5 = dem_pad[1:-1, 1:-1]
    z6 = dem_pad[1:-1, 2:]
    z7 = dem_pad[2:, 0:-2]
    z8 = dem_pad[2:, 1:-1]
    z9 = dem_pad[2:, 2:]

    d_coeff = ((z6 - z4) / (2.0 * cs))
    e_coeff = ((z2 - z8) / (2.0 * cs))
    a_coeff = ((z4 + z6 - 2.0 * z5) / (2.0 * cs2))
    b_coeff = ((z2 + z8 - 2.0 * z5) / (2.0 * cs2))
    c_coeff = ((z3 - z1 - z9 + z7) / (4.0 * cs2))

    d2_e2 = d_coeff ** 2 + e_coeff ** 2
    d2_e2_safe = np.where(d2_e2 > 1e-12, d2_e2, 1e-12)

    plan_num = -(a_coeff * e_coeff ** 2 - c_coeff * d_coeff * e_coeff + b_coeff * d_coeff ** 2)
    plan_curvature = (plan_num / (d2_e2_safe ** 1.5)).astype(np.float32)

    prof_num = -(a_coeff * d_coeff ** 2 + c_coeff * d_coeff * e_coeff + b_coeff * e_coeff ** 2)
    prof_denom = d2_e2_safe * (1.0 + d2_e2) ** 1.5
    profile_curvature = (prof_num / prof_denom).astype(np.float32)

    plan_curvature = np.clip(plan_curvature, -5.0, 5.0)
    profile_curvature = np.clip(profile_curvature, -5.0, 5.0)

    # Topographic Wetness Index (TWI) with the exact formula:
    # TWI = ln((As + 10) / tan(max(beta, 0.001)))
    tan_beta = np.tan(np.maximum(slope.astype(np.float64), 0.001))
    twi = np.log((flow_acc.astype(np.float64) + 10.0) / tan_beta).astype(np.float32)
    twi = np.clip(twi, 0.0, 30.0)

    return slope, plan_curvature, profile_curvature, twi


# ======================================================================
# 5. Multi-Azimuth SAR + DEM Structural Lineament Extractor
# ======================================================================

def _hillshade(
    dem: np.ndarray,
    azimuth_deg: float,
    altitude_deg: float = 30.0,
    cell_size: float = 10.0,
) -> np.ndarray:
    """Analytical hillshade at specified azimuth and altitude (30 degrees)."""
    dem64 = dem.astype(np.float64)
    dz_dy, dz_dx = np.gradient(dem64, cell_size)
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect = np.arctan2(-dz_dy, dz_dx)

    az_rad = math.radians(azimuth_deg)
    alt_rad = math.radians(altitude_deg)
    zen = math.pi / 2.0 - alt_rad

    shade = (
        math.cos(zen) * np.cos(slope)
        + math.sin(zen) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    shade = np.clip(shade, 0.0, 1.0)
    return (shade * 255.0).astype(np.uint8)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mx - mn < 1e-9:
        return np.full(arr.shape, 128, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)


def _merge_collinear_segments(
    lines: List[Tuple[int, int, int, int]],
    angle_tol_deg: float = 10.0,
    perp_dist_m: float = 30.0,
    cell_size: float = 10.0,
) -> List[Tuple[int, int, int, int]]:
    """Merge segments that are nearly collinear (within 10 degrees) and adjacent."""
    if not lines:
        return lines

    def _seg_angle(x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0

    def _point_line_dist(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1) * cell_size
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y) * cell_size

    merged = list(lines)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            best_j = -1
            best_len = 0.0
            x1, y1, x2, y2 = merged[i]
            a_i = _seg_angle(x1, y1, x2, y2)
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                x3, y3, x4, y4 = merged[j]
                a_j = _seg_angle(x3, y3, x4, y4)
                angle_diff = abs(a_i - a_j)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                if angle_diff > angle_tol_deg:
                    continue
                d1 = _point_line_dist(x3, y3, x1, y1, x2, y2)
                d2 = _point_line_dist(x4, y4, x1, y1, x2, y2)
                if min(d1, d2) > perp_dist_m:
                    continue
                comb_len = math.hypot(x2 - x1, y2 - y1) + math.hypot(x4 - x3, y4 - y3)
                if comb_len > best_len:
                    best_len = comb_len
                    best_j = j

            if best_j >= 0:
                x3, y3, x4, y4 = merged[best_j]
                pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                a_rad = math.radians(a_i)
                projs = [(px * math.cos(a_rad) + py * math.sin(a_rad), px, py) for px, py in pts]
                projs.sort()
                new_seg = (projs[0][1], projs[0][2], projs[-1][1], projs[-1][2])
                new_merged.append(new_seg)
                used[i] = True
                used[best_j] = True
                changed = True
            else:
                new_merged.append(merged[i])
                used[i] = True

        for j in range(len(merged)):
            if not used[j]:
                new_merged.append(merged[j])
        merged = new_merged
    return merged


def extract_lineaments(
    dem: np.ndarray,
    s1_vv_db: np.ndarray,
    s1_vh_db: np.ndarray,
    s2_nir: np.ndarray,
    s2_red: np.ndarray,
    utm_transform: Any,
    utm_epsg: int,
    parcel_coords_wgs84: List[List[List[float]]],
    cell_size: float = 10.0,
) -> Dict[str, Any]:
    """
    Multi-azimuth SAR+DEM structural lineament extraction:
      1. 4 directional DEM hillshades at azimuths 0°, 45°, 90°, 135° (altitude 30°)
      2. Sentinel-1 log10(VH/VV) gradient magnitude
      3. Fused composite -> Canny edge detection + cv2.HoughLinesP
      4. Collinear segment merging (10° angular tolerance)
      5. High-NDVI agricultural boundary filtering
      6. Conversion to lineaments_geojson in EPSG:4326 with length_m and azimuth_deg
    """
    H, W = dem.shape

    # 1. 4-directional DEM hillshades at 30° altitude
    hs_layers = []
    for az in (0.0, 45.0, 90.0, 135.0):
        hs_layers.append(_hillshade(dem, az, altitude_deg=30.0, cell_size=cell_size))

    # 2. Sentinel-1 log10(VH/VV) gradient magnitude
    vv_lin = np.power(10.0, s1_vv_db / 10.0)
    vh_lin = np.power(10.0, s1_vh_db / 10.0)
    ratio = np.where(vv_lin > 1e-10, vh_lin / vv_lin, 0.01)
    log_ratio = np.log10(np.maximum(ratio, 1e-10)).astype(np.float32)

    if cv2 is not None:
        sx = cv2.Sobel(log_ratio, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(log_ratio, cv2.CV_64F, 0, 1, ksize=3)
    else:
        sy_pad, sx_pad = np.gradient(log_ratio.astype(np.float64))
        sx, sy = sx_pad, sy_pad
    sar_grad = np.sqrt(sx ** 2 + sy ** 2).astype(np.float32)

    # 3. Fused composite & Canny edge detection
    layers_u8 = [_normalize_to_uint8(l) for l in hs_layers]
    layers_u8.append(_normalize_to_uint8(sar_grad))
    composite = np.maximum.reduce(layers_u8)

    raw_lines: List[Tuple[int, int, int, int]] = []
    if cv2 is not None:
        blurred = cv2.GaussianBlur(composite, (0, 0), sigmaX=1.5)
        edges = cv2.Canny(blurred, 50, 150)
        hough = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=50,
            maxLineGap=15,
        )
        if hough is not None:
            for seg in hough:
                x1, y1, x2, y2 = seg[0]
                raw_lines.append((int(x1), int(y1), int(x2), int(y2)))
    else:
        # Fallback linear ridge scanning across multiple directions
        raw_lines = _directional_line_detector_fallback(composite, min_length=15)

    logger.info("Raw extracted line segments: %d", len(raw_lines))

    # 4. Collinear segment merging
    merged_lines = _merge_collinear_segments(
        raw_lines, angle_tol_deg=10.0, perp_dist_m=30.0, cell_size=cell_size
    )
    logger.info("Merged collinear lineaments: %d", len(merged_lines))

    # 5. Mask out high-NDVI agricultural field edges
    nir = s2_nir.astype(np.float64)
    red = s2_red.astype(np.float64)
    ndvi = (nir - red) / (nir + red + 1e-10)

    ndvi_p85 = float(np.percentile(ndvi, 85))
    filtered_lines = []
    for x1, y1, x2, y2 in merged_lines:
        ndvi_vals = []
        for t in np.linspace(0, 1, 10):
            sx_pt = max(0, min(int(round(x1 + t * (x2 - x1))), W - 1))
            sy_pt = max(0, min(int(round(y1 + t * (y2 - y1))), H - 1))
            ndvi_vals.append(ndvi[sy_pt, sx_pt])
        mean_ndvi = float(np.mean(ndvi_vals))
        std_ndvi = float(np.std(ndvi_vals))
        # Agricultural crop rows / hedge boundaries have high persistent vegetation and low along-track variance
        if mean_ndvi > 0.55 and mean_ndvi > ndvi_p85 and std_ndvi < 0.05:
            continue
        filtered_lines.append((x1, y1, x2, y2))

    logger.info("Lineaments after agricultural mask: %d", len(filtered_lines))

    # 6. Export lineaments_geojson in EPSG:4326 with length_m and azimuth_deg
    def _pixel_to_utm(col: int, row: int) -> Tuple[float, float]:
        if utm_transform is not None:
            try:
                x = utm_transform.c + col * utm_transform.a
                y = utm_transform.f + row * utm_transform.e
                return x, y
            except AttributeError:
                pass
            if isinstance(utm_transform, (list, tuple)) and len(utm_transform) >= 6:
                a, b, c_val, d, e, f = utm_transform[:6]
                return c_val + col * a, f + row * e
        return float(col * cell_size), float((H - row) * cell_size)

    def _to_wgs84(x: float, y: float) -> Tuple[float, float]:
        if pyproj is not None:
            tr = pyproj.Transformer.from_crs(utm_epsg, 4326, always_xy=True)
            return tr.transform(x, y)
        from backend.services.satellite_fetcher import _utm_to_wgs84
        return _utm_to_wgs84(x, y, utm_epsg)

    features: List[Dict[str, Any]] = []
    for idx, (x1, y1, x2, y2) in enumerate(filtered_lines):
        ux1, uy1 = _pixel_to_utm(x1, y1)
        ux2, uy2 = _pixel_to_utm(x2, y2)
        length_m = math.hypot(ux2 - ux1, uy2 - uy1)
        if length_m < 20.0:
            continue

        dx_utm = ux2 - ux1
        dy_utm = uy2 - uy1
        azimuth = math.degrees(math.atan2(dx_utm, dy_utm)) % 180.0

        lon1, lat1 = _to_wgs84(ux1, uy1)
        lon2, lat2 = _to_wgs84(ux2, uy2)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [round(lon1, 7), round(lat1, 7)],
                    [round(lon2, 7), round(lat2, 7)],
                ],
            },
            "properties": {
                "lineament_id": idx + 1,
                "azimuth_deg": round(azimuth, 1),
                "length_m": round(length_m, 1),
            },
        })

    return {"type": "FeatureCollection", "features": features}


def _directional_line_detector_fallback(
    composite_u8: np.ndarray,
    min_length: int = 15,
) -> List[Tuple[int, int, int, int]]:
    """
    Directional linear ridge scanner fallback when cv2.HoughLinesP is unavailable.
    Identifies dominant linear fractures across azimuths 0°, 45°, 90°, 135°.
    """
    H, W = composite_u8.shape
    threshold = float(np.percentile(composite_u8, 86))
    edges = composite_u8 >= threshold
    lines: List[Tuple[int, int, int, int]] = []

    # Horizontal runs
    step_r = max(4, H // 50)
    for r in range(step_r, H - step_r, step_r):
        row = edges[r, :]
        in_run = False
        start_c = 0
        for c in range(W):
            if row[c] and not in_run:
                in_run = True
                start_c = c
            elif not row[c] and in_run:
                in_run = False
                if (c - start_c) >= min_length:
                    lines.append((start_c, r, c, r))
        if in_run and (W - start_c) >= min_length:
            lines.append((start_c, r, W - 1, r))

    # Vertical runs
    step_c = max(4, W // 50)
    for c in range(step_c, W - step_c, step_c):
        col = edges[:, c]
        in_run = False
        start_r = 0
        for r in range(H):
            if col[r] and not in_run:
                in_run = True
                start_r = r
            elif not col[r] and in_run:
                in_run = False
                if (r - start_r) >= min_length:
                    lines.append((c, start_r, c, r))
        if in_run and (H - start_r) >= min_length:
            lines.append((c, start_r, c, H - 1))

    # Diagonal runs
    step_d = max(6, min(H, W) // 35)
    for offset in range(-H + 30, W - 30, step_d):
        diag = np.diagonal(edges, offset=offset)
        if len(diag) < min_length:
            continue
        in_run = False
        start_i = 0
        for i in range(len(diag)):
            if diag[i] and not in_run:
                in_run = True
                start_i = i
            elif not diag[i] and in_run:
                in_run = False
                if (i - start_i) >= min_length:
                    if offset >= 0:
                        lines.append((start_i + offset, start_i, i + offset, i))
                    else:
                        lines.append((start_i, start_i - offset, i, i - offset))
        if in_run and (len(diag) - start_i) >= min_length:
            end_i = len(diag) - 1
            if offset >= 0:
                lines.append((start_i + offset, start_i, end_i + offset, end_i))
            else:
                lines.append((start_i, start_i - offset, end_i, end_i - offset))

    return lines


# ======================================================================
# 6. Spatial Distance & Density Rasters
# ======================================================================

def _edt_distance(mask: np.ndarray, cell_size: float = 10.0) -> np.ndarray:
    """Euclidean distance transform (m) from True cells in mask."""
    if distance_transform_edt is not None:
        return (distance_transform_edt(~mask) * cell_size).astype(np.float32)

    H, W = mask.shape
    dist = np.full((H, W), 1e9, dtype=np.float64)
    dist[mask] = 0.0
    changed = True
    while changed:
        changed = False
        for dr, dc, d in [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]:
            shifted = np.roll(np.roll(dist, -dr, axis=0), -dc, axis=1) + d * cell_size
            update = shifted < dist
            if np.any(update):
                dist[update] = shifted[update]
                changed = True
    return dist.astype(np.float32)


def compute_lineament_density_km_km2(
    lineaments_geojson: Dict[str, Any],
    grid_shape: Tuple[int, int],
    utm_transform: Any,
    cell_size: float = 10.0,
    radius_m: float = 100.0,
) -> np.ndarray:
    """
    Computes a 10m lineament_density raster in km/km^2 within a 100m circular kernel.
    Area of 100m circular kernel = pi * (0.1 km)^2 = 0.0314159 km^2.
    """
    H, W = grid_shape
    # Mask of lineament segment length in km per pixel
    length_km_grid = np.zeros((H, W), dtype=np.float64)
    cell_km = (cell_size / 1000.0)

    def _wgs84_to_pixel(lon: float, lat: float, epsg_utm: int) -> Tuple[int, int]:
        if pyproj is not None:
            tr = pyproj.Transformer.from_crs(4326, epsg_utm, always_xy=True)
            ux, uy = tr.transform(lon, lat)
        else:
            from backend.services.satellite_fetcher import _wgs84_to_utm
            ux, uy = _wgs84_to_utm(lon, lat, epsg_utm)
        try:
            col = int(round((ux - utm_transform.c) / utm_transform.a))
            row = int(round((uy - utm_transform.f) / utm_transform.e))
        except AttributeError:
            if isinstance(utm_transform, (list, tuple)) and len(utm_transform) >= 6:
                a, b, c_val, d, e, f = utm_transform[:6]
                col = int(round((ux - c_val) / a))
                row = int(round((uy - f) / e))
            else:
                col, row = 0, 0
        return row, col

    epsg_utm = 32643
    for feat in lineaments_geojson.get("features", []):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if len(coords) >= 2:
            lon0, lat0 = coords[0]
            zone = int((lon0 + 180.0) / 6.0) + 1
            epsg_utm = (32600 + zone) if lat0 >= 0 else (32700 + zone)
            break

    # Rasterize lineaments into length_km_grid
    for feat in lineaments_geojson.get("features", []):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        for k in range(len(coords) - 1):
            r0, c0 = _wgs84_to_pixel(coords[k][0], coords[k][1], epsg_utm)
            r1, c1 = _wgs84_to_pixel(coords[k + 1][0], coords[k + 1][1], epsg_utm)
            steps = max(abs(r1 - r0), abs(c1 - c0), 1)
            # Step length in km
            step_len_km = math.hypot((r1 - r0) * cell_km, (c1 - c0) * cell_km) / steps
            for s in range(steps + 1):
                t = s / steps
                rr = int(round(r0 + t * (r1 - r0)))
                cc = int(round(c0 + t * (c1 - c0)))
                if 0 <= rr < H and 0 <= cc < W:
                    length_km_grid[rr, cc] += step_len_km

    # 100m circular kernel mask
    rad_px = int(round(radius_m / cell_size))  # 10 pixels for 100m at 10m cell
    y, x = np.ogrid[-rad_px:rad_px + 1, -rad_px:rad_px + 1]
    circular_kernel = ((x * x + y * y) <= (rad_px * rad_px)).astype(np.float64)
    kernel_area_km2 = math.pi * ((radius_m / 1000.0) ** 2)  # ~0.0314159 km^2

    # Convolve length_km_grid with circular_kernel
    if cv2 is not None:
        convolved_km = cv2.filter2D(length_km_grid, -1, circular_kernel, borderType=cv2.BORDER_REFLECT)
    elif uniform_filter is not None:
        # Uniform filter sum scaled to circular area
        k_size = rad_px * 2 + 1
        box_sum = uniform_filter(length_km_grid, size=k_size) * (k_size * k_size)
        convolved_km = box_sum * (np.sum(circular_kernel) / (k_size * k_size))
    else:
        # 2D cumulative sum box approximation scaled to circle
        k_size = rad_px * 2 + 1
        p = rad_px
        pad_arr = np.pad(length_km_grid, ((p + 1, p), (p + 1, p)), mode="reflect")
        cum = np.cumsum(np.cumsum(pad_arr, axis=0), axis=1)
        box_sum = (
            cum[k_size:, k_size:]
            - cum[:-k_size, k_size:]
            - cum[k_size:, :-k_size]
            + cum[:-k_size, :-k_size]
        )
        convolved_km = box_sum * (np.sum(circular_kernel) / (k_size * k_size))

    density_km_km2 = (convolved_km / kernel_area_km2).astype(np.float32)
    return density_km_km2


# ======================================================================
# 7. Subsurface Fracture-Drainage Intersection Finder
# ======================================================================

def find_intersections(
    lineaments_geojson: Dict[str, Any],
    drainage_geojson: Dict[str, Any],
    parcel_geometry_wgs84: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pinpoint geometric intersections within the parcel polygon:
      (a) lineament LineStrings x micro-drainage LineStrings
      (b) conjugate lineament LineStrings x lineament LineStrings
    """
    features: List[Dict[str, Any]] = []
    node_id = 0

    if ShapelyLineString is None or shapely_shape is None:
        poly_coords = parcel_geometry_wgs84.get("coordinates", [[]])[0]
        if not poly_coords:
            return {"type": "FeatureCollection", "features": []}

        def _pt_in_poly(px: float, py: float) -> bool:
            inside = False
            n = len(poly_coords)
            p1x, p1y = poly_coords[0][0], poly_coords[0][1]
            for idx in range(1, n + 1):
                p2x, p2y = poly_coords[idx % n][0], poly_coords[idx % n][1]
                if min(p1y, p2y) < py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x if p1y != p2y else p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        def _seg_ix(p1: List[float], p2: List[float], p3: List[float], p4: List[float]) -> Optional[Tuple[float, float]]:
            det = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
            if abs(det) < 1e-12:
                return None
            t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / det
            u = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0])) / det
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
            return None

        for l_feat in lineaments_geojson.get("features", []):
            l_coords = l_feat.get("geometry", {}).get("coordinates", [])
            l_props = l_feat.get("properties", {})
            for d_feat in drainage_geojson.get("features", []):
                d_coords = d_feat.get("geometry", {}).get("coordinates", [])
                d_props = d_feat.get("properties", {})
                for li in range(len(l_coords) - 1):
                    for di in range(len(d_coords) - 1):
                        ix_pt = _seg_ix(l_coords[li], l_coords[li + 1], d_coords[di], d_coords[di + 1])
                        if ix_pt and _pt_in_poly(ix_pt[0], ix_pt[1]):
                            node_id += 1
                            features.append({
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [round(ix_pt[0], 7), round(ix_pt[1], 7)]},
                                "properties": {
                                    "node_id": node_id,
                                    "intersection_type": "lineament-drainage",
                                    "lineament_azimuth_deg": l_props.get("azimuth_deg", 0),
                                    "lineament_length_m": l_props.get("length_m", 0),
                                    "stream_order": d_props.get("stream_order", d_props.get("strahler_order", 0)),
                                    "strahler_order": d_props.get("strahler_order", d_props.get("stream_order", 0)),
                                },
                            })

        lin_feats = lineaments_geojson.get("features", [])
        for i in range(len(lin_feats)):
            c1 = lin_feats[i].get("geometry", {}).get("coordinates", [])
            p1 = lin_feats[i].get("properties", {})
            for j in range(i + 1, len(lin_feats)):
                c2 = lin_feats[j].get("geometry", {}).get("coordinates", [])
                p2 = lin_feats[j].get("properties", {})
                for k1 in range(len(c1) - 1):
                    for k2 in range(len(c2) - 1):
                        ix_pt = _seg_ix(c1[k1], c1[k1 + 1], c2[k2], c2[k2 + 1])
                        if ix_pt and _pt_in_poly(ix_pt[0], ix_pt[1]):
                            node_id += 1
                            az_i = p1.get("azimuth_deg", 0)
                            az_j = p2.get("azimuth_deg", 0)
                            features.append({
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [round(ix_pt[0], 7), round(ix_pt[1], 7)]},
                                "properties": {
                                    "node_id": node_id,
                                    "intersection_type": "conjugate-fracture",
                                    "azimuth_a_deg": az_i,
                                    "azimuth_b_deg": az_j,
                                    "conjugate_angle_deg": round(abs(az_i - az_j), 1),
                                },
                            })
        return {"type": "FeatureCollection", "features": features}

    parcel_poly = shapely_shape(parcel_geometry_wgs84)

    lin_lines: List[Tuple[Any, Dict]] = []
    for feat in lineaments_geojson.get("features", []):
        geom = feat.get("geometry")
        if geom and geom.get("type") == "LineString":
            coords = geom["coordinates"]
            if len(coords) >= 2:
                ls = ShapelyLineString(coords)
                lin_lines.append((ls, feat.get("properties", {})))

    drain_lines: List[Tuple[Any, Dict]] = []
    for feat in drainage_geojson.get("features", []):
        geom = feat.get("geometry")
        if geom and geom.get("type") == "LineString":
            coords = geom["coordinates"]
            if len(coords) >= 2:
                ls = ShapelyLineString(coords)
                drain_lines.append((ls, feat.get("properties", {})))

    def _add_point(pt, int_type, props):
        nonlocal node_id
        if pt.is_empty:
            return
        if pt.geom_type == "Point":
            if parcel_poly.contains(pt) or parcel_poly.touches(pt):
                node_id += 1
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(pt.x, 7), round(pt.y, 7)],
                    },
                    "properties": {
                        "node_id": node_id,
                        "intersection_type": int_type,
                        **props,
                    },
                })
        elif pt.geom_type == "MultiPoint":
            for p in pt.geoms:
                _add_point(p, int_type, props)
        elif pt.geom_type in ("GeometryCollection",):
            for g in pt.geoms:
                _add_point(g, int_type, props)

    # (a) Lineament x Drainage
    for lin_geom, lin_props in lin_lines:
        for drain_geom, drain_props in drain_lines:
            try:
                ix = lin_geom.intersection(drain_geom)
            except Exception:
                continue
            props = {
                "lineament_azimuth_deg": lin_props.get("azimuth_deg", 0),
                "lineament_length_m": lin_props.get("length_m", 0),
                "stream_order": drain_props.get("stream_order", drain_props.get("strahler_order", 0)),
                "strahler_order": drain_props.get("strahler_order", drain_props.get("stream_order", 0)),
            }
            _add_point(ix, "lineament-drainage", props)

    # (b) Conjugate Lineament x Lineament
    for i in range(len(lin_lines)):
        for j in range(i + 1, len(lin_lines)):
            try:
                ix = lin_lines[i][0].intersection(lin_lines[j][0])
            except Exception:
                continue
            az_i = lin_lines[i][1].get("azimuth_deg", 0)
            az_j = lin_lines[j][1].get("azimuth_deg", 0)
            props = {
                "azimuth_a_deg": az_i,
                "azimuth_b_deg": az_j,
                "conjugate_angle_deg": round(abs(az_i - az_j), 1),
            }
            _add_point(ix, "conjugate-fracture", props)

    logger.info("Identified %d fracture intersection nodes inside parcel", len(features))
    return {"type": "FeatureCollection", "features": features}


# ======================================================================
# 8. 12-Band Feature Stack Assembly
# ======================================================================

def build_feature_stack(
    twi: np.ndarray,
    flow_acc: np.ndarray,
    lineaments_geojson: Dict[str, Any],
    intersections_geojson: Dict[str, Any],
    s1_vv_db: np.ndarray,
    s1_vh_db: np.ndarray,
    s2_nir: np.ndarray,
    s2_swir: np.ndarray,
    s2_green: np.ndarray,
    plan_curvature: np.ndarray,
    slope: np.ndarray,
    ksat_value: float,
    insar_velocity_grid: np.ndarray,
    utm_transform: Any,
    cell_size: float = 10.0,
    stream_threshold: int = 100,
) -> np.ndarray:
    """
    Assemble the aligned (12, H, W) float32 feature array:
      Band  0: twi
      Band  1: dist_to_drainage (m)
      Band  2: lineament_density (km/km^2 in 100m circular kernel)
      Band  3: dist_to_intersection (m)
      Band  4: s1_vv_db
      Band  5: s1_vh_vv_ratio (linear)
      Band  6: s2_ndmi = (NIR - SWIR)/(NIR + SWIR)
      Band  7: s2_ndwi = (Green - NIR)/(Green + NIR)
      Band  8: curvature (plan curvature)
      Band  9: slope (degrees)
      Band 10: k_sat (m/day scalar broadcast)
      Band 11: insar_velocity (mm/yr resampled to grid)
    """
    H, W = twi.shape

    # Band 1: dist_to_drainage
    stream_mask = flow_acc >= stream_threshold
    dist_drain = _edt_distance(stream_mask, cell_size)

    # Band 2: lineament_density in km/km^2 within 100m circular kernel
    lin_density = compute_lineament_density_km_km2(
        lineaments_geojson, (H, W), utm_transform, cell_size=cell_size, radius_m=100.0
    )

    # Band 3: dist_to_intersection
    ix_mask = np.zeros((H, W), dtype=bool)
    for feat in intersections_geojson.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"]
            zone = int((lon + 180.0) / 6.0) + 1
            epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
            if pyproj is not None:
                tr = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
                ux, uy = tr.transform(lon, lat)
            else:
                from backend.services.satellite_fetcher import _wgs84_to_utm
                ux, uy = _wgs84_to_utm(lon, lat, epsg)
            try:
                col = int(round((ux - utm_transform.c) / utm_transform.a))
                row = int(round((uy - utm_transform.f) / utm_transform.e))
            except AttributeError:
                col, row = 0, 0
            if 0 <= row < H and 0 <= col < W:
                ix_mask[row, col] = True

    if not ix_mask.any():
        ix_mask[H // 2, W // 2] = True
    dist_ix = _edt_distance(ix_mask, cell_size)

    # Band 5: VH/VV linear ratio
    vv_lin = np.power(10.0, s1_vv_db / 10.0)
    vh_lin = np.power(10.0, s1_vh_db / 10.0) if s1_vh_db is not None else vv_lin * 0.1
    vh_vv_ratio = np.where(vv_lin > 1e-10, vh_lin / vv_lin, 0.01).astype(np.float32)

    # Band 6: NDMI = (NIR - SWIR) / (NIR + SWIR)
    ndmi = ((s2_nir - s2_swir) / (s2_nir + s2_swir + 1e-10)).astype(np.float32)
    ndmi = np.clip(ndmi, -1.0, 1.0)

    # Band 7: NDWI = (Green - NIR) / (Green + NIR)
    ndwi = ((s2_green - s2_nir) / (s2_green + s2_nir + 1e-10)).astype(np.float32)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    # Band 9: slope in degrees
    slope_deg = np.degrees(slope).astype(np.float32)

    # Band 10: k_sat
    ksat_grid = np.full((H, W), ksat_value, dtype=np.float32)

    # Band 11: InSAR velocity resampled to (H, W)
    if insar_velocity_grid.shape != (H, W):
        iy = np.linspace(0, insar_velocity_grid.shape[0] - 1, H)
        ix = np.linspace(0, insar_velocity_grid.shape[1] - 1, W)
        iy_grid, ix_grid = np.meshgrid(iy, ix, indexing="ij")
        iy0 = np.floor(iy_grid).astype(int)
        ix0 = np.floor(ix_grid).astype(int)
        iy1 = np.minimum(iy0 + 1, insar_velocity_grid.shape[0] - 1)
        ix1 = np.minimum(ix0 + 1, insar_velocity_grid.shape[1] - 1)
        fy = iy_grid - iy0
        fx = ix_grid - ix0
        insar_resized = (
            insar_velocity_grid[iy0, ix0] * (1 - fy) * (1 - fx)
            + insar_velocity_grid[iy1, ix0] * fy * (1 - fx)
            + insar_velocity_grid[iy0, ix1] * (1 - fy) * fx
            + insar_velocity_grid[iy1, ix1] * fy * fx
        ).astype(np.float32)
    else:
        insar_resized = insar_velocity_grid.astype(np.float32)

    stack = np.stack([
        twi,                # 0
        dist_drain,         # 1
        lin_density,        # 2
        dist_ix,            # 3
        s1_vv_db,           # 4
        vh_vv_ratio,        # 5
        ndmi,               # 6
        ndwi,               # 7
        plan_curvature,     # 8
        slope_deg,          # 9
        ksat_grid,          # 10
        insar_resized,      # 11
    ], axis=0).astype(np.float32)

    return stack


# ======================================================================
# 9. Main Hydrogeology Engine Orchestrator
# ======================================================================

class HydroGeologyEngine:
    """End-to-end parcel-scale hydrogeological analysis engine."""

    def __init__(self, cell_size: float = 10.0, stream_threshold: int = 100):
        self.cell_size = cell_size
        self.stream_threshold = stream_threshold

    async def run_hydro_analysis(
        self,
        parcel_geometry: Dict[str, Any],
        buffer_m: float,
        satellite_stack: SatelliteDataStack,
        insar_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute complete hydrogeological workflow:
          1. Pit filling with skimage.morphology.reconstruction
          2. D8 flow direction and accumulation
          3. Strahler stream extraction with smooth LineStrings
          4. Terrain derivatives and Topographic Wetness Index (TWI)
          5. Multi-azimuth SAR+DEM structural lineaments
          6. Subsurface fracture-drainage intersection pinpointing
          7. Aligned 12-band float32 feature stack assembly
        """
        import asyncio

        dem = satellite_stack.dem
        utm_transform = satellite_stack.utm_transform
        utm_epsg = satellite_stack.utm_crs if isinstance(satellite_stack.utm_crs, int) else 32643
        H, W = satellite_stack.grid_shape

        # 1-3. Hydrology in separate thread
        def _hydrology():
            filled = fill_depressions(dem, epsilon=0.001)
            fdir = compute_d8_flow_direction(filled, self.cell_size)
            facc = compute_flow_accumulation(fdir)
            drainage = extract_strahler_streams(
                fdir, facc, self.cell_size, self.stream_threshold,
                utm_transform, utm_epsg,
            )
            return filled, fdir, facc, drainage

        filled, fdir, facc, drainage_geojson = await asyncio.to_thread(_hydrology)

        # 4. Terrain derivatives & TWI
        slope, plan_curv, prof_curv, twi = compute_terrain_derivatives(
            filled, facc, self.cell_size,
        )

        # 5. Multi-azimuth structural lineaments
        lineaments_geojson = extract_lineaments(
            dem=dem,
            s1_vv_db=satellite_stack.s1_vv_db,
            s1_vh_db=satellite_stack.s1_vh_db,
            s2_nir=satellite_stack.s2_nir,
            s2_red=satellite_stack.s2_red,
            utm_transform=utm_transform,
            utm_epsg=utm_epsg,
            parcel_coords_wgs84=parcel_geometry.get("coordinates", []),
            cell_size=self.cell_size,
        )

        # 6. Intersections within parcel boundary
        intersections_geojson = find_intersections(
            lineaments_geojson, drainage_geojson, parcel_geometry,
        )

        # 7. 12-band feature stack
        ksat_value = insar_result.get("soil_properties", {}).get("ksat_m_day", 1.0)
        insar_vel = insar_result.get("insar_velocity_grid", np.zeros((64, 64), dtype=np.float32))
        if not isinstance(insar_vel, np.ndarray):
            insar_vel = np.zeros((64, 64), dtype=np.float32)

        feature_stack = build_feature_stack(
            twi=twi,
            flow_acc=facc,
            lineaments_geojson=lineaments_geojson,
            intersections_geojson=intersections_geojson,
            s1_vv_db=satellite_stack.s1_vv_db,
            s1_vh_db=satellite_stack.s1_vh_db,
            s2_nir=satellite_stack.s2_nir,
            s2_swir=satellite_stack.s2_swir,
            s2_green=satellite_stack.s2_green,
            plan_curvature=plan_curv,
            slope=slope,
            ksat_value=ksat_value,
            insar_velocity_grid=insar_vel,
            utm_transform=utm_transform,
            cell_size=self.cell_size,
            stream_threshold=self.stream_threshold,
        )

        intersection_candidates = []
        for feat in intersections_geojson.get("features", []):
            coords = feat["geometry"]["coordinates"]
            intersection_candidates.append({
                "lon": coords[0],
                "lat": coords[1],
                "type": feat["properties"].get("intersection_type", "unknown"),
                "stream_order": feat["properties"].get("stream_order", feat["properties"].get("strahler_order", 0)),
                "strahler_order": feat["properties"].get("strahler_order", feat["properties"].get("stream_order", 0)),
                "azimuth_deg": feat["properties"].get(
                    "lineament_azimuth_deg",
                    feat["properties"].get("azimuth_a_deg", 0),
                ),
            })

        terrain_stats = {
            "elevation_min_m": round(float(np.nanmin(dem)), 1),
            "elevation_max_m": round(float(np.nanmax(dem)), 1),
            "elevation_mean_m": round(float(np.nanmean(dem)), 1),
            "slope_mean_deg": round(float(np.nanmean(np.degrees(slope))), 2),
            "slope_max_deg": round(float(np.nanmax(np.degrees(slope))), 2),
            "twi_mean": round(float(np.nanmean(twi)), 2),
            "twi_max": round(float(np.nanmax(twi)), 2),
            "stream_cells_count": int(np.sum(facc >= self.stream_threshold)),
            "lineament_count": len(lineaments_geojson.get("features", [])),
            "intersection_count": len(intersections_geojson.get("features", [])),
        }

        return {
            "drainage_geojson": drainage_geojson,
            "lineaments_geojson": lineaments_geojson,
            "intersections_geojson": intersections_geojson,
            "feature_stack": feature_stack,
            "intersection_candidates": intersection_candidates,
            "terrain_stats": terrain_stats,
            "filled_dem": filled,
            "flow_accumulation": facc,
            "slope": slope,
            "plan_curvature": plan_curv,
            "profile_curvature": prof_curv,
            "twi": twi,
        }
