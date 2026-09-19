# =============================================================================
# rem_ingest/common.py — shared utilities for MonWatch-CLI satellite ingest
#
# (C) 2025-2026 PWARDS-weather
# SPDX-License-Identifier: Apache-2.0 OR GPL-3.0-or-later
#
# Dual-licensed. You may use, modify, and distribute this package under the
# terms of EITHER the Apache License, Version 2.0, or the GNU General Public
# License, Version 3.0 or later — not both. See LICENSE.txt in the project
# root for the full license texts and the copyright notice.
# =============================================================================
import os
import io
import logging
import threading
import tempfile

import numpy as np
import s3fs
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import matplotlib.colors as mcolors
from PIL import Image
import pyproj
from pyresample import AreaDefinition

fs = s3fs.S3FileSystem(
    anon=True,
    default_block_size=256 * 1024,
    config_kwargs={
        "max_pool_connections": 256,
        "connect_timeout": 10,
        "read_timeout": 120,
        "retries": {"max_attempts": 5},
        "s3": {"payload_signing_enabled": False},
    },
)

_download_session = requests.Session()
_dl_retries = Retry(total=5, backoff_factor=0.5,
                    status_forcelist=[500, 502, 503, 504])
_download_session.mount(
    "https://",
    HTTPAdapter(pool_connections=64, pool_maxsize=128, max_retries=_dl_retries),
)

BASIN_ALIASES = {
    "WP": "WP", "WPAC": "WP", "WESTPAC": "WP", "WESTERNPACIFIC": "WP",
    "EP": "EP", "EPAC": "EP", "EASTPAC": "EP", "EASTERNPACIFIC": "EP",
    "CP": "CP", "CPAC": "CP", "CENTRALPACIFIC": "CP",
    "AL": "AL", "ATL": "AL", "NATL": "AL", "NA": "AL", "NORTHATLANTIC": "AL",
    "IO": "IO", "NIO": "IO", "NORTHINDIAN": "IO",
    "SH": "SH", "SHEM": "SH", "SOUTHERNHEMISPHERE": "SH",
    "SI": "SI", "SIO": "SI", "SOUTHINDIAN": "SI",
    "AU": "AU", "AUS": "AU",
    "SP": "SP", "SOUTHPAC": "SP", "SOUTHPACIFIC": "SP",
}
_WPAC_LAT_MIN, _WPAC_LAT_MAX = -3.0, 30.0
_WPAC_LON_MIN, _WPAC_LON_MAX = 100.0, 180.0


def normalize_basin(value):
    key = (value or "").strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    return BASIN_ALIASES.get(key, key)


def storm_in_wpac(storm):
    lat = storm.get("latitude")
    lon = storm.get("longitude")
    if lat is None or lon is None:
        return False
    lon_n = ((float(lon) + 180.0) % 360.0) - 180.0
    return (_WPAC_LAT_MIN <= float(lat) <= _WPAC_LAT_MAX
            and _WPAC_LON_MIN <= lon_n <= _WPAC_LON_MAX)

OTT_nodes = [
    (0.0, "#860684"),
    ((183.15 - 173.15) / 140, "#860684"),
    ((192.15 - 173.15) / 140, "#E664BC"),
    ((192.16 - 173.15) / 140, "#D8D8D8"),
    ((203.15 - 173.15) / 140, "#000000"),
    ((213.15 - 173.15) / 140, "#FF0000"),
    ((223.15 - 173.15) / 140, "#FFFF00"),
    ((233.15 - 173.15) / 140, "#00FF00"),
    ((243.15 - 173.15) / 140, "#000064"),
    ((253.15 - 173.15) / 140, "#00FFFF"),
    ((253.16 - 173.15) / 140, "#C6C6C6"),
    ((303.15 - 173.15) / 140, "#000000"),
    (1.0, "#000000"),
]

OTT2_nodes = [
    (0.0, "#000000"),
    (10.0 / 150.0, "#000000"),
    (70.0 / 150.0, "#c8c8c8"),
    (70.0 / 150.0, "#00fafa"),
    (80.0 / 150.0, "#000064"),
    (90.0 / 150.0, "#00fa00"),
    (100.0 / 150.0, "#fafa00"),
    (110.0 / 150.0, "#fa0000"),
    (120.0 / 150.0, "#000000"),
    (130.0 / 150.0, "#e1e1e1"),
    (130.0 / 150.0, "#fa7daf"),
    (140.0 / 150.0, "#640064"),
    (140.0 / 150.0, "#fafa00"),
    (1.0, "#000000"),
]

DVORAK_nodes = [
    (0.0, "#585858"),
    ((193.15 - 173.15) / 140, "#585858"),
    ((193.16 - 173.15) / 140, "#888888"),
    ((198.15 - 173.15) / 140, "#888888"),
    ((198.16 - 173.15) / 140, "#FFFFFF"),
    ((204.15 - 173.15) / 140, "#FFFFFF"),
    ((204.16 - 173.15) / 140, "#000000"),
    ((210.15 - 173.15) / 140, "#000000"),
    ((210.16 - 173.15) / 140, "#A0A0A0"),
    ((220.15 - 173.15) / 140, "#A0A0A0"),
    ((220.16 - 173.15) / 140, "#707070"),
    ((232.15 - 173.15) / 140, "#707070"),
    ((232.16 - 173.15) / 140, "#404040"),
    ((243.15 - 173.15) / 140, "#404040"),
    ((243.16 - 173.15) / 140, "#D2D2D2"),
    ((282.15 - 173.15) / 140, "#3A3A3A"),
    ((282.16 - 173.15) / 140, "#FAFAFA"),
    ((299.15 - 173.15) / 140, "#2A2A2A"),
    (1.0, "#000000"),
]

INFRARED_HIM_nodes = [
    (0.0, "#FFFFFF"),
    (0.5, "#808080"),
    (1.0, "#000000"),
]

hotspot_SIR_nodes = [
    (0, "#000000"),
    (0.5, "#FF0000"),
    (0.66, "#FFFF00"),
    (1, "#FFFFFF"),
]

def _build_sandwich_lut():
    lut = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i * 150 / 255 - 100
        if t <= -72:
            rgb = np.array([251/255.0, 5/255.0, 0.0], dtype=np.float32)
        elif t <= -65:
            f = (t + 72) / 7.0
            rgb = (np.array([251/255.0, 5/255.0, 0.0], dtype=np.float32) * (1 - f)
                   + np.array([1.0, 0.5, 0.0], dtype=np.float32) * f)
        elif t <= -58:
            f = (t + 65) / 7.0
            rgb = (np.array([1.0, 0.5, 0.0], dtype=np.float32) * (1 - f)
                   + np.array([1.0, 1.0, 0.0], dtype=np.float32) * f)
        elif t <= -52:
            f = (t + 58) / 6.0
            rgb = (np.array([0.0, 1.0, 0.0], dtype=np.float32) * (1 - f)
                   + np.array([0.0, 1.0, 1.0], dtype=np.float32) * f)
        elif t <= -32:
            f = (t + 52) / 20.0
            rgb = (np.array([0.0, 1.0, 1.0], dtype=np.float32) * (1 - f)
                   + np.array([14/255.0, 14/255.0, 146/255.0], dtype=np.float32) * f)
        elif t <= -25:
            rgb = np.array([14/255.0, 14/255.0, 146/255.0], dtype=np.float32)
        else:
            f = np.clip((t + 25) / 75.0, 0.0, 1.0)
            rgb = (np.array([0.5, 0.5, 0.5], dtype=np.float32) * (1 - f)
                   + np.array([0.0, 0.0, 0.0], dtype=np.float32) * f)
        lut[i] = np.clip(rgb, 0.0, 1.0)
    return lut

_SANDWICH_IR_LUT = _build_sandwich_lut()


def _build_dvorak_lut():
    lut = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i * 150 / 255 - 100
        if t <= -85:
            rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif t <= -80:
            f = (t + 85) / 5.0
            rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32) * f
        elif t <= -70:
            f = (t + 80) / 10.0
            rgb = (np.array([1.0, 1.0, 1.0], dtype=np.float32) * (1 - f)
                   + np.array([1.0, 0.0, 0.0], dtype=np.float32) * f)
        elif t <= -60:
            f = (t + 70) / 10.0
            rgb = (np.array([1.0, 0.0, 0.0], dtype=np.float32) * (1 - f)
                   + np.array([1.0, 0.5, 0.0], dtype=np.float32) * f)
        elif t <= -50:
            f = (t + 60) / 10.0
            rgb = (np.array([1.0, 0.5, 0.0], dtype=np.float32) * (1 - f)
                   + np.array([1.0, 1.0, 0.0], dtype=np.float32) * f)
        elif t <= -40:
            f = (t + 50) / 10.0
            rgb = (np.array([1.0, 1.0, 0.0], dtype=np.float32) * (1 - f)
                   + np.array([0.0, 1.0, 0.0], dtype=np.float32) * f)
        elif t <= -30:
            f = (t + 40) / 10.0
            rgb = (np.array([0.0, 1.0, 0.0], dtype=np.float32) * (1 - f)
                   + np.array([0.0, 1.0, 1.0], dtype=np.float32) * f)
        elif t <= -20:
            f = (t + 30) / 10.0
            rgb = (np.array([0.0, 1.0, 1.0], dtype=np.float32) * (1 - f)
                   + np.array([0.0, 0.0, 1.0], dtype=np.float32) * f)
        elif t <= -10:
            f = (t + 20) / 10.0
            rgb = (np.array([0.0, 0.0, 1.0], dtype=np.float32) * (1 - f)
                   + np.array([0.0, 0.0, 0.5], dtype=np.float32) * f)
        else:
            f = np.clip((t + 10) / 40.0, 0.0, 1.0)
            rgb = (np.array([0.0, 0.0, 0.5], dtype=np.float32) * (1 - f)
                   + np.array([0.5, 0.5, 0.5], dtype=np.float32) * f)
        lut[i] = np.clip(rgb, 0.0, 1.0)
    return lut

_DVORAK_IR_LUT = _build_dvorak_lut()


def sandwich_ir_lookup(bt_kelvin):
    celsius = np.asarray(bt_kelvin, dtype=np.float32) - 273.15
    idx = np.clip((celsius + 100) * 255 / 150, 0, 255).astype(np.uint8)
    return _SANDWICH_IR_LUT[idx]


def dvorak_ir_lookup(bt_kelvin):
    celsius = np.asarray(bt_kelvin, dtype=np.float32) - 273.15
    idx = np.clip((celsius + 100) * 255 / 150, 0, 255).astype(np.uint8)
    return _DVORAK_IR_LUT[idx]


def dvorak_cmap():
    n = 1501
    colors = np.zeros((n, 3), dtype=np.float64)
    temps = np.linspace(-100, 50, n)
    for i, t in enumerate(temps):
        if t > 9:        colors[i] = [0.45, 0.45, 0.45]
        elif t > -30:    colors[i] = [0.90, 0.90, 0.90]
        elif t > -41:    colors[i] = [0.20, 0.20, 0.20]
        elif t > -53:    colors[i] = [0.50, 0.50, 0.50]
        elif t > -63:    colors[i] = [0.75, 0.75, 0.75]
        elif t > -69:    colors[i] = [0.0, 0.0, 0.0]
        elif t > -75:    colors[i] = [1.0, 1.0, 1.0]
        elif t > -81:    colors[i] = [0.50, 0.50, 0.50]
        else:            colors[i] = [0.20, 0.20, 0.20]
    return mcolors.ListedColormap(colors, name="dvorak")

AHI_TO_ABI = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
              11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16}

AHI_TO_FCI = {1: "vis_04", 2: "vis_05", 3: "vis_06", 4: "vis_08", 5: "nir_16",
              6: "nir_22", 7: "ir_38", 8: None, 9: None, 10: None,
              11: "ir_87", 12: "ir_97", 13: "ir_105", 14: "ir_123",
              15: "ir_123", 16: "ir_133"}

AHI_TO_MTSAT = {1: "VIS", 2: "VIS", 3: "VIS", 4: "VIS",
                7: "IR4", 9: "IR3", 13: "IR1", 14: "IR2"}

def normalize_reflectance(arr):
    return np.clip(arr / 100.0 if np.nanmax(arr) > 1.0 else arr, 0.0, 1.0)


def linear_normalize(arr, vmin, vmax, gamma=1.0, invert=False):
    data = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0)
    if vmin is None: vmin = float(np.min(data))
    if vmax is None: vmax = float(np.max(data))
    if vmax <= vmin: vmax = vmin + 1e-6
    out = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, 1.0 / gamma)
    if invert:
        out = 1.0 - out
    return np.clip(out, 0.0, 1.0)


def stack_rgb(r, g, b):
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255).astype(np.uint8)


def false_color_rgb(vis, ir, sza, advanced=False):
    vis_raw = np.asarray(vis, dtype=np.float32)
    ir_raw = np.asarray(ir, dtype=np.float32)
    h, w = vis_raw.shape
    ir_raw = np.where(np.isnan(ir_raw), 300.0, ir_raw)
    sza = np.asarray(sza, dtype=np.float32)
    if sza.shape != (h, w):
        sza = np.zeros((h, w), dtype=np.float32)
    sza = np.nan_to_num(sza, nan=90.0)

    vis_norm = normalize_reflectance(np.nan_to_num(vis_raw, nan=0.0))

    cos_sza = np.clip(np.cos(np.radians(sza)), 0.25, 1.0)
    path_sun = 1.0 / cos_sza
    vis_bright = vis_norm * path_sun

    day_weight = np.clip((90.0 - sza) / 5.0, 0.0, 1.0)
    night_weight = 1.0 - day_weight

    ir_norm = np.clip((323.15 - ir_raw) / (313.15 - 173.15), 0.0, 1.0)
    ir_layer = np.power(ir_norm, 1.1)

    if advanced:
        r = np.power(vis_bright, 0.88) * 0.9 + (ir_layer * night_weight * 0.5) + (ir_layer * day_weight * 0.25)
        g = vis_bright * 0.75 + (ir_layer * night_weight * 0.6) + (ir_layer * day_weight * 0.44)
        b = vis_bright * 0.1 + (ir_layer * day_weight) + (ir_layer * night_weight)
    else:
        r = vis_bright + (ir_layer * night_weight * 0.45) + (ir_layer * day_weight * 0.2)
        g = vis_bright * 0.9 + (ir_layer * night_weight * 0.45) + (ir_layer * day_weight * 0.25)
        b = vis_bright * 0.1 + ir_layer

    return (np.clip(r, 0.0, 1.0),
            np.clip(g, 0.0, 1.0),
            np.clip(b, 0.0, 1.0))


def resize_like(arr, target_shape):
    h, w = arr.shape
    th, tw = target_shape
    if (h, w) == (th, tw):
        return arr
    mask = np.isfinite(arr)
    fill = np.nanmin(arr) if np.any(mask) else 0.0
    data = np.where(mask, arr, fill)
    img = Image.fromarray(data.astype(np.float32))
    img = img.resize((tw, th))
    out = np.asarray(img).astype(np.float32)
    if np.any(~mask):
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img = mask_img.resize((tw, th))
        out = np.where(np.asarray(mask_img) > 0.5, out, np.nan)
    return out


def upscale_rgb(r, g, b, target_area):
    h, w = r.shape
    tw, th = target_area.y_size, target_area.x_size
    if (w, h) == (tw, th):
        return r, g, b
    rgb = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)], axis=-1)
    img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
    img = img.resize((tw, th))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[..., 0], arr[..., 1], arr[..., 2]


def reduced_area(area, source_pixel_m=500.0):
    w, h = area.x_size, area.y_size
    extent = area.area_extent
    width_m = extent[2] - extent[0]
    height_m = extent[3] - extent[1]
    if width_m <= 0 or height_m <= 0:
        return area
    target_w = max(int(round(width_m / source_pixel_m)), 1)
    target_h = max(int(round(height_m / source_pixel_m)), 1)
    if target_w >= w and target_h >= h:
        return area
    scale = max(w / float(target_w), h / float(target_h))
    new_w = max(int(round(w / scale)), 1)
    new_h = max(int(round(h / scale)), 1)
    return AreaDefinition(area.area_id, area.description, area.proj_id,
                          area.proj_dict, new_w, new_h, extent)


def area_with_shape(area, shape):
    if area is None or shape is None or len(shape) < 2:
        return area
    ny, nx = int(shape[0]), int(shape[1])
    if area.x_size == nx and area.y_size == ny:
        return area
    return AreaDefinition(area.area_id, area.description, area.proj_id,
                          area.proj_dict, nx, ny, area.area_extent)


def align_result_to_area(result, area):
    if result is None or area is None:
        return result
    target = (int(area.y_size), int(area.x_size))
    out = []
    for arr in result:
        if arr is None:
            out.append(None)
        elif getattr(arr, "ndim", 0) == 2 and arr.shape != target:
            out.append(resize_like(arr, target))
        else:
            out.append(arr)
    return tuple(out)

_GEOS_R_KM = 6371.0
_GEOS_H_KM = 35786.0


def project_is_native_geos(name):
    key = (name or "flat").strip().lower().replace("-", "_").replace(" ", "_")
    return key in ("geos", "geostationary", "geo", "native", "nat")


def geos_area_for_crop(lat, lon, half_lon, half_lat, sat_lon, width, height,
                       sat_height=35785831.0, a=6378137.0, b=6356752.31414):
    lon_norm = ((float(lon) + 180) % 360) - 180
    sat_lon = float(sat_lon)
    p = pyproj.Proj(proj="geos", h=sat_height, lon_0=sat_lon, a=a, b=b, sweep="x")
    xs, ys = [], []
    n = 24
    lons = np.linspace(lon_norm - half_lon, lon_norm + half_lon, n)
    lats = np.linspace(lat - half_lat, lat + half_lat, n)
    edge = []
    for lo in lons:
        edge.append((lo, lat - half_lat))
        edge.append((lo, lat + half_lat))
    for la in lats:
        edge.append((lon_norm - half_lon, la))
        edge.append((lon_norm + half_lon, la))
    edge.append((lon_norm, lat))
    for clo, cla in edge:
        x, y = p(float(clo), float(cla))
        if np.isfinite(x) and np.isfinite(y) and abs(x) < 1e20 and abs(y) < 1e20:
            xs.append(float(x)); ys.append(float(y))
    if len(xs) < 3:
        cx, cy = p(lon_norm, lat)
        if not (np.isfinite(cx) and abs(cx) < 1e20):
            cx, cy = 0.0, 0.0
        m = max(half_lon, half_lat) * 1.2e5
        xs = [cx - m, cx + m]; ys = [cy - m, cy + m]
    pad = 0.01 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid geos crop extent")
    proj_dict = {"proj": "geos", "lon_0": sat_lon, "h": sat_height,
                 "a": a, "b": b, "sweep": "x", "units": "m"}
    return AreaDefinition("storm_crop_geos", "Storm Crop (GEOS)", "geos",
                          proj_dict, int(width), int(height),
                          (x_min, y_min, x_max, y_max))


def sat_subpoint_lon(sat_source):
    if sat_source == "mtg":                             return -0.3
    if sat_source in ("goes", "goes16", "goes19"):      return -75.0
    if sat_source in ("goes17", "goes18"):              return -137.0
    if sat_source == "gk2a":                            return 128.2
    if sat_source in ("mtsat", "mtsat2"):               return 145.0
    if sat_source == "mtsat1":                          return 140.0
    return 140.7


def standard_fulldisk_geos_area(sat_source, nx=None, ny=None):
    sat_lon = sat_subpoint_lon(sat_source)
    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        h = 35786023.0; half = 5434894.885056
        nx = int(nx or 5424); ny = int(ny or 5424); sweep = "x"
    elif sat_source == "mtg":
        h = 35786400.0; half = 5568000.0
        nx = int(nx or 5568); ny = int(ny or 5568); sweep = "y"
    else:
        h = 35785831.0; half = 5499999.901531426
        nx = int(nx or 5500); ny = int(ny or 5500); sweep = "x"
    proj_dict = {"proj": "geos", "lon_0": float(sat_lon), "h": h,
                 "a": 6378137.0, "b": 6356752.31414, "sweep": sweep,
                 "units": "m"}
    return AreaDefinition("fldk_geos", "Full Disk (native GEOS)", "geos",
                          proj_dict, nx, ny, (-half, -half, half, half))

def get_required_segments(bounds, sat_lon=140.7, buffer=False,
                          single_segment_max_gap_frac=0.03):
    lat, lon, crop_deg = bounds
    if crop_deg >= 50.0:
        return [f"S{i:02d}" for i in range(1, 11)]

    p_geos = pyproj.Proj(proj='geos', h=35785863.0, lon_0=sat_lon, sweep='x')
    n = 24
    lats = np.linspace(lat - crop_deg, lat + crop_deg, n)
    lons = np.linspace(lon - crop_deg, lon + crop_deg, n)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    _, y = p_geos(lon_grid, lat_grid)
    valid_y = y[y < 1e20]
    if len(valid_y) == 0:
        return [f"S{i:02d}" for i in range(1, 11)]
    y_max = float(np.nanmax(valid_y)); y_min = float(np.nanmin(valid_y))
    y_extent = 5434894.885056
    y_margin = 0.003 * 2 * y_extent
    norm_y_min = (y_extent - (y_max + y_margin)) / (2 * y_extent)
    norm_y_max = (y_extent - (y_min - y_margin)) / (2 * y_extent)
    seg_start = int(np.clip(np.floor(norm_y_min * 10), 0, 9)) + 1
    seg_end   = int(np.clip(np.floor(norm_y_max * 10), 0, 9)) + 1
    if seg_end > seg_start:
        seg_bounds = [y_extent * (1 - 2 * k / 10) for k in range(11)]
        crop_span_m = max(y_max - y_min, 1.0)
        best_seg = None; best_gap_m = None
        for k in range(seg_start, seg_end + 1):
            seg_top = seg_bounds[k - 1]; seg_bot = seg_bounds[k]
            overlap_m = max(min(y_max, seg_top) - max(y_min, seg_bot), 0.0)
            if overlap_m <= 0:
                continue
            gap_m = crop_span_m - overlap_m
            if best_gap_m is None or gap_m < best_gap_m:
                best_gap_m = gap_m; best_seg = k
        allowed_gap_m = single_segment_max_gap_frac * crop_span_m
        if best_seg is not None and best_gap_m <= allowed_gap_m:
            return [f"S{best_seg:02d}"]
    if buffer:
        seg_start = max(1, seg_start - 1)
        seg_end   = min(10, seg_end + 1)
    return [f"S{i:02d}" for i in range(seg_start, seg_end + 1)]

import datetime as _dt
def generate_time_slots(date_from, date_to, time_from, time_to):
    slots = []
    if not date_from:
        return slots
    start_date = _dt.datetime.strptime(date_from, "%Y%m%d").date()
    end_date = _dt.datetime.strptime(date_to, "%Y%m%d").date() if date_to else start_date
    start_hour = int(time_from[:2]) if time_from else 0
    start_minute = int(time_from[2:]) if time_from else 0
    if time_to:
        end_hour = int(time_to[:2]); end_minute = int(time_to[2:])
    else:
        now_utc = _dt.datetime.now(_dt.timezone.utc)
        end_hour = now_utc.hour; end_minute = now_utc.minute
    current_date = start_date
    while current_date <= end_date:
        for hour in range(24):
            for minute in range(0, 60, 10):
                dt = _dt.datetime.combine(current_date, _dt.time(hour, minute))
                if current_date == start_date:
                    if hour < start_hour or (hour == start_hour and minute < start_minute):
                        continue
                if current_date == end_date:
                    if hour > end_hour or (hour == end_hour and minute > end_minute):
                        continue
                slots.append(dt)
        current_date += _dt.timedelta(days=1)
    return slots


def get_latest_available_dt_generic(candidates):
    raise NotImplementedError