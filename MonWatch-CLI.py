#!/usr/bin/env python3
# =============================================================================
#
# MonWatch-CLI -- Automated Himawari-9/8, GK-2A, GOES, MTG, MTSAT Storm Imagery
# - A CLI version of MonWatch-UI for Servers and Automated Systems (PART OF
# THE PWARDS ECOSYSTEM -- "FREE SCIENCE FOR EVERYONE" -- OPENSOURCE TOOLS
#
# (C) 2025-2026 PWARDS-weather
# This project is dual-licensed under the Apache License, Version 2.0, and the 
# GNU General Public License, Version 3.0 (SAME AS MONWATCH-UI)
#
# =============================================================================

import os
import sys
import re
import gc
import bz2
import dask
import threading
import shutil
import tempfile
import argparse
import logging
import datetime
import io
import traceback
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import s3fs
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import xarray as xr
from satpy import Scene
from pyresample import AreaDefinition
from pyresample import kd_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
try:
    import pillow_avif
except ImportError:
    pass
import pyproj
import subprocess
import zipfile
import ftplib
import tarfile
import gzip

def create_mp4_from_frames(frame_paths, output_path, fps=30):
    if not frame_paths:
        logging.warning("No frames to create MP4")
        return False
    
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.warning("ffmpeg not found, cannot create MP4. Install ffmpeg for MP4 support.")
        return False

    frames = sorted(frame_paths)
    work_dir = tempfile.mkdtemp(prefix="mp4_frames_")
    try:
        staging = os.path.join(work_dir, "seq")
        os.makedirs(staging, exist_ok=True)
        for i, p in enumerate(frames):
            shutil.copyfile(p, os.path.join(staging, f"frame_{i:04d}.png"))
        pattern = os.path.join(staging, "frame_%04d.png")

        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
            "-vf", f"fps={fps},format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logging.info(f"Created MP4: {output_path}")
            return True
        else:
            logging.error(f"ffmpeg failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logging.error("ffmpeg timed out")
        return False
    except Exception as e:
        logging.error(f"Failed to create MP4: {e}")
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def create_mp4_from_png_bytes(png_frames, output_path, fps=30):
    if not png_frames:
        logging.warning("No frames to create MP4")
        return False

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.warning("ffmpeg not found, cannot create MP4. Install ffmpeg for MP4 support.")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe",
        "-probesize", "100M", "-analyzeduration", "1000000",
        "-framerate", str(fps), "-i", "pipe:0",
        "-vf", f"fps={fps},format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in png_frames:
                proc.stdin.write(frame)
                proc.stdin.flush()
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        _, stderr = proc.communicate(timeout=300)
        if proc.returncode == 0:
            logging.info(f"Created MP4: {output_path}")
            return True
        tail = stderr.decode(errors='ignore').strip().splitlines()
        logging.error(f"ffmpeg pipe failed ({proc.returncode}), retrying via temp-file staging: {tail[-1:] if tail else ''}")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        logging.error("ffmpeg (pipe) timed out, retrying via temp-file staging")
    except Exception as e:
        logging.error(f"Failed to create MP4 (pipe): {e}")

    return _create_mp4_from_png_bytes_staged(png_frames, output_path, fps=fps)


def _create_mp4_from_png_bytes_staged(png_frames, output_path, fps=30):
    work_dir = tempfile.mkdtemp(prefix="mp4_staged_")
    try:
        staging = os.path.join(work_dir, "seq")
        os.makedirs(staging, exist_ok=True)
        for i, blob in enumerate(png_frames):
            with open(os.path.join(staging, f"frame_{i:04d}.png"), "wb") as f:
                f.write(blob)
        pattern = os.path.join(staging, "frame_%04d.png")

        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
            "-vf", f"fps={fps},format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logging.info(f"Created MP4: {output_path}")
            return True
        logging.error(f"ffmpeg failed: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logging.error("ffmpeg timed out")
        return False
    except Exception as e:
        logging.error(f"Failed to create MP4: {e}")
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _sandwich_ir_lookup(bt_kelvin: np.ndarray) -> np.ndarray:
    celsius = np.asarray(bt_kelvin, dtype=np.float32) - 273.15
    idx = np.clip((celsius + 100) * 255 / 150, 0, 255).astype(np.uint8)
    return _SANDWICH_IR_LUT[idx]

_SANDWICH_IR_LUT = np.zeros((256, 3), dtype=np.float32)
for i in range(256):
    t = i * 150 / 255 - 100
    if t <= -72:
        rgb = np.array([251/255.0, 5/255.0, 0.0], dtype=np.float32)
    elif t <= -65:
        f = (t + 72) / 7.0
        rgb = np.array([251/255.0, 5/255.0, 0.0], dtype=np.float32) * (1-f) + np.array([1.0, 0.5, 0.0], dtype=np.float32) * f
    elif t <= -58:
        f = (t + 65) / 7.0
        rgb = np.array([1.0, 0.5, 0.0], dtype=np.float32) * (1-f) + np.array([1.0, 1.0, 0.0], dtype=np.float32) * f
    elif t <= -52:
        f = (t + 58) / 6.0
        rgb = np.array([0.0, 1.0, 0.0], dtype=np.float32) * (1-f) + np.array([0.0, 1.0, 1.0], dtype=np.float32) * f
    elif t <= -32:
        f = (t + 52) / 20.0
        rgb = np.array([0.0, 1.0, 1.0], dtype=np.float32) * (1-f) + np.array([14/255.0, 14/255.0, 146/255.0], dtype=np.float32) * f
    elif t <= -25:
        rgb = np.array([14/255.0, 14/255.0, 146/255.0], dtype=np.float32)
    else:
        f = np.clip((t + 25) / 75.0, 0.0, 1.0)
        rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32) * (1-f) + np.array([0.0, 0.0, 0.0], dtype=np.float32) * f
    _SANDWICH_IR_LUT[i] = np.clip(rgb, 0.0, 1.0)


def _normalize_reflectance(arr):
    return np.clip(arr / 100.0 if np.nanmax(arr) > 1.0 else arr, 0.0, 1.0)


def _linear_normalize(arr, vmin, vmax, gamma=1.0, invert=False):
    data = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0)
    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    out = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, 1.0 / gamma)
    if invert:
        out = 1.0 - out
    return np.clip(out, 0.0, 1.0)


def _stack_rgb(r, g, b):
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255).astype(np.uint8)


def _false_color_rgb(vis, ir, sza, advanced=False):
    vis_raw = np.asarray(vis, dtype=np.float32)
    ir_raw = np.asarray(ir, dtype=np.float32)
    h, w = vis_raw.shape
    ir_raw = np.where(np.isnan(ir_raw), 300.0, ir_raw)
    sza = np.asarray(sza, dtype=np.float32)
    if sza.shape != (h, w):
        sza = np.zeros((h, w), dtype=np.float32)
    sza = np.nan_to_num(sza, nan=90.0)

    vis_norm = _normalize_reflectance(np.nan_to_num(vis_raw, nan=0.0))

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

    return (np.clip(r, 0.0, 1.0), np.clip(g, 0.0, 1.0), np.clip(b, 0.0, 1.0))

try:
    from atcf import fetch_knackwx_atcf
except ImportError:
    import requests
    def fetch_knackwx_atcf():
        try:
            resp = requests.get("https://api.knackwx.com/atcf/v2", timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []

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

S3_HTTPS_BASE = "https://noaa-himawari9.s3.amazonaws.com"
_download_session = requests.Session()
_dl_retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
_download_session.mount("https://", HTTPAdapter(pool_connections=64, pool_maxsize=128, max_retries=_dl_retries))

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
    (1.0, "#000000")
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
    (1.0, "#000000")
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
    (1.0, "#000000")
]

BT_ENHANCED_nodes = [
    (0.0, "#000080"),
    (0.1, "#0000FF"),
    (0.2, "#00FFFF"),
    (0.3, "#00FF00"),
    (0.4, "#FFFF00"),
    (0.5, "#FF8000"),
    (0.6, "#FF0000"),
    (0.7, "#800000"),
    (0.8, "#FFFFFF"),
    (0.9, "#C0C0C0"),
    (1.0, "#808080"),
]

hotspot_SIR_nodes = [
    (0, "#000000"),
    (0.5, "#FF0000"),
    (0.66, "#FFFF00"),
    (1, "#FFFFFF"),
]

SANDWICH_IR_nodes = [
    (0.0, "#860684"),
    (0.15, "#E664BC"),
    (0.25, "#D8D8D8"),
    (0.35, "#000000"),
    (0.45, "#FF0000"),
    (0.55, "#FFFF00"),
    (0.65, "#00FF00"),
    (0.75, "#000064"),
    (0.85, "#00FFFF"),
    (0.9, "#C6C6C6"),
    (1.0, "#000000"),
]

DVORAK_IR_nodes = [
    (0.0, "#585858"),
    (0.15, "#888888"),
    (0.2, "#FFFFFF"),
    (0.25, "#000000"),
    (0.3, "#A0A0A0"),
    (0.4, "#707070"),
    (0.5, "#404040"),
    (0.6, "#D2D2D2"),
    (0.8, "#3A3A3A"),
    (0.85, "#FAFAFA"),
    (0.9, "#2A2A2A"),
    (1.0, "#000000"),
]

INFRARED_HIM_nodes = [
    (0.0, "#FFFFFF"),
    (0.5, "#808080"),
    (1.0, "#000000"),
]

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

def _normalize_basin(value):
    key = (value or "").strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    return BASIN_ALIASES.get(key, key)


def _filter_storms_by_basin(storms, basin_filter):
    if not basin_filter:
        return list(storms)
    return [s for s in storms if _normalize_basin(s.get("basin", "")) in basin_filter]

def _dvorak_ir_lookup(bt_kelvin: np.ndarray) -> np.ndarray:
    celsius = np.asarray(bt_kelvin, dtype=np.float32) - 273.15
    idx = np.clip((celsius + 100) * 255 / 150, 0, 255).astype(np.uint8)
    return _DVORAK_IR_LUT[idx]

_DVORAK_IR_LUT = np.zeros((256, 3), dtype=np.float32)
for i in range(256):
    t = i * 150 / 255 - 100 
    if t <= -85:
        rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    elif t <= -80:
        f = (t + 85) / 5.0
        rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32) * (1-f) + np.array([1.0, 1.0, 1.0], dtype=np.float32) * f
    elif t <= -70:
        f = (t + 80) / 10.0
        rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32) * (1-f) + np.array([1.0, 0.0, 0.0], dtype=np.float32) * f
    elif t <= -60:
        f = (t + 70) / 10.0
        rgb = np.array([1.0, 0.0, 0.0], dtype=np.float32) * (1-f) + np.array([1.0, 0.5, 0.0], dtype=np.float32) * f
    elif t <= -50:
        f = (t + 60) / 10.0
        rgb = np.array([1.0, 0.5, 0.0], dtype=np.float32) * (1-f) + np.array([1.0, 1.0, 0.0], dtype=np.float32) * f
    elif t <= -40:
        f = (t + 50) / 10.0
        rgb = np.array([1.0, 1.0, 0.0], dtype=np.float32) * (1-f) + np.array([0.0, 1.0, 0.0], dtype=np.float32) * f
    elif t <= -30:
        f = (t + 40) / 10.0
        rgb = np.array([0.0, 1.0, 0.0], dtype=np.float32) * (1-f) + np.array([0.0, 1.0, 1.0], dtype=np.float32) * f
    elif t <= -20:
        f = (t + 30) / 10.0
        rgb = np.array([0.0, 1.0, 1.0], dtype=np.float32) * (1-f) + np.array([0.0, 0.0, 1.0], dtype=np.float32) * f
    elif t <= -10:
        f = (t + 20) / 10.0
        rgb = np.array([0.0, 0.0, 1.0], dtype=np.float32) * (1-f) + np.array([0.0, 0.0, 0.5], dtype=np.float32) * f
    else:
        f = np.clip((t + 10) / 40.0, 0.0, 1.0)
        rgb = np.array([0.0, 0.0, 0.5], dtype=np.float32) * (1-f) + np.array([0.5, 0.5, 0.5], dtype=np.float32) * f
    _DVORAK_IR_LUT[i] = np.clip(rgb, 0.0, 1.0)


def _dvorak_cmap():
    n = 1501
    colors = np.zeros((n, 3), dtype=np.float64)
    temps = np.linspace(-100, 50, n)
    for i, t in enumerate(temps):
        if t > 9:
            colors[i] = [0.45, 0.45, 0.45]
        elif t > -30:
            colors[i] = [0.90, 0.90, 0.90]
        elif t > -41:
            colors[i] = [0.20, 0.20, 0.20]
        elif t > -53:
            colors[i] = [0.50, 0.50, 0.50]
        elif t > -63:
            colors[i] = [0.75, 0.75, 0.75]
        elif t > -69:
            colors[i] = [0.0, 0.0, 0.0]
        elif t > -75:
            colors[i] = [1.0, 1.0, 1.0]
        elif t > -81:
            colors[i] = [0.50, 0.50, 0.50]
        else:
            colors[i] = [0.20, 0.20, 0.20]
    return mcolors.ListedColormap(colors, name="dvorak")

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

    y_max = float(np.nanmax(valid_y))
    y_min = float(np.nanmin(valid_y))

    y_extent = 5434894.885056

    y_margin = 0.003 * 2 * y_extent

    norm_y_min = (y_extent - (y_max + y_margin)) / (2 * y_extent)
    norm_y_max = (y_extent - (y_min - y_margin)) / (2 * y_extent)

    seg_start = int(np.clip(np.floor(norm_y_min * 10), 0, 9)) + 1
    seg_end   = int(np.clip(np.floor(norm_y_max * 10), 0, 9)) + 1

    if seg_end > seg_start:
        seg_bounds = [y_extent * (1 - 2 * k / 10) for k in range(11)]
        crop_span_m = max(y_max - y_min, 1.0)

        best_seg = None
        best_gap_m = None
        for k in range(seg_start, seg_end + 1):
            seg_top = seg_bounds[k - 1]   
            seg_bot = seg_bounds[k]       
            overlap_m = max(min(y_max, seg_top) - max(y_min, seg_bot), 0.0)
            if overlap_m <= 0:
                continue
            gap_m = crop_span_m - overlap_m
            if best_gap_m is None or gap_m < best_gap_m:
                best_gap_m = gap_m
                best_seg = k

        allowed_gap_m = single_segment_max_gap_frac * crop_span_m
        if best_seg is not None and best_gap_m <= allowed_gap_m:
            return [f"S{best_seg:02d}"]

    if buffer:
        seg_start = max(1, seg_start - 1)
        seg_end   = min(10, seg_end + 1)

    return [f"S{i:02d}" for i in range(seg_start, seg_end + 1)]
    
def generate_time_slots(date_from, date_to, time_from, time_to):
    slots = []
    if not date_from:
        return slots
    
    start_date = datetime.datetime.strptime(date_from, "%Y%m%d").date()
    end_date = datetime.datetime.strptime(date_to, "%Y%m%d").date() if date_to else start_date
    
    start_hour = int(time_from[:2]) if time_from else 0
    start_minute = int(time_from[2:]) if time_from else 0
    if time_to:
        end_hour = int(time_to[:2])
        end_minute = int(time_to[2:])
    else:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        end_hour = now_utc.hour
        end_minute = now_utc.minute
    
    current_date = start_date
    while current_date <= end_date:
        for hour in range(24):
            for minute in range(0, 60, 10):
                dt = datetime.datetime.combine(current_date, datetime.time(hour, minute))

                if current_date == start_date:
                    if hour < start_hour or (hour == start_hour and minute < start_minute):
                        continue
                if current_date == end_date:
                    if hour > end_hour or (hour == end_hour and minute > end_minute):
                        continue
                
                slots.append(dt)
        current_date += datetime.timedelta(days=1)
    
    return slots

def discover_ahi_files(satellite, dt_obj, bands_list, segments, use_target=False, target_segment=None):
    if use_target:
        path = (
            f"{satellite}/AHI-L1b-Target/"
            f"{dt_obj.year:04d}/{dt_obj.month:02d}/{dt_obj.day:02d}/"
            f"{dt_obj.hour:02d}{dt_obj.minute:02d}/"
        )
    else:
        path = (
            f"{satellite}/AHI-L1b-FLDK/"
            f"{dt_obj.year:04d}/{dt_obj.month:02d}/{dt_obj.day:02d}/"
            f"{dt_obj.hour:02d}{dt_obj.minute:02d}/"
        )
    logging.info(f"Searching {path}")
    try:
        all_files = fs.ls(path)
    except FileNotFoundError:
        logging.warning(f"Path not found: {path}")
        return {}
    discovered = {}
    for b in bands_list:
        band_str = f"B{b:02d}"
        candidates = []
        if use_target:
            band_pattern = f"_{band_str}_"
            if target_segment:
                seg_pattern = f"_{target_segment}_"
                matches = [f for f in all_files if band_pattern in f and seg_pattern in f]
            else:
                matches = [f for f in all_files if band_pattern in f and "_R3" in f]
            if matches:
                candidates = sorted(matches)
        else:
            for seg in segments:
                seg_pattern = f"_{seg}"
                band_pattern = f"_{band_str}_"
                matches = [f for f in all_files if band_pattern in f and seg_pattern in f]
                candidates.extend(matches)
        if candidates:
            discovered[b] = sorted(candidates)
        else:
            logging.warning(f"No files found for band {b} in segments {segments}")
    return discovered

def get_latest_available_dt(satellite, segments, bands, max_attempts=18, use_target=False):
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    dt = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)

    for _ in range(max_attempts):
        if use_target:
            path = (f"{satellite}/AHI-L1b-Target/"
                    f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/"
                    f"{dt.hour:02d}{dt.minute:02d}/")
        else:
            path = (f"{satellite}/AHI-L1b-FLDK/"
                    f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/"
                    f"{dt.hour:02d}{dt.minute:02d}/")

        try:
            all_files = fs.ls(path)
        except FileNotFoundError:
            dt -= datetime.timedelta(minutes=10)
            continue

        all_present = True
        for band in bands:
            band_str = f"B{band:02d}"
            if use_target:
                if not any(band_str in f and "_R3" in f for f in all_files):
                    all_present = False
                    break
            else:
                for seg in segments:
                    seg_pattern = f"_{seg}"
                    if not any(band_str in f and seg_pattern in f for f in all_files):
                        all_present = False
                        break
            if not all_present:
                break

        if all_present:
            return dt

        dt -= datetime.timedelta(minutes=10)

    return None
    
def download_one(remote_path, local_path, band, segment, cancel_event=None, retries=3):
    thread_name = threading.current_thread().name
    for attempt in range(1, retries + 1):
        if cancel_event and cancel_event.is_set():
            logging.info(f"{thread_name}: cancelled B{band} segment {segment}")
            return False
        try:
            logging.info(f"{thread_name}: downloading B{band} segment {segment} from {os.path.basename(remote_path)} (attempt {attempt}/{retries})")
            key = remote_path
            bucket = "noaa-himawari9"
            for prefix in ("noaa-himawari9", "noaa-himawari8"):
                if remote_path.startswith(f"{prefix}/"):
                    bucket = prefix
                    key = remote_path[len(prefix) + 1:]
                    break
            url = f"https://{bucket}.s3.amazonaws.com/{key}"
            resp = _download_session.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        f.write(chunk)
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            logging.info(f"{thread_name}: downloaded B{band} segment {segment} ({file_size:.1f} MB)")
            return True
        except Exception as e:
            tb = traceback.format_exc()
            logging.warning(f"{thread_name}: attempt {attempt}/{retries} failed for {remote_path}: {e}\n{tb}")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
    if cancel_event is not None:
        cancel_event.set()
    return False

def decompress_one(bz2_path, dat_path, band, segment):
    if os.path.exists(dat_path):
        return
    try:
        thread_name = threading.current_thread().name
        with bz2.open(bz2_path, "rb") as f_in, open(dat_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        logging.info(f"{thread_name}: decompressed B{band} segment {segment} -> {os.path.basename(dat_path)}")
    except Exception as e:
        logging.error(f"Failed to decompress {bz2_path}: {e}")

def download_and_decompress_all(remote_map, local_dir, download_workers=16, decompress_workers=8):
    os.makedirs(local_dir, exist_ok=True)
    download_tasks = []
    bz2_local = {}
    dat_local = {}
    band_map = {}
    seg_map = {}

    for band, rpaths in remote_map.items():
        for rpath in rpaths:
            seg_match = re.search(r'_S(\d{2})', rpath)
            segment = seg_match.group(1) if seg_match else '??'
            base = os.path.basename(rpath)
            if rpath.endswith(".bz2"):
                dat_name = base[:-4]
                lpath_bz2 = os.path.join(local_dir, base)
                lpath_dat = os.path.join(local_dir, dat_name)
                bz2_local[rpath] = lpath_bz2
                dat_local[rpath] = lpath_dat
                band_map[rpath] = band
                seg_map[rpath] = segment
                if not os.path.exists(lpath_dat) and not os.path.exists(lpath_bz2):
                    download_tasks.append((rpath, lpath_bz2, band, segment))
            else:
                lpath_dat = os.path.join(local_dir, base)
                dat_local[rpath] = lpath_dat
                band_map[rpath] = band
                seg_map[rpath] = segment
                if not os.path.exists(lpath_dat):
                    download_tasks.append((rpath, lpath_dat, band, segment))

    if download_tasks:
        logging.info(f"Downloading {len(download_tasks)} files with {download_workers} workers...")
        cancel_event = threading.Event()
        all_ok = True
        with ThreadPoolExecutor(max_workers=download_workers, thread_name_prefix='Downloader') as ex:
            futures = {ex.submit(download_one, r, l, b, s, cancel_event): (r, l, b, s) for r, l, b, s in download_tasks}
            for f in as_completed(futures):
                if cancel_event.is_set():
                    for other in futures:
                        other.cancel()
                    all_ok = False
                    break
                if not f.result():
                    all_ok = False
                    cancel_event.set()
                    for other in futures:
                        other.cancel()
                    break
        if not all_ok:
            logging.warning("Download aborted due to failures; falling back")
            return None

    decompress_tasks = []
    for rpath, lpath_bz2 in bz2_local.items():
        lpath_dat = dat_local[rpath]
        if not os.path.exists(lpath_dat) and os.path.exists(lpath_bz2):
            decompress_tasks.append((lpath_bz2, lpath_dat, band_map[rpath], seg_map[rpath]))

    if decompress_tasks:
        logging.info(f"Decompressing {len(decompress_tasks)} files with {decompress_workers} workers...")
        with ThreadPoolExecutor(max_workers=decompress_workers, thread_name_prefix='Decompressor') as ex:
            futures = [ex.submit(decompress_one, bz, dat, b, s) for bz, dat, b, s in decompress_tasks]
            for f in as_completed(futures):
                f.result()

    local_map = {}
    for rpath, lpath_dat in dat_local.items():
        if os.path.exists(lpath_dat):
            band = band_map[rpath]
            local_map.setdefault(band, []).append(lpath_dat)
    return local_map


def prefetch_all_slots(satellites, bands, segments, time_slots, use_target,
                       download_workers, decompress_workers):
    slots = [dt for dt in time_slots if dt is not None]
    if not slots:
        return None, None
    cache_dir = tempfile.mkdtemp(prefix="automata_cache_")
    all_remote = {}
    fetched = 0
    for dt in slots:
        for sat in satellites:
            remote_map = discover_ahi_files(sat, dt, bands, segments, use_target=use_target)
            if not remote_map:
                continue
            missing = [b for b in bands if b not in remote_map or not remote_map[b]]
            if missing:
                continue
            for b, paths in remote_map.items():
                all_remote.setdefault(b, []).extend(paths)
            fetched += 1
            break
        else:
            logging.warning(f"  Prefetch: no complete data for {dt.strftime('%Y-%m-%d %H:%M')}Z")
    if not all_remote:
        shutil.rmtree(cache_dir, ignore_errors=True)
        return None, None
    logging.info(f"Prefetching {len(all_remote)} band set(s) for {fetched} slot(s) into {cache_dir}...")
    local_map = download_and_decompress_all(all_remote, cache_dir,
                                            download_workers, decompress_workers)
    if local_map is None:
        shutil.rmtree(cache_dir, ignore_errors=True)
        return None, None
    slot_map = {}
    for band, paths in local_map.items():
        for p in paths:
            m = re.search(r'_(\d{8})_(\d{4})_', os.path.basename(p))
            if not m:
                continue
            try:
                key = datetime.datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M")
            except ValueError:
                continue
            slot_map.setdefault(key, {}).setdefault(band, []).append(p)
    return cache_dir, slot_map

def apply_rgb_corrections(r, g, b, ir, target_area, target_dt, mode=1,
                          saturation_factor=1.5, gamma_cor=0.88):
    from pyorbital.astronomy import sun_zenith_angle
    r_norm = np.clip(r / 100.0 if np.nanmax(r) > 1.0 else r, 0.0, 1.0)
    g_norm = np.clip(g / 100.0 if np.nanmax(g) > 1.0 else g, 0.0, 1.0)
    b_norm = np.clip(b / 100.0 if np.nanmax(b) > 1.0 else b, 0.0, 1.0)

    lons, lats = target_area.get_lonlats()
    sza = sun_zenith_angle(target_dt, lons, lats)
    cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
    cos2_sza = np.clip(np.cos(np.radians(sza)), 0.38, 1.0)

    path_sun = 1.0 / cos2_sza
    path_sun_a = 1.0 / cos_sza

    r_bright = r_norm * path_sun_a * 0.9 + 0.01
    g_bright = g_norm * path_sun_a * 0.9 + 0.01
    b_bright = b_norm * path_sun_a * 0.9 + 0.01

    if mode == 1:
        rayleigh_r = 0.011 * path_sun + 0.001
        rayleigh_g = 0.031 * path_sun + 0.001
        rayleigh_b = 0.051 * path_sun + 0.002
    else:
        rayleigh_r = 0.011 * path_sun + 0.001
        rayleigh_g = 0.031 * path_sun + 0.004
        rayleigh_b = 0.051 * path_sun + 0.005

    r_corr = np.clip(r_bright - rayleigh_r, 0.0, 1.0)
    g_corr = np.clip(g_bright - rayleigh_g, 0.0, 1.0)
    b_corr = np.clip(b_bright - rayleigh_b, 0.0, 1.0)

    day_weight = np.clip((91.0 - sza) / 5.0, 0.0, 1.0)
    night_weight = 1.0 - day_weight

    r_final_vis = r_corr * day_weight
    g_final_vis = g_corr * day_weight
    b_final_vis = b_corr * day_weight

    if gamma_cor != 1.0:
        r_final_vis = np.clip(np.power(r_final_vis, gamma_cor), 0.0, 1.0)
        g_final_vis = np.clip(np.power(g_final_vis, gamma_cor), 0.0, 1.0)
        b_final_vis = np.clip(np.power(b_final_vis, gamma_cor), 0.0, 1.0)

    ir_norm = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
    ir_layer = np.power(ir_norm, 1.5) * 0.66

    r_final = r_final_vis + ir_layer * night_weight
    g_final = g_final_vis + ir_layer * night_weight
    b_final = b_final_vis + ir_layer * night_weight

    if saturation_factor != 1.0:
        luminance = 0.2989 * r_final + 0.5870 * g_final + 0.1140 * b_final
        r_final = np.clip(luminance + saturation_factor * (r_final - luminance), 0.0, 1.0)
        g_final = np.clip(luminance + saturation_factor * (g_final - luminance), 0.0, 1.0)
        b_final = np.clip(luminance + saturation_factor * (b_final - luminance), 0.0, 1.0)

    return r_final, g_final, b_final

def _reduced_area(area, source_pixel_m=500.0):
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

def _upscale_rgb(r, g, b, target_area):
    from PIL import Image
    h, w = r.shape
    tw, th = target_area.y_size, target_area.x_size
    if (w, h) == (tw, th):
        return r, g, b
    rgb = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)], axis=-1)
    img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
    img = img.resize((tw, th))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[..., 0], arr[..., 1], arr[..., 2]

def _resize_like(arr, target_shape):
    from PIL import Image
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

def process_vpsift_ahi_data(local_files_map, target_area, target_dt, composite_type, resample_type="nearest", sat_source="him"):
    if sat_source == "gk2a":
        return process_gk2a_data(local_files_map, target_area, target_dt, composite_type, resample_type)
    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        return _process_goes_storm_data(local_files_map, target_area, target_dt,
                                        composite_type, resample_type)
    if sat_source == "mtg":
        return _process_mtg_storm_data(local_files_map, target_area, target_dt,
                                       composite_type, resample_type)
    if sat_source in MTSAT_SAT_CONFIG:
        return _process_mtsat_storm_data(local_files_map, target_area, target_dt,
                                         composite_type, resample_type, sat_source=sat_source)
    import logging
    all_files = []
    for paths in local_files_map.values():
        all_files.extend(paths)
    if not all_files:
        raise ValueError("No local files")
    scn = Scene(filenames=all_files, reader="ahi_hsd")

    if target_area is None:
        if composite_type in ("infrared", "dvorak"):
            scn.load(["B13"])
            ir = scn["B13"].compute().astype(np.float32)
            logging.info(f"DEBUG: B13 native range: {np.nanmin(ir):.1f} - {np.nanmax(ir):.1f} K, shape: {ir.shape}, nans: {np.isnan(ir).sum()}")
            return ir, None, None, None
        elif composite_type == "sandwich":
            scn.load(["B03", "B13"])
            vis = scn["B03"].compute().astype(np.float32)
            ir = scn["B13"].compute().astype(np.float32)
            if vis.shape != ir.shape:
                vis = _resize_like(vis, ir.shape)
            return vis, ir, None, None
        elif composite_type == "b03":
            scn.load(["B03"])
            b03 = scn["B03"].compute().astype(np.float32)
            return b03, None, None, None
        elif composite_type == "b07":
            scn.load(["B07"])
            b07 = scn["B07"].compute().astype(np.float32)
            return b07, None, None, None
        elif composite_type == "irv":
            scn.load(["B03", "B13"])
            vis = scn["B03"].compute().astype(np.float32)
            ir = scn["B13"].compute().astype(np.float32)
            if vis.shape != ir.shape:
                vis = _resize_like(vis, ir.shape)
            return vis, ir, None, None
        elif composite_type in ("falsecolor", "falsecoloradv"):
            from pyorbital.astronomy import sun_zenith_angle
            scn.load(["B03", "B13"])
            vis = scn["B03"].compute().astype(np.float32)
            ir = scn["B13"].compute().astype(np.float32)
            target_shape = min([vis.shape, ir.shape], key=lambda s: s[0] * s[1])
            vis = _resize_like(vis, target_shape)
            ir = _resize_like(ir, target_shape)
            area = scn["B13"].attrs.get("area")
            if area is None:
                area = scn["B03"].attrs.get("area")
            if area is None:
                area = _native_target_area(local_files_map)
            if area is not None:
                area = _area_with_shape(area, target_shape)
                sza = sun_zenith_angle(target_dt, *area.get_lonlats())
                sza = _resize_like(np.asarray(sza, dtype=np.float32), target_shape)
            else:
                sza = np.zeros(target_shape, dtype=np.float32)
            r, g, b = _false_color_rgb(vis, ir, sza, advanced=(composite_type == "falsecoloradv"))
            return r, g, b, None
        elif composite_type == "b09":
            scn.load(["B09"])
            b09 = scn["B09"].compute().astype(np.float32)
            return b09, None, None, None
        elif composite_type == "firetemp":
            scn.load(["B07", "B06", "B09"])
            b07 = scn["B07"].compute().astype(np.float32)
            b06 = scn["B06"].compute().astype(np.float32)
            b09 = scn["B09"].compute().astype(np.float32)
            target_shape = min([b07.shape, b06.shape, b09.shape], key=lambda s: s[0] * s[1])
            b07 = _resize_like(b07, target_shape)
            b06 = _resize_like(b06, target_shape)
            b09 = _resize_like(b09, target_shape)
            r = _linear_normalize(b07, 273.0, 350.0)
            g = _linear_normalize(b06, 0.0, 50.0)
            b = _linear_normalize(b09, 0.0, 50.0)
            return r, g, b, None
        elif composite_type == "dayconv":
            scn.load(["B05", "B03", "B07", "B08", "B10", "B13"])
            b05 = scn["B05"].compute().astype(np.float32)
            b03 = scn["B03"].compute().astype(np.float32)
            b07 = scn["B07"].compute().astype(np.float32)
            b08 = scn["B08"].compute().astype(np.float32)
            b10 = scn["B10"].compute().astype(np.float32)
            b13 = scn["B13"].compute().astype(np.float32)
            target_shape = min([b05.shape, b03.shape, b07.shape, b08.shape, b10.shape, b13.shape],
                               key=lambda s: s[0] * s[1])
            b05 = _resize_like(b05, target_shape)
            b03 = _resize_like(b03, target_shape)
            b07 = _resize_like(b07, target_shape)
            b08 = _resize_like(b08, target_shape)
            b10 = _resize_like(b10, target_shape)
            b13 = _resize_like(b13, target_shape)
            r = _linear_normalize(b08 - b10, -35.0, 5.0)
            g = _linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
            b = _linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
            return r, g, b, None
        elif composite_type == "true":
            scn.load(["B01", "B03", "B04", "B13"])
            r = scn["B03"].compute().astype(np.float32)
            b = _resize_like(scn["B01"].compute().astype(np.float32), r.shape)
            v = _resize_like(scn["B04"].compute().astype(np.float32), r.shape)
            ir = _resize_like(scn["B13"].compute().astype(np.float32), r.shape)
            g = 0.45 * r + 0.10 * v + 0.45 * b
            area = scn["B03"].attrs.get("area")
            if area is None:
                area = _native_target_area(local_files_map)
            if area is not None:
                r, g, b = apply_rgb_corrections(r, g, b, ir, area, target_dt, mode=1)
                return r, g, b, None
            return r, g, b, ir
        else:
            raise ValueError(f"Unsupported composite_type for native data: {composite_type}")

    if composite_type == "true":
        scn.load(["B01", "B03", "B04", "B13"])
        work_area = _reduced_area(target_area)
        res = scn.resample(work_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        r_lazy = res["B03"].data
        b_lazy = res["B01"].data
        v_lazy = res["B04"].data
        ir_lazy = res["B13"].data
        r, b, veggie, ir = dask.compute(r_lazy, b_lazy, v_lazy, ir_lazy)
        r = r.astype(np.float32)
        b = b.astype(np.float32)
        veggie = veggie.astype(np.float32)
        ir = ir.astype(np.float32)
        g = 0.45 * r + 0.10 * veggie + 0.45 * b
        r_c, g_c, b_c = apply_rgb_corrections(r, g, b, ir, work_area, target_dt, mode=1)
        r_c, g_c, b_c = _upscale_rgb(r_c, g_c, b_c, target_area)
        return r_c, g_c, b_c, None

    elif composite_type in ("infrared", "dvorak"):
        scn.load(["B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        ir = res["B13"].data.compute().astype(np.float32)
        logging.info(f"DEBUG: B13 raw range: {np.nanmin(ir):.1f} - {np.nanmax(ir):.1f} K, shape: {ir.shape}, nans: {np.isnan(ir).sum()}")
        return ir, None, None, None

    elif composite_type == "sandwich":
        scn.load(["B03", "B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        vis_lazy = res["B03"].data
        ir_lazy = res["B13"].data
        vis, ir = dask.compute(vis_lazy, ir_lazy)
        vis = vis.astype(np.float32)
        ir = ir.astype(np.float32)
        return vis, ir, None, None

    elif composite_type == "b03":
        scn.load(["B03"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        b03 = res["B03"].data.compute().astype(np.float32)
        return b03, None, None, None

    elif composite_type == "irv":
        scn.load(["B03", "B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        vis_lazy = res["B03"].data
        ir_lazy = res["B13"].data
        vis, ir = dask.compute(vis_lazy, ir_lazy)
        vis = vis.astype(np.float32)
        ir = ir.astype(np.float32)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        vis_norm = np.clip(vis / 100.0 if np.nanmax(vis) > 1.0 else vis, 0.0, 1.0)
        cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
        cos2_sza = np.clip(np.cos(np.radians(sza)), 0.38, 1.0)
        path_sun = 1.0 / cos2_sza
        path_sun_a = 1.0 / cos_sza
        vis_bright = vis_norm * path_sun_a * 0.9 + 0.01
        rayleigh = 0.011 * path_sun + 0.001
        vis_corr = np.clip(vis_bright - rayleigh, 0.0, 1.0)
        day_weight = np.clip((91.0 - sza) / 5.0, 0.0, 1.0)
        night_weight = 1.0 - day_weight
        ir_norm = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
        ir_layer = np.power(ir_norm, 1.5) * 0.66
        result = vis_corr * day_weight + ir_layer * night_weight
        result = np.clip(result, 0.0, 1.0)
        return result, None, None, None

    elif composite_type in ("falsecolor", "falsecoloradv"):
        scn.load(["B03", "B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        vis_lazy = res["B03"].data
        ir_lazy = res["B13"].data
        vis, ir = dask.compute(vis_lazy, ir_lazy)
        vis = vis.astype(np.float32)
        ir = ir.astype(np.float32)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        r, g, b = _false_color_rgb(vis, ir, sza, advanced=(composite_type == "falsecoloradv"))
        return r, g, b, None

    elif composite_type == "b09":
        scn.load(["B09"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        b09 = res["B09"].data.compute().astype(np.float32)
        return b09, None, None, None
    elif composite_type == "b07":
        scn.load(["B07"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        b07 = res["B07"].data.compute().astype(np.float32)
        return b07, None, None, None

    elif composite_type == "firetemp":
        scn.load(["B07", "B06", "B09"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        b07, b06, b09 = dask.compute(res["B07"].data, res["B06"].data, res["B09"].data)
        b07 = b07.astype(np.float32)
        b06 = b06.astype(np.float32)
        b09 = b09.astype(np.float32)
        r = _linear_normalize(b07, 273.0, 350.0)
        g = _linear_normalize(b06, 0.0, 50.0)
        b = _linear_normalize(b09, 0.0, 50.0)
        return r, g, b, None

    elif composite_type == "dayconv":
        scn.load(["B05", "B03", "B07", "B08", "B10", "B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        b05, b03, b07, b08, b10, b13 = dask.compute(
            res["B05"].data, res["B03"].data, res["B07"].data,
            res["B08"].data, res["B10"].data, res["B13"].data)
        b05 = b05.astype(np.float32)
        b03 = b03.astype(np.float32)
        b07 = b07.astype(np.float32)
        b08 = b08.astype(np.float32)
        b10 = b10.astype(np.float32)
        b13 = b13.astype(np.float32)
        r = _linear_normalize(b08 - b10, -35.0, 5.0)
        g = _linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
        b = _linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
        return r, g, b, None

    else:
        raise ValueError(f"Unsupported MonWatch-CLI product: {composite_type}")

GK2A_BUCKET = "noaa-gk2a-pds"
GK2A_HTTPS_BASE = "https://noaa-gk2a-pds.s3.amazonaws.com"
GK2A_DEFAULT_GRID_COLOR = "#FFFF00"
GK2A_DEFAULT_COASTLINE_COLOR = "#e433ff"

GK2A_CHANNEL_BAND = {
    "vi004": (1, 0.470), "vi005": (2, 0.509), "vi006": (3, 0.639), "vi008": (4, 0.863),
    "nr016": (5, 1.610),
    "sw038": (7, 3.830), "wv063": (8, 6.210), "wv069": (9, 6.940), "wv073": (10, 7.330),
    "ir087": (11, 8.590), "ir096": (12, 9.620), "ir105": (13, 10.350), "ir112": (14, 11.230),
    "ir123": (15, 12.360), "ir133": (16, 13.290),
}
GK2A_BAND_CHANNEL = {b: c for c, (b, _) in GK2A_CHANNEL_BAND.items()}
GK2A_CHANNEL_CWL = {c: wl for c, (_, wl) in GK2A_CHANNEL_BAND.items()}
GK2A_IR_CHANNELS = set(["sw038", "wv063", "wv069", "wv073",
                        "ir087", "ir096", "ir105", "ir112", "ir123", "ir133"])


def _gk2a_channel_from_path(path):
    base = os.path.basename(path)
    for code in GK2A_CHANNEL_CWL:
        if re.search(rf"_{code}(?:\.|_)", base):
            return code
    return None


def _gk2a_build_area(cfac, lfac, coff, loff, ncols, nlines, sub_lon_deg, h, a_rad, b_rad):
    def xy(line, col):
        x = (float(col) - coff) / (cfac / 2 ** 16)
        y = (float(line) - loff) / (lfac / 2 ** 16)
        return np.deg2rad(x), np.deg2rad(y)
    x_tl, y_tl = xy(0.5, 0.5)
    x_br, y_br = xy(nlines + 0.5, ncols + 0.5)
    aex = (x_tl * h, y_br * h, x_br * h, y_tl * h)
    proj = {"proj": "geos", "h": h, "lon_0": sub_lon_deg, "a": a_rad, "b": b_rad, "sweep": "x"}
    return AreaDefinition("gk2a", "GK2A AMI", "geos", proj, ncols, nlines, aex)


def _gk2a_full_disk_area(attrs):
    h = float(attrs["nominal_satellite_height"]) - float(attrs["earth_equatorial_radius"])
    sub_lon = np.rad2deg(float(attrs["sub_longitude"]))
    return _gk2a_build_area(
        float(attrs["cfac"]), float(attrs["lfac"]),
        float(attrs["coff"]), float(attrs["loff"]),
        int(attrs["number_of_columns"]), int(attrs["number_of_lines"]),
        sub_lon, h,
        float(attrs["earth_equatorial_radius"]), float(attrs["earth_polar_radius"]))


_GK2A_PLANK = {"h": 6.62606957e-34, "c": 2.99792458e8, "k": 1.3806488e-23}


def _gk2a_radiance(attrs, dn):
    gain = float(attrs["DN_to_Radiance_Gain"])
    offset = float(attrs["DN_to_Radiance_Offset"])
    rad = dn * gain + offset
    return np.where(rad < 0, np.nan, rad)


def _gk2a_brightness_temperature(attrs, rad, wl_um):
    h = _GK2A_PLANK["h"]
    c = _GK2A_PLANK["c"]
    k = _GK2A_PLANK["k"]
    wn = (10000.0 / wl_um) * 100.0
    e1 = 2 * h * c * c * wn ** 3
    e2 = rad * 1e-5
    with np.errstate(divide="ignore", invalid="ignore"):
        t_eff = (h * c / k) * wn / np.log(e1 / e2 + 1.0)
    c0 = float(attrs["Teff_to_Tbb_c0"])
    c1 = float(attrs["Teff_to_Tbb_c1"])
    c2 = float(attrs["Teff_to_Tbb_c2"])
    bt = c0 + c1 * t_eff + c2 * t_eff * t_eff
    return np.where(np.isfinite(bt), bt, np.nan)


def _gk2a_source_bbox(full_area, attrs, target_area, radius_m=50000):
    src_p = pyproj.Proj(full_area.proj_dict)
    tgt_p = pyproj.Proj(target_area.proj_dict)
    x0, y0, x1, y1 = target_area.area_extent
    xs = np.array([x0, x1, x0, x1, (x0 + x1) / 2.0, (x0 + x1) / 2.0, x0, x1])
    ys = np.array([y0, y0, y1, y1, (y0 + y1) / 2.0, (y0 + y1) / 2.0, y0, y1])
    lons, lats = tgt_p(xs, ys, inverse=True)
    gx, gy = src_p(lons, lats)
    h = float(attrs["nominal_satellite_height"]) - float(attrs["earth_equatorial_radius"])
    cfac = float(attrs["cfac"])
    lfac = float(attrs["lfac"])
    coff = float(attrs["coff"])
    loff = float(attrs["loff"])
    ncols = int(attrs["number_of_columns"])
    nlines = int(attrs["number_of_lines"])
    cols = coff + np.rad2deg(gx / h) * (cfac / 2 ** 16)
    rows = loff + np.rad2deg(gy / h) * (lfac / 2 ** 16)
    finite = np.isfinite(cols) & np.isfinite(rows)
    if not finite.any():
        return None
    cols = cols[finite]
    rows = rows[finite]
    gsd = h * (1.0 / (cfac / 2 ** 16)) * (np.pi / 180.0)
    margin = int(np.ceil(radius_m / gsd)) + 20
    col0 = max(0, int(np.floor(cols.min())) - margin)
    col1 = min(ncols, int(np.ceil(cols.max())) + margin)
    row0 = max(0, int(np.floor(rows.min())) - margin)
    row1 = min(nlines, int(np.ceil(rows.max())) + margin)
    if col1 <= col0 or row1 <= row0:
        return None
    return row0, row1, col0, col1


def _gk2a_read_band(local_path, target_area, resample_type="nearest"):
    code = _gk2a_channel_from_path(local_path)
    if code is None:
        raise ValueError(f"Unrecognized GK2A band file: {os.path.basename(local_path)}")
    ds = xr.open_dataset(local_path)
    try:
        attrs = dict(ds.attrs)
        full_area = _gk2a_full_disk_area(attrs)
        if target_area is None:
            data = ds["image_pixel_values"].values.astype(np.float32)
            rad = _gk2a_radiance(attrs, data)
            if code in GK2A_IR_CHANNELS:
                data = _gk2a_brightness_temperature(attrs, rad, GK2A_CHANNEL_CWL[code])
            else:
                rad_to_alb = float(attrs.get("Radiance_to_Albedo_c", 0.0))
                data = rad * rad_to_alb * 100.0
            return np.where(np.isfinite(data), data, np.nan).astype(np.float32), full_area
        bbox = _gk2a_source_bbox(full_area, attrs, target_area)
        if bbox is None:
            logging.info(f"GK2A: global/wide target area; full-disk resample for {os.path.basename(local_path)}")
            data = ds["image_pixel_values"].values.astype(np.float32)
            rad = _gk2a_radiance(attrs, data)
            if code in GK2A_IR_CHANNELS:
                data = _gk2a_brightness_temperature(attrs, rad, GK2A_CHANNEL_CWL[code])
            else:
                rad_to_alb = float(attrs.get("Radiance_to_Albedo_c", 0.0))
                data = rad * rad_to_alb * 100.0
            data = np.where(np.isfinite(data), data, np.nan).astype(np.float32)
            out = kd_tree.resample_nearest(full_area, data, target_area,
                                           radius_of_influence=50000,
                                           fill_value=np.nan, reduce_data=True)
            return out.astype(np.float32), full_area
        row0, row1, col0, col1 = bbox
        dn = ds["image_pixel_values"].isel(dim_image_y=slice(row0, row1), dim_image_x=slice(col0, col1)).values.astype(np.float32)
        rad = _gk2a_radiance(attrs, dn)
        if code in GK2A_IR_CHANNELS:
            data = _gk2a_brightness_temperature(attrs, rad, GK2A_CHANNEL_CWL[code])
        else:
            rad_to_alb = float(attrs.get("Radiance_to_Albedo_c", 0.0))
            data = rad * rad_to_alb * 100.0
        data = np.where(np.isfinite(data), data, np.nan).astype(np.float32)
        sub_area = _gk2a_build_area(
            float(attrs["cfac"]), float(attrs["lfac"]),
            float(attrs["coff"]) - col0, float(attrs["loff"]) - row0,
            col1 - col0, row1 - row0,
            np.rad2deg(float(attrs["sub_longitude"])),
            float(attrs["nominal_satellite_height"]) - float(attrs["earth_equatorial_radius"]),
            float(attrs["earth_equatorial_radius"]), float(attrs["earth_polar_radius"]))
        out = kd_tree.resample_nearest(sub_area, data, target_area,
                                       radius_of_influence=50000,
                                       fill_value=np.nan, reduce_data=True)
        return out.astype(np.float32), full_area
    finally:
        ds.close()


def discover_gk2a_files(dt_obj, bands_list):
    path = (f"{GK2A_BUCKET}/AMI/L1B/FD/"
            f"{dt_obj.year:04d}{dt_obj.month:02d}/{dt_obj.day:02d}/"
            f"{dt_obj.hour:02d}/")
    stamp = f"_{dt_obj.year:04d}{dt_obj.month:02d}{dt_obj.day:02d}{dt_obj.hour:02d}{dt_obj.minute:02d}.nc"
    logging.info(f"Searching GK2A {path}")
    try:
        all_files = fs.ls(path)
    except FileNotFoundError:
        logging.warning(f"Path not found: {path}")
        return {}
    discovered = {}
    for b in bands_list:
        code = GK2A_BAND_CHANNEL.get(b)
        if code is None:
            logging.warning(f"GK2A has no channel for band B{b:02d}; skipping")
            continue
        matches = [f for f in all_files if f"_{code}_" in f and stamp in f]
        if matches:
            discovered[b] = sorted(matches)
        else:
            logging.warning(f"No GK2A files found for band B{b:02d} ({code}) in {path}")
    return discovered


def get_latest_available_dt_gk2a(bands, max_attempts=18):
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    dt = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
    for _ in range(max_attempts):
        path = (f"{GK2A_BUCKET}/AMI/L1B/FD/"
                f"{dt.year:04d}{dt.month:02d}/{dt.day:02d}/{dt.hour:02d}/")
        try:
            all_files = fs.ls(path)
        except FileNotFoundError:
            dt -= datetime.timedelta(minutes=10)
            continue
        stamp = f"_{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}.nc"
        all_present = True
        for band in bands:
            code = GK2A_BAND_CHANNEL.get(band)
            if code is None:
                continue
            if not any(f"_{code}_" in f and stamp in f for f in all_files):
                all_present = False
                break
        if all_present:
            return dt
        dt -= datetime.timedelta(minutes=10)
    return None


def download_gk2a_file(remote_path, local_path, band, retries=3):
    thread_name = threading.current_thread().name
    for attempt in range(1, retries + 1):
        try:
            key = remote_path
            bucket = GK2A_BUCKET
            if remote_path.startswith(f"{GK2A_BUCKET}/"):
                key = remote_path[len(GK2A_BUCKET) + 1:]
            url = f"https://{bucket}.s3.amazonaws.com/{key}"
            resp = _download_session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        f.write(chunk)
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            logging.info(f"{thread_name}: downloaded B{band:02d} {os.path.basename(local_path)} ({file_size:.1f} MB)")
            return True
        except Exception as e:
            logging.warning(f"{thread_name}: attempt {attempt}/{retries} failed for {os.path.basename(remote_path)}: {e}")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
    return False


def download_gk2a_files(remote_map, local_dir, download_workers=16):
    os.makedirs(local_dir, exist_ok=True)
    tasks = []
    local_paths = {}
    for band, rpaths in remote_map.items():
        for rpath in rpaths:
            lpath = os.path.join(local_dir, os.path.basename(rpath))
            local_paths[rpath] = lpath
            if not os.path.exists(lpath):
                tasks.append((rpath, lpath, band))
    if tasks:
        logging.info(f"Downloading {len(tasks)} GK2A files with {download_workers} workers...")
        all_ok = True
        with ThreadPoolExecutor(max_workers=download_workers, thread_name_prefix='Downloader') as ex:
            futures = {ex.submit(download_gk2a_file, r, l, b): (r, l, b) for r, l, b in tasks}
            for f in as_completed(futures):
                if not f.result():
                    all_ok = False
                    break
        if not all_ok:
            logging.warning("GK2A download aborted due to failures")
            return None
    local_map = {}
    for band, rpaths in remote_map.items():
        for rpath in rpaths:
            lpath = local_paths[rpath]
            if os.path.exists(lpath):
                local_map.setdefault(band, []).append(lpath)
    return local_map


def prefetch_all_slots_gk2a(bands, time_slots, download_workers=16):
    slots = [dt for dt in time_slots if dt is not None]
    if not slots:
        return None, None
    cache_dir = tempfile.mkdtemp(prefix="automata_gk2a_cache_")
    all_remote = {}
    fetched = 0
    for dt in slots:
        remote_map = discover_gk2a_files(dt, bands)
        if not remote_map:
            logging.warning(f"  Prefetch: no GK2A data for {dt.strftime('%Y-%m-%d %H:%M')}Z")
            continue
        missing = [b for b in bands if GK2A_BAND_CHANNEL.get(b) and b not in remote_map]
        if missing:
            logging.warning(f"  Prefetch: missing GK2A bands {missing} for {dt.strftime('%Y-%m-%d %H:%M')}Z")
            continue
        for b, paths in remote_map.items():
            all_remote.setdefault(b, []).extend(paths)
        fetched += 1
    if not all_remote:
        shutil.rmtree(cache_dir, ignore_errors=True)
        return None, None
    logging.info(f"Prefetching {len(all_remote)} GK2A band set(s) for {fetched} slot(s) into {cache_dir}...")
    local_map = download_gk2a_files(all_remote, cache_dir, download_workers)
    if local_map is None:
        shutil.rmtree(cache_dir, ignore_errors=True)
        return None, None
    slot_map = {}
    for band, paths in local_map.items():
        for p in paths:
            m = re.search(r'_(\d{8})(\d{4})\.nc$', os.path.basename(p))
            if not m:
                continue
            try:
                key = datetime.datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M")
            except ValueError:
                continue
            slot_map.setdefault(key, {}).setdefault(band, []).append(p)
    return cache_dir, slot_map


def process_gk2a_data(local_files_map, target_area, target_dt, composite_type, resample_type="nearest"):
    def _band(b):
        code = GK2A_BAND_CHANNEL.get(b)
        if code is None:
            raise ValueError(f"GK-2A has no channel for band B{b:02d}")
        paths = local_files_map.get(b)
        if not paths:
            raise ValueError(f"GK2A missing band B{b:02d} ({code}) for composite {composite_type}")
        data, _ = _gk2a_read_band(paths[0], target_area, resample_type)
        return data

    if composite_type == "firetemp":
        r = _linear_normalize(_band(7), 273.0, 350.0)
        g = _linear_normalize(_band(5), 0.0, 50.0)
        b = _linear_normalize(_band(9), 0.0, 50.0)
        return r, g, b, None

    if composite_type == "fire":
        b07 = _band(7)
        return b07, None, None, None

    if composite_type in ("infrared", "dvorak"):
        return _band(13), None, None, None

    if composite_type in ("sandwich", "irv"):
        return _band(3), _band(13), None, None

    if composite_type == "b03":
        return _band(3), None, None, None

    if composite_type in ("falsecolor", "falsecoloradv"):
        vis = _band(3)
        ir = _band(13)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        r, g, b = _false_color_rgb(vis, ir, sza, advanced=(composite_type == "falsecoloradv"))
        return r, g, b, None

    if composite_type == "true":
        r = _band(3)
        b = _band(1)
        v = _band(4)
        ir = _band(13)
        g = 0.45 * r + 0.10 * v + 0.45 * b
        r, g, b = apply_rgb_corrections(r, g, b, ir, target_area, target_dt, mode=1)
        return r, g, b, None

    if composite_type == "dayconv":
        b03 = _band(3)
        b05 = _band(5)
        b07 = _band(7)
        b08 = _band(8)
        b10 = _band(10)
        b13 = _band(13)
        r = _linear_normalize(b08 - b10, -35.0, 5.0)
        g = _linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
        b = _linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
        return r, g, b, None

    raise ValueError(f"Unsupported GK2A composite: {composite_type}")

GOES_SATELLITE_MAP = {
    "goes16": "noaa-goes16",
    "goes17": "noaa-goes17",
    "goes18": "noaa-goes18",
    "goes19": "noaa-goes19",
}
GOES_DEFAULT = "goes18"

def _resolve_goes_source(sat_source, lon=None):
    if sat_source not in ("goes", "goes16", "goes17", "goes18", "goes19"):
        return sat_source
    if sat_source == "goes":
        lon_n = None
        if lon is not None:
            lon_n = ((float(lon) + 180) % 360) - 180
        if lon_n is not None and lon_n <= -106.0:
            return "goes18"
        return "goes19"
    return sat_source


def _goes_candidate_buckets(sat_source):
    if sat_source == "goes16":
        return ("noaa-goes16",)
    if sat_source == "goes17":
        return ("noaa-goes17",)
    if sat_source == "goes18":
        return ("noaa-goes18", "noaa-goes17")
    if sat_source == "goes19":
        return ("noaa-goes19", "noaa-goes16")
    if sat_source == "goes":
        return ("noaa-goes19", "noaa-goes16", "noaa-goes18", "noaa-goes17")
    return (GOES_SATELLITE_MAP.get(sat_source, sat_source),)




def _goes_extract_time(filename):
    m = re.search(r's(\d{13})', filename)
    if m:
        try:
            return datetime.datetime.strptime(m.group(1)[:11], "%Y%j%H%M")
        except ValueError:
            return datetime.datetime.min
    return datetime.datetime.min


def discover_goes_files(satellite, dt_obj, bands_list):
    bucket = GOES_SATELLITE_MAP.get(satellite, satellite)
    path = f"{bucket}/ABI-L1b-RadF/{dt_obj.year}/{dt_obj.strftime('%j')}/{dt_obj.hour:02d}/"
    logging.info(f"Searching GOES {path}")
    try:
        all_files = fs.ls(path)
    except FileNotFoundError:
        logging.warning(f"Path not found: {path}")
        return {}
    target_naive = dt_obj.replace(tzinfo=None)
    discovered = {}
    for b in bands_list:
        band_str = f"C{b:02d}"
        candidates = [f for f in all_files
                      if f"M6{band_str}" in f]
        if not candidates:
            logging.warning(f"No GOES files found for band {band_str} in {path}")
            continue
        closest = min(candidates, key=lambda f: abs((_goes_extract_time(f) - target_naive).total_seconds()))
        discovered[b] = [closest]
    return discovered


def get_latest_available_dt_goes(satellite, max_hours=4):
    bucket = GOES_SATELLITE_MAP.get(satellite, satellite)
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    best = None
    for hour_offset in range(0, max_hours + 1):
        test_dt = now - datetime.timedelta(hours=hour_offset)
        path = f"{bucket}/ABI-L1b-RadF/{test_dt.year}/{test_dt.strftime('%j')}/{test_dt.hour:02d}/"
        try:
            files = fs.ls(path)
        except FileNotFoundError:
            continue
        candidates = [f for f in files if "M6C13" in f]
        if not candidates:
            continue
        for f in sorted(candidates, reverse=True):
            t = _goes_extract_time(f)
            if t == datetime.datetime.min:
                continue
            rounded = t.replace(minute=(t.minute // 10) * 10, second=0, microsecond=0)
            if best is None or rounded > best:
                best = rounded
        if best is not None:
            return best
    return None


def download_goes_file(remote_path, local_path, band, retries=3):
    thread_name = threading.current_thread().name
    for attempt in range(1, retries + 1):
        try:
            bucket = None
            key = remote_path
            for b in GOES_SATELLITE_MAP.values():
                if remote_path.startswith(f"{b}/"):
                    bucket = b
                    key = remote_path[len(b) + 1:]
                    break
            if bucket is None:
                return False
            url = f"https://{bucket}.s3.amazonaws.com/{key}"
            resp = _download_session.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        f.write(chunk)
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            logging.info(f"{thread_name}: downloaded B{band:02d} {os.path.basename(local_path)} ({file_size:.1f} MB)")
            return True
        except Exception as e:
            logging.warning(f"{thread_name}: attempt {attempt}/{retries} failed for {os.path.basename(remote_path)}: {e}")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
    return False


def download_goes_files(remote_map, local_dir, download_workers=16):
    os.makedirs(local_dir, exist_ok=True)
    tasks = []
    local_paths = {}
    for band, rpaths in remote_map.items():
        for rpath in rpaths:
            lpath = os.path.join(local_dir, os.path.basename(rpath))
            local_paths[rpath] = lpath
            if not os.path.exists(lpath):
                tasks.append((rpath, lpath, band))
    if tasks:
        logging.info(f"Downloading {len(tasks)} GOES files with {download_workers} workers...")
        all_ok = True
        with ThreadPoolExecutor(max_workers=download_workers, thread_name_prefix='Downloader') as ex:
            futures = {ex.submit(download_goes_file, r, l, b): (r, l, b) for r, l, b in tasks}
            for f in as_completed(futures):
                if not f.result():
                    all_ok = False
                    break
        if not all_ok:
            return None
    return {b: [local_paths[p] for p in ps] for b, ps in remote_map.items()}


def _goes_full_disk_area(ds):
    g = ds["goes_imager_projection"].attrs
    h = float(g["perspective_point_height"])
    x = ds["x"].values
    y = ds["y"].values
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)
    aex = ((x[0] - dx / 2) * h, (y[-1] - dy / 2) * h,
           (x[-1] + dx / 2) * h, (y[0] + dy / 2) * h)
    proj = {
        "proj": "geos", "h": h,
        "lon_0": float(g["longitude_of_projection_origin"]),
        "a": float(g["semi_major_axis"]), "b": float(g["semi_minor_axis"]),
        "sweep": str(g["sweep_angle_axis"]),
    }
    return AreaDefinition("goes", "GOES ABI", "geos", proj, len(x), len(y), aex)

_GOES_PLANK = {"h": 6.62606957e-34, "c": 2.99792458e8, "k": 1.3806488e-23}


def _goes_calibrate_l1b(ds, rad):
    rad = np.asarray(rad, dtype=np.float32)
    try:
        band_id = int(ds["band_id"].values.item())
    except Exception:
        try:
            wl = float(ds["band_wavelength"].values.item())
            if wl < 4.0:
                band_id = int(round(wl * 10))
            else:
                band_id = 13
        except Exception:
            band_id = 13

    if band_id >= 7:
        try:
            planck_fk1 = float(ds["planck_fk1"].values.item())
            planck_fk2 = float(ds["planck_fk2"].values.item())
            planck_bc1 = float(ds["planck_bc1"].values.item())
            planck_bc2 = float(ds["planck_bc2"].values.item())
        except Exception:
            try:
                wl_um = float(ds["band_wavelength"].values.item())
            except Exception:
                wl_um = 10.35
            planck_fk1 = 2 * _GOES_PLANK["h"] * _GOES_PLANK["c"]**2 * (1e4 / wl_um)**3 * 1e-5
            planck_fk2 = _GOES_PLANK["h"] * _GOES_PLANK["c"] * (1e4 / wl_um) * 100 / _GOES_PLANK["k"]
            planck_bc1 = 0.0
            planck_bc2 = 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            bt = planck_fk2 / np.log(planck_fk1 / rad + 1.0)
        bt = planck_bc1 + planck_bc2 * bt
        return np.where(np.isfinite(bt) & (rad > 0), bt, np.nan).astype(np.float32)
    else:
        try:
            kappa0 = float(ds["kappa0"].values.item())
        except Exception:
            try:
                wl_um = float(ds["band_wavelength"].values.item())
            except Exception:
                wl_um = 0.64
            kappa0 = 1500.0 if wl_um < 0.7 else 800.0

        refl = np.pi * rad * 100.0 / kappa0
        return np.where(np.isfinite(refl), refl, np.nan).astype(np.float32)


def _goes_read_band(local_path, target_area=None):
    ds = xr.open_dataset(local_path)
    try:
        if "CMI" in ds.data_vars:
            data = ds["CMI"].values.squeeze().astype(np.float32)
        elif "Rad" in ds.data_vars:
            rad = ds["Rad"].values.squeeze().astype(np.float32)
            data = _goes_calibrate_l1b(ds, rad)
        else:
            raise ValueError(f"Unknown GOES format: no CMI or Rad variable in {local_path}")
        area = _goes_full_disk_area(ds)
        if target_area is None:
            return np.where(np.isfinite(data), data, np.nan).astype(np.float32), area
        out = kd_tree.resample_nearest(area, data, target_area,
                                       radius_of_influence=60000,
                                       fill_value=np.nan, reduce_data=True)
        return out.astype(np.float32), area
    finally:
        ds.close()


def _global_read_goes(local_map, target_area, want_vis=False):
    ir, _ = _goes_read_band(local_map[13][0], target_area)
    vis = None
    if want_vis and 2 in local_map and local_map[2]:
        vis, _ = _goes_read_band(local_map[2][0], target_area)
    return ir, vis

MTG_COLLECTION = "EO:EUM:DAT:0662"


def _load_eumetsat_creds():
    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if key and secret:
        return key, secret
    base_dir = os.path.dirname(os.path.abspath(__file__))
    creds_file = os.path.join(base_dir, "creds.txt")
    if os.path.exists(creds_file):
        try:
            pairs = {}
            for line in open(creds_file, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                pairs[k.strip()] = v.strip()
            k = pairs.get("EUMETSAT_CONSUMER_KEY")
            s = pairs.get("EUMETSAT_CONSUMER_SECRET")
            if k and s:
                return k, s
        except Exception:
            pass
    vpsift = os.path.join(base_dir, "!!VPSIFT.py")
    if os.path.exists(vpsift):
        try:
            src = open(vpsift, encoding="utf-8", errors="replace").read()
            m = re.search(r'EUMETSAT_CONSUMER_KEY\s*=\s*["\']([^"\']+)["\']', src)
            n = re.search(r'EUMETSAT_CONSUMER_SECRET\s*=\s*["\']([^"\']+)["\']', src)
            if m and n:
                k, s = m.group(1), n.group(1)
                if "get your own" not in k and "get your own" not in s:
                    key, secret = k, s
        except Exception:
            pass
    return key, secret


def _ensure_mtg_setup():
    try:
        import eumdac
    except ImportError:
        raise RuntimeError(
            "MTG (EUMETSAT) support requires the 'eumdac' package. "
            "Install it with: pip install eumdac hdf5plugin  "
            "and set EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET as environment "
            "variables or in a creds.txt file in the imager folder."
        )
    key, secret = _load_eumetsat_creds()
    if not key or not secret:
        raise RuntimeError(
            "MTG (EUMETSAT) requires credentials. "
            "Set EUMETSAT_CONSUMER_KEY and EUMETSAT_CONSUMER_SECRET as environment "
            "variables, or put them in creds.txt in the imager folder "
            "(one 'EUMETSAT_CONSUMER_KEY=...' / 'EUMETSAT_CONSUMER_SECRET=...' per line)."
        )
    return eumdac


def _download_product_with_retry(product, download_path, attempts=3):
    last_err = None
    for attempt in range(1, attempts + 1):
        if os.path.exists(download_path):
            try:
                os.remove(download_path)
            except OSError:
                pass
        try:
            logging.info(f"Fetching MTG product package ({attempt}/{attempts}): {product}")
            with product.open() as f_src, open(download_path, "wb") as f_dst:
                total = 0
                block = 256 * 1024
                while True:
                    buf = f_src.read(block)
                    if not buf:
                        break
                    f_dst.write(buf)
                    total += len(buf)
                    if total // (128 * 1024 * 1024) > (total - len(buf)) // (128 * 1024 * 1024):
                        logging.info(f"  MTG download: {total / 1e6:.0f} MB ...")
            logging.info(f"MTG product downloaded: {total / 1e6:.0f} MB")
            return download_path
        except Exception as e:
            last_err = e
            logging.warning(f"MTG download attempt {attempt}/{attempts} failed: {e}")
    raise last_err


FCI_CHANNEL_WAVELENGTH_UM = {
    "vis_04": 0.443, "vis_05": 0.510, "vis_06": 0.640, "vis_08": 0.865,
    "nir_13": 1.375, "nir_16": 1.610, "nir_22": 2.250,
    "ir_38": 3.80, "wv_63": 6.25, "wv_73": 7.35, "ir_87": 8.70,
    "ir_97": 9.60, "ir_105": 10.50, "ir_123": 12.30, "ir_133": 13.30,
}
FCI_IR_CHANNELS = {"ir_38", "wv_63", "wv_73", "ir_87", "ir_97",
                   "ir_105", "ir_123", "ir_133"}
FCI_CHANNEL_SSD = {
    "vis_04": "1km", "vis_05": "1km", "vis_06": "1km", "vis_08": "1km",
    "nir_13": "1km", "nir_16": "1km", "nir_22": "1km",
    "ir_38": "2km", "wv_63": "2km", "wv_73": "2km", "ir_87": "2km",
    "ir_97": "2km", "ir_105": "2km", "ir_123": "2km", "ir_133": "2km",
}

_FCI_PLANK = {"h": 6.62606957e-34, "c": 2.99792458e8, "k": 1.3806488e-23}


def _mtg_radiance_to_bt(rad, wl_um):
    rad = np.asarray(rad, dtype=np.float32)
    h, c, k = _FCI_PLANK["h"], _FCI_PLANK["c"], _FCI_PLANK["k"]
    wn = (10000.0 / wl_um) * 100.0
    e1 = 2 * h * c * c * wn ** 3
    e2 = rad * 1.0e-5
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = (h * c / k) * wn / np.log(e1 / e2 + 1.0)
    return np.where(np.isfinite(bt) & (rad > 0), bt, np.nan).astype(np.float32)


def _mtg_area_from_tailored(ds):
    x = np.asarray(ds["x"].values, dtype=np.float64)
    y = np.asarray(ds["y"].values, dtype=np.float64)
    ncols, nlines = x.size, y.size
    gm = {}
    if "geostationary" in ds:
        attrs = ds["geostationary"].attrs
        try:
            gm = {"lon_0": float(attrs["longitude_of_projection_origin"]),
                  "h": float(attrs["perspective_point_height"]),
                  "a": float(attrs["semi_major_axis"]),
                  "rf": float(attrs["inverse_flattening"]),
                  "sweep": str(attrs["sweep_angle_axis"])}
        except (KeyError, ValueError, TypeError):
            gm = {}
    if not gm:
        gm = {"lon_0": 0.0, "h": 35786400.0, "a": 6378137.0,
              "b": 6356752.3142, "sweep": "y"}
    proj = {"proj": "geos", "lon_0": gm["lon_0"], "h": gm["h"],
            "a": gm["a"], "sweep": gm["sweep"]}
    if "rf" in gm:
        proj["rf"] = gm["rf"]
    elif "b" in gm:
        proj["b"] = gm["b"]
    return AreaDefinition("mtg_tailor", "MTG-I1 FCI subset", "geos", proj,
                          ncols, nlines, (x[0], y[0], x[-1], y[-1]))


def _read_mtg_tailored_channel(local_paths, channel, target_area, resample_type="nearest"):
    var = f"{channel}_effective_radiance"
    ds = None
    for p in local_paths:
        try:
            d = xr.open_dataset(p)
            if var in d:
                ds = d
                break
            d.close()
        except Exception:
            continue
    if ds is None:
        tailored_like = False
        for p in local_paths:
            try:
                d = xr.open_dataset(p)
                has_gm = "geostationary" in d
                has_any_rad = any(str(v).endswith("_effective_radiance")
                                  for v in d.data_vars)
                d.close()
                if has_gm or has_any_rad:
                    tailored_like = True
                    break
            except Exception:
                continue
        if tailored_like:
            raise ValueError(f"MTG tailored files have no channel {channel} "
                             "(was it requested in the Data Tailor job?)")
        scn = Scene(filenames=local_paths, reader="fci_l1c_nc")
        scn.load([channel])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=60000)
        return res[channel].data.compute().astype(np.float32)
    try:
        area = _mtg_area_from_tailored(ds)
        arr = np.asarray(ds[var].values, dtype=np.float32)
        if float(np.asarray(ds["y"].values)[0]) < float(np.asarray(ds["y"].values)[-1]):
            arr = np.flipud(arr)
        if channel in FCI_IR_CHANNELS:
            data = _mtg_radiance_to_bt(arr, FCI_CHANNEL_WAVELENGTH_UM[channel])
        else:
            data = np.where(np.isfinite(arr), arr, np.nan)
        data = np.where(np.isfinite(data), data, np.nan).astype(np.float32)
        if target_area is None:
            return data
        out = kd_tree.resample_nearest(area, data, target_area,
                                       radius_of_influence=60000,
                                       fill_value=np.nan, reduce_data=True)
        return out.astype(np.float32)
    finally:
        ds.close()


def _mtg_tailor_download(datastore, datatailor, product, channels, roi_nswe, temp_dir):
    import time
    from eumdac.tailor_models import Chain, Filter, RegionOfInterest

    try:
        for c in list(datatailor.customisations):
            st = c.status
            if st in ("DONE", "FAILED", "KILLED"):
                logging.info(f"MTG tailor cleanup: deleting {c._id} ({st})")
                c.delete()
            elif st in ("QUEUED", "RUNNING"):
                try:
                    c.kill()
                    c.delete()
                except Exception:
                    pass
    except Exception as cleanup_err:
        logging.warning(f"MTG tailor cleanup skipped ({cleanup_err})")

    groups = {}
    for ch in channels:
        groups.setdefault(FCI_CHANNEL_SSD.get(ch, "2km"), []).append(ch)
    out_files = []
    for ssd, chans in sorted(groups.items()):
        bands = [f"{c}_effective_radiance" for c in sorted(chans)]
        chain = Chain(
            format="netcdf4",
            product="FCIL1FDHSI",
            filter=Filter(name="custom", product="FCIL1FDHSI", bands=bands),
            roi=RegionOfInterest(NSWE=roi_nswe, name="custom") if roi_nswe else None,
        )
        logging.info(f"MTG tailor job ({ssd}): {bands}")
        custs = None
        for attempt in range(1, 4):
            try:
                custs = datatailor.new_customisations([product], chain=chain)
                break
            except Exception as submit_err:
                logging.warning(f"MTG tailor submit attempt {attempt}/3 failed: {submit_err}")
                time.sleep(5)
        if custs is None:
            raise RuntimeError("MTG Data Tailor submit failed after 3 attempts")
        for c in custs:
            import socket
            _old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(30)
            try:
                last = None
                last_beat = 0.0
                deadline = time.time() + 1500
                while True:
                    try:
                        st = c.status
                        pr = getattr(c, "progress", None)
                    except Exception:
                        st = None
                        pr = None
                    if st != last:
                        logging.info(f"  tailor {c._id}: {st} ({pr})")
                        last = st
                    elif time.time() - last_beat >= 30:
                        logging.info(f"  tailor {c._id}: still {st} ({pr}) after "
                                     f"{int(time.time() - deadline + 1500)}s")
                        last_beat = time.time()
                    if st in ("DONE", "FAILED", "KILLED", "INACTIVE"):
                        break
                    if st in ("QUEUED", "RUNNING", None) and time.time() > deadline:
                        logging.error(f"MTG tailor job {c._id} did not finish within the "
                                      f"poll deadline (last status {st}); killing it")
                        try:
                            c.kill()
                        except Exception:
                            pass
                        st = "TIMEOUT"
                        break
                    time.sleep(15)
            finally:
                socket.setdefaulttimeout(_old_timeout)
            if st != "DONE":
                try:
                    log = c.logfile[-800:]
                except Exception:
                    log = ""
                raise RuntimeError(f"MTG tailor job {c._id} ended {st}: {log}")
            for oid in c.outputs:
                out_path = os.path.join(temp_dir, os.path.basename(oid))
                with open(out_path, "wb") as f:
                    with c.stream_output_iter_content(oid) as chunks:
                        for chunk in chunks:
                            f.write(chunk)
                logging.info(f"MTG tailor output: {os.path.basename(oid)} "
                             f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
                out_files.append(out_path)
            try:
                c.delete()
            except Exception:
                pass
    return out_files


def discover_mtg_files(satellite, dt_obj, bands_list, temp_dir="temp_data", roi_nswe=None):
    eumdac = _ensure_mtg_setup()
    key, secret = _load_eumetsat_creds()
    os.makedirs(temp_dir, exist_ok=True)
    token = eumdac.AccessToken((key, secret))
    datastore = eumdac.DataStore(token)
    datatailor = eumdac.DataTailor(token)
    collection = datastore.get_collection(satellite)
    start_time = dt_obj - datetime.timedelta(minutes=5)
    end_time = dt_obj + datetime.timedelta(minutes=5)
    products = collection.search(dtstart=start_time, dtend=end_time)
    product = products.first()
    if not product:
        logging.warning(f"No MTG FCI products found near {dt_obj.strftime('%Y-%m-%d %H:%M')}Z")
        return {}
    channels = sorted({AHI_TO_FCI[b] for b in bands_list if AHI_TO_FCI.get(b)})
    if not channels:
        logging.warning(f"No MTG FCI channels requested for bands {bands_list}")
        return {}
    try:
        out_files = _mtg_tailor_download(datastore, datatailor, product,
                                         channels, roi_nswe, temp_dir)
    except Exception as tailor_err:
        logging.warning(f"Data Tailor failed ({tailor_err}); falling back to full package download")
        download_path = os.path.join(temp_dir, f"{product}.zip")
        _download_product_with_retry(product, download_path, attempts=3)
        extracted_files = []
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
            extracted_files = [os.path.join(temp_dir, f)
                               for f in zip_ref.namelist() if f.endswith(".nc")]
        try:
            os.remove(download_path)
        except OSError:
            pass
        if not extracted_files:
            return {}
        return {b: extracted_files for b in bands_list}
    if not out_files:
        return {}
    return {b: out_files for b in bands_list}


def _global_read_mtg(local_map, target_area, want_vis=False):
    all_files = []
    for paths in local_map.values():
        all_files.extend(paths)
    ir = _read_mtg_tailored_channel(all_files, "ir_105", target_area, "nearest")
    vis = None
    if want_vis:
        try:
            vis = _read_mtg_tailored_channel(all_files, "vis_06", target_area, "nearest")
        except ValueError:
            vis = None
    return ir, vis

AHI_TO_ABI = {
    1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
}
AHI_TO_FCI = {
    1: "vis_04", 2: "vis_05", 3: "vis_06", 4: "vis_08", 5: "nir_16",
    6: "nir_22", 7: "ir_38", 8: None, 9: None, 10: None,
    11: "ir_87", 12: "ir_97", 13: "ir_105", 14: "ir_123", 15: "ir_123", 16: "ir_133",
}

MTSAT_FTP_HOSTS = ["mtsat.cr.chiba-u.ac.jp", "gms.cr.chiba-u.ac.jp"]
MTSAT_FTP_ROOT = "/pub"
MTSAT_SAT_CONFIG = {
    "mtsat":   {"code": "MTSAT2", "dir": "MTSAT-2", "reader": "mtsat2-imager_hrit",
                "name": "MTSAT-2", "lon": 145.0},
    "mtsat2":  {"code": "MTSAT2", "dir": "MTSAT-2", "reader": "mtsat2-imager_hrit",
                "name": "MTSAT-2", "lon": 145.0},
    "mtsat1":  {"code": "MTSAT1", "dir": "MTSAT-1R", "reader": "jami_hrit",
                "name": "MTSAT-1R", "lon": 140.0},
}
AHI_TO_MTSAT = {
    1: "VIS", 2: "VIS", 3: "VIS", 4: "VIS",
    7: "IR4", 9: "IR3", 13: "IR1", 14: "IR2",
}


def _mtsat_cfg(sat_source):
    return MTSAT_SAT_CONFIG.get(sat_source) or MTSAT_SAT_CONFIG["mtsat"]


def _mtsat_ftp(host):
    ftp = ftplib.FTP(host, timeout=60)
    try:
        ftp.login()
        ftp.voidcmd("TYPE I")
    except Exception:
        try:
            ftp.close()
        except Exception:
            pass
        raise
    return ftp


def _mtsat_remote_tar(cfg, dt_obj):
    return (f"{MTSAT_FTP_ROOT}/{cfg['dir']}/HRIT/"
            f"{dt_obj.year:04d}{dt_obj.month:02d}/{dt_obj.day:02d}/"
            f"HRIT_{cfg['code']}_{dt_obj:%Y%m%d%H%M}.tar")


def mtsat_tar_exists(cfg, dt_obj):
    remote = _mtsat_remote_tar(cfg, dt_obj)
    for host in MTSAT_FTP_HOSTS:
        try:
            ftp = _mtsat_ftp(host)
            try:
                try:
                    ftp.size(remote)
                    return True
                except ftplib.error_perm:
                    pass
            finally:
                try:
                    ftp.quit()
                except Exception:
                    pass
        except Exception as e:
            logging.warning(f"MTSAT FTP {host} unreachable: {e}")
    return False


def discover_mtsat_files(sat_source, dt_obj, bands_list):
    cfg = _mtsat_cfg(sat_source)
    probe = dt_obj.replace(minute=0, second=0, microsecond=0)
    for _ in range(48):
        if mtsat_tar_exists(cfg, probe):
            tar = _mtsat_remote_tar(cfg, probe)
            logging.info(f"MTSAT ({cfg['code']}): archive slot {probe:%Y-%m-%d %H:%M}Z -> {os.path.basename(tar)}")
            return {b: [tar] for b in bands_list}
        probe -= datetime.timedelta(hours=1)
    logging.warning(f"MTSAT archive tar not found for {dt_obj.strftime('%Y-%m-%d %H:%M')}Z")
    return {}


def download_mtsat_tar(tar_remote, local_path):
    for host in MTSAT_FTP_HOSTS:
        for attempt in range(3):
            try:
                logging.info(f"  Downloading {os.path.basename(tar_remote)} from {host} (attempt {attempt + 1}/3)")
                ftp = _mtsat_ftp(host)
                try:
                    with open(local_path, "wb") as fh:
                        ftp.retrbinary(f"RETR {tar_remote}", fh.write, blocksize=1024 * 1024)
                    logging.info(f"  Downloaded {os.path.basename(tar_remote)} ({os.path.getsize(local_path) / (1024 * 1024):.1f} MB)")
                    return True
                finally:
                    try:
                        ftp.quit()
                    except Exception:
                        pass
            except Exception as e:
                logging.warning(f"MTSAT download attempt {attempt + 1} failed: {e}")
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
    return False


def download_mtsat_tars(remote_map, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    tars = sorted({p for paths in remote_map.values() for p in paths})
    if not tars:
        return None
    local = {}
    for tar_remote in tars:
        tar_local = os.path.join(tmpdir, os.path.basename(tar_remote))
        if not os.path.exists(tar_local):
            if not download_mtsat_tar(tar_remote, tar_local):
                logging.warning(f"MTSAT download failed for {tar_remote}")
                return None
        try:
            with tarfile.open(tar_local, "r:*") as tf:
                members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".gz")]
                for m in members:
                    m.name = os.path.basename(m.name)
                tf.extractall(tmpdir, members=members, filter="data")
            for gz_path in glob.glob(os.path.join(tmpdir, "*.gz")):
                plain = gz_path[:-3]
                try:
                    with gzip.open(gz_path, "rb") as f_in, open(plain, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
                    os.remove(gz_path)
                except Exception as e:
                    logging.warning(f"MTSAT gzip decompress failed for {os.path.basename(gz_path)}: {e}")
        except Exception as e:
            logging.error(f"MTSAT extract failed for {tar_local}: {e}")
            return None
    hrit_files = [os.path.join(tmpdir, n) for n in os.listdir(tmpdir)
                  if n.startswith("HRIT_") and not n.endswith(".gz")]
    for b, _paths in remote_map.items():
        code = AHI_TO_MTSAT.get(b)
        if not code:
            logging.warning(f"AHI band {b} has no MTSAT channel; skipping")
            continue
        tag = f"DK01{code}"
        matches = [f for f in hrit_files if tag in os.path.basename(f)]
        if matches:
            local[b] = sorted(matches)
    return local or None

IBTRACS_CSV_URL = ("https://www.ncei.noaa.gov/data/international-best-track-"
                   "archive-for-climate-stewardship-ibtracs/v04r01/access/csv/"
                   "ibtracs.{basin}.list.v04r01.csv")


def _ibtracs_cache_path(basin):
    cache_dir = os.path.join(tempfile.gettempdir(), "ibtracs_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        pass
    return os.path.join(cache_dir, f"ibtracs.{basin}.list.v04r01.csv")


def _ibtracs_open_csv_file(basin):
    cache_path = _ibtracs_cache_path(basin)
    if not os.path.exists(cache_path):
        url = IBTRACS_CSV_URL.format(basin=basin)
        logging.info(f"IBTrACS: downloading {basin} archive CSV ({url}) to cache...")
        tmp_path = cache_path + ".tmp"
        try:
            resp = _download_session.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        fh.write(chunk)
            os.replace(tmp_path, cache_path)
            logging.info(f"IBTrACS: cached {cache_path} ({os.path.getsize(cache_path) / (1024 * 1024):.0f} MB)")
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
    return open(cache_path, "r", encoding="utf-8", errors="replace")


def _ibtracs_float(*candidates):
    for c in candidates:
        if c:
            try:
                return float(str(c).strip())
            except ValueError:
                continue
    return None


def _ibtracs_int(*candidates):
    for c in candidates:
        if c:
            try:
                return int(float(str(c).strip()))
            except ValueError:
                continue
    return None


def fetch_ibtracs_track(storm_name, year, basin="WP"):
    import csv as _csv
    logging.info(f"IBTrACS: locating {storm_name} ({year}) in {basin} basin archive")
    try:
        fh = _ibtracs_open_csv_file(basin)
    except Exception as e:
        logging.warning(f"IBTrACS CSV download failed: {e}")
        return None
    rows = []
    with fh:
        reader = _csv.DictReader(fh)
        for rec in reader:
            if (rec.get("SEASON") or "").strip() != str(year):
                continue
            name = (rec.get("NAME") or "").strip()
            if name.upper() != storm_name.upper():
                continue
            iso = (rec.get("ISO_TIME") or "").strip()
            if not iso:
                continue
            dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    dt = datetime.datetime.strptime(iso, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                continue
            lat = _ibtracs_float(rec.get("LAT"), rec.get("USA_LAT"))
            lon = _ibtracs_float(rec.get("LON"), rec.get("USA_LON"))
            if lat is None or lon is None:
                continue
            rows.append({
                "dt": dt,
                "lat": lat,
                "lon": lon,
                "wind": _ibtracs_int(rec.get("USA_WIND")),
                "pres": _ibtracs_int(rec.get("USA_PRES")),
                "atcf_id": (rec.get("USA_ATCF_ID") or "").strip(),
            })
    if not rows:
        logging.warning(f"IBTrACS: no track rows found for {storm_name} ({year}) "
                        f"in {basin} basin")
        return None
    rows.sort(key=lambda p: p["dt"])
    logging.info(f"IBTrACS: {storm_name} ({year}) -> {len(rows)} fixes, "
                 f"{rows[0]['dt']:%Y-%m-%d %H:%M}Z .. {rows[-1]['dt']:%Y-%m-%d %H:%M}Z")
    return rows


def interpolate_track_position(track, dt):
    pts = [(p["dt"], p["lat"], p["lon"]) for p in track]
    pts.sort(key=lambda x: x[0])
    if not pts:
        return None, None
    if dt <= pts[0][0]:
        return pts[0][1], pts[0][2]
    if dt >= pts[-1][0]:
        return pts[-1][1], pts[-1][2]
    for i in range(len(pts) - 1):
        t0, la0, lo0 = pts[i]
        t1, la1, lo1 = pts[i + 1]
        if t0 <= dt < t1:
            span = (t1 - t0).total_seconds()
            if span <= 0:
                return la0, lo0
            f = (dt - t0).total_seconds() / span
            if lo1 - lo0 > 180:
                lo1 -= 360
            elif lo1 - lo0 < -180:
                lo1 += 360
            lon = lo0 + (lo1 - lo0) * f
            return la0 + (la1 - la0) * f, lon
    return None, None


def track_peak_fix(track):
    valid = [p for p in track if p.get("wind") is not None]
    if not valid:
        return track[0] if track else None
    return max(valid, key=lambda p: p["wind"])


def build_ibtracs_storm(track, target, date_str=None, time_str=None, peak=False):
    first = track[0]
    if peak:
        fix = track_peak_fix(track) or first
        lat0, lon0 = fix["lat"], fix["lon"]
    elif date_str and time_str:
        try:
            d0 = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
            ilat, ilon = interpolate_track_position(track, d0)
            lat0, lon0 = (ilat, ilon) if ilat is not None else (first["lat"], first["lon"])
        except ValueError:
            lat0, lon0 = first["lat"], first["lon"]
    else:
        lat0, lon0 = first["lat"], first["lon"]
    peak_fix = track_peak_fix(track)
    atcf_id = (track[0].get("atcf_id") or "").strip() or str(target)
    return {
        "atcf_id": atcf_id,
        "long_atcf_id": atcf_id,
        "storm_name": target,
        "latitude": lat0,
        "longitude": lon0,
        "basin": "WP",
        "winds": peak_fix.get("wind") if peak_fix else None,
        "pressure": peak_fix.get("pres") if peak_fix else None,
    }


def _storm_bands_for_composite(composite_type):
    bands = {
        "true": [1, 3, 4, 13],
        "infrared": [13],
        "dvorak": [13],
        "sandwich": [3, 13],
        "b03": [3],
        "irv": [3, 13],
        "falsecolor": [3, 13],
        "falsecoloradv": [3, 13],
        "firetemp": [7, 6, 9],
        "fire": [7],
        "dayconv": [5, 3, 7, 8, 10, 13],
    }.get(composite_type)
    if bands is None:
        raise ValueError(f"Unsupported composite for GOES/MTG/MTSAT: {composite_type}")
    return bands


def _process_goes_storm_data(local_files_map, target_area, target_dt,
                             composite_type, resample_type="nearest"):
    def _read(ahi_band, area=None):
        paths = local_files_map.get(ahi_band)
        if not paths:
            raise ValueError(f"Missing GOES band {ahi_band} (ABI C{AHI_TO_ABI[ahi_band]:02d})")
        data, _ = _goes_read_band(paths[0], area if area is not None else target_area)
        return data

    if composite_type in ("infrared", "dvorak"):
        return _read(13), None, None, None

    if composite_type == "sandwich":
        return _read(3), _read(13), None, None

    if composite_type == "b03":
        return _read(3), None, None, None

    if composite_type == "irv":
        vis, ir = _read(3), _read(13)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        vis_norm = np.clip(vis / 100.0 if np.nanmax(vis) > 1.0 else vis, 0.0, 1.0)
        cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
        cos2_sza = np.clip(np.cos(np.radians(sza)), 0.38, 1.0)
        path_sun = 1.0 / cos2_sza
        path_sun_a = 1.0 / cos_sza
        vis_bright = vis_norm * path_sun_a * 0.9 + 0.01
        rayleigh = 0.011 * path_sun + 0.001
        vis_corr = np.clip(vis_bright - rayleigh, 0.0, 1.0)
        day_weight = np.clip((91.0 - sza) / 5.0, 0.0, 1.0)
        night_weight = 1.0 - day_weight
        ir_norm = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
        ir_layer = np.power(ir_norm, 1.5) * 0.66
        result = np.clip(vis_corr * day_weight + ir_layer * night_weight, 0.0, 1.0)
        return result, None, None, None

    if composite_type in ("falsecolor", "falsecoloradv"):
        vis, ir = _read(3), _read(13)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        r, g, b = _false_color_rgb(vis, ir, sza, advanced=(composite_type == "falsecoloradv"))
        return r, g, b, None

    if composite_type == "firetemp":
        r = _linear_normalize(_read(7), 273.0, 350.0)
        g = _linear_normalize(_read(6), 0.0, 50.0)
        b = _linear_normalize(_read(9), 0.0, 50.0)
        return r, g, b, None

    if composite_type == "fire":
        b07 = _read(7)
        return b07, None, None, None

    if composite_type == "dayconv":
        b05, b03, b07, b08, b10, b13 = (_read(5), _read(3), _read(7),
                                           _read(8), _read(10), _read(13))
        r = _linear_normalize(b08 - b10, -35.0, 5.0)
        g = _linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
        b = _linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
        return r, g, b, None

    if composite_type == "true":
        r = _read(3)
        b = _resize_like(_read(1), r.shape)
        v = _resize_like(_read(4), r.shape)
        ir = _resize_like(_read(13), r.shape)
        g = 0.45 * r + 0.10 * v + 0.45 * b
        r, g, b = apply_rgb_corrections(r, g, b, ir, target_area, target_dt, mode=1)
        return r, g, b, None

    raise ValueError(f"Unsupported GOES composite: {composite_type}")


def _process_mtg_storm_data(local_files_map, target_area, target_dt,
                            composite_type, resample_type="nearest"):
    all_files = []
    for paths in local_files_map.values():
        all_files.extend(paths)
    if not all_files:
        raise ValueError("No MTG local files")
    bands_needed = _storm_bands_for_composite(composite_type)
    for b in bands_needed:
        if AHI_TO_FCI[b] is None:
            raise ValueError(f"MTG FCI has no channel for AHI band {b}; "
                             f"composite '{composite_type}' unsupported")

    def _read(ahi_band):
        return _read_mtg_tailored_channel(all_files, AHI_TO_FCI[ahi_band],
                                          target_area, resample_type)

    if composite_type in ("infrared", "dvorak"):
        return _read(13), None, None, None

    if composite_type == "sandwich":
        return _read(3), _read(13), None, None

    if composite_type == "b03":
        return _read(3), None, None, None

    if composite_type == "irv":
        vis, ir = _read(3), _read(13)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        vis_norm = np.clip(vis / 100.0 if np.nanmax(vis) > 1.0 else vis, 0.0, 1.0)
        cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
        cos2_sza = np.clip(np.cos(np.radians(sza)), 0.38, 1.0)
        path_sun = 1.0 / cos2_sza
        path_sun_a = 1.0 / cos_sza
        vis_bright = vis_norm * path_sun_a * 0.9 + 0.01
        rayleigh = 0.011 * path_sun + 0.001
        vis_corr = np.clip(vis_bright - rayleigh, 0.0, 1.0)
        day_weight = np.clip((91.0 - sza) / 5.0, 0.0, 1.0)
        night_weight = 1.0 - day_weight
        ir_norm = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
        ir_layer = np.power(ir_norm, 1.5) * 0.66
        result = np.clip(vis_corr * day_weight + ir_layer * night_weight, 0.0, 1.0)
        return result, None, None, None

    if composite_type in ("falsecolor", "falsecoloradv"):
        vis, ir = _read(3), _read(13)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        r, g, b = _false_color_rgb(vis, ir, sza, advanced=(composite_type == "falsecoloradv"))
        return r, g, b, None

    if composite_type == "firetemp":
        r = _linear_normalize(_read(7), 273.0, 350.0)
        g = _linear_normalize(_read(6), 0.0, 50.0)
        b = _linear_normalize(_read(9), 0.0, 50.0)
        return r, g, b, None

    if composite_type == "fire":
        b07 = _read(7)
        return b07, None, None, None

    if composite_type == "true":
        r = _read(3)
        b = _resize_like(_read(1), r.shape)
        v = _resize_like(_read(4), r.shape)
        ir = _resize_like(_read(13), r.shape)
        g = 0.45 * r + 0.10 * v + 0.45 * b
        r, g, b = apply_rgb_corrections(r, g, b, ir, target_area, target_dt, mode=1)
        return r, g, b, None

    raise ValueError(f"Unsupported MTG composite: {composite_type}")


def _process_mtsat_storm_data(local_files_map, target_area, target_dt,
                              composite_type, resample_type="nearest",
                              sat_source="mtsat"):
    all_files = []
    for paths in local_files_map.values():
        all_files.extend(paths)
    if not all_files:
        raise ValueError("No MTSAT local files")
    bands_needed = _storm_bands_for_composite(composite_type)
    for b in bands_needed:
        if b not in AHI_TO_MTSAT:
            raise ValueError(f"MTSAT has no channel for AHI band {b}; "
                             f"composite '{composite_type}' unsupported")
    channels = sorted({AHI_TO_MTSAT[b] for b in bands_needed})
    reader = _mtsat_cfg(sat_source)["reader"]
    scn = Scene(filenames=all_files, reader=reader)
    scn.load(channels)
    
    if target_area is None:
        scn = scn.compute(scheduler="sync")

        def _native(ahi_band):
            return np.asarray(scn[AHI_TO_MTSAT[ahi_band]].data).astype(np.float32)

        if composite_type in ("infrared", "dvorak"):
            return _native(13), None, None, None
        if composite_type == "b03":
            return _native(3), None, None, None
        return _native(3), _native(13), None, None

    res = scn.resample(target_area, resampler=resample_type,
                       reduce_data=True, radius_of_influence=60000)

    def _read(ahi_band):
        return np.asarray(res[AHI_TO_MTSAT[ahi_band]].data.compute(scheduler="sync")).astype(np.float32)

    if composite_type in ("infrared", "dvorak"):
        return _read(13), None, None, None

    if composite_type == "sandwich":
        return _read(3), _read(13), None, None

    if composite_type == "b03":
        return _read(3), None, None, None

    if composite_type == "irv":
        vis, ir = _read(3), _read(13)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        vis_norm = np.clip(vis / 100.0 if np.nanmax(vis) > 1.0 else vis, 0.0, 1.0)
        cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
        cos2_sza = np.clip(np.cos(np.radians(sza)), 0.38, 1.0)
        path_sun = 1.0 / cos2_sza
        path_sun_a = 1.0 / cos_sza
        vis_bright = vis_norm * path_sun_a * 0.9 + 0.01
        rayleigh = 0.011 * path_sun + 0.001
        vis_corr = np.clip(vis_bright - rayleigh, 0.0, 1.0)
        day_weight = np.clip((91.0 - sza) / 5.0, 0.0, 1.0)
        night_weight = 1.0 - day_weight
        ir_norm = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
        ir_layer = np.power(ir_norm, 1.5) * 0.66
        result = np.clip(vis_corr * day_weight + ir_layer * night_weight, 0.0, 1.0)
        return result, None, None, None

    if composite_type in ("falsecolor", "falsecoloradv"):
        vis, ir = _read(3), _read(13)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        r, g, b = _false_color_rgb(vis, ir, sza, advanced=(composite_type == "falsecoloradv"))
        return r, g, b, None

    if composite_type == "fire":
        b07 = _read(7)
        return b07, None, None, None

    raise ValueError(f"Unsupported MTSAT composite: {composite_type}")


def _discover_files(sat_source, sat, dt, bands, segments,
                    use_target=False, target_segment=None, temp_dir=None,
                    center_lat=None, center_lon=None, roi_deg=None):
    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        abi_bands = [AHI_TO_ABI[b] for b in bands]
        remote = discover_goes_files(sat, dt, abi_bands)
        if not remote:
            return {}
        return {next((a for a, c in AHI_TO_ABI.items() if c == abi_b), abi_b): files
                for abi_b, files in remote.items()}
    if sat_source == "mtg":
        roi_nswe = None
        if center_lat is not None and center_lon is not None and roi_deg:
            lon0 = ((float(center_lon) + 180.0) % 360.0) - 180.0
            lat0 = float(center_lat)
            half = float(roi_deg)
            if half >= 80.0:
                roi_nswe = None
            else:
                north = min(90.0, lat0 + half)
                south = max(-90.0, lat0 - half)
                west = lon0 - half
                east = lon0 + half
                if west < -180.0 or east > 180.0 or east <= west:
                    logging.info("MTG ROI would wrap dateline or exceed [-180,180]; using full product")
                    roi_nswe = None
                else:
                    roi_nswe = [north, south, west, east]
        return discover_mtg_files(MTG_COLLECTION, dt, bands,
                                  temp_dir=temp_dir or "temp_data", roi_nswe=roi_nswe)
    if sat_source in MTSAT_SAT_CONFIG:
        eff = sat if sat in MTSAT_SAT_CONFIG else sat_source
        return discover_mtsat_files(eff, dt, bands)
    return discover_ahi_files(sat, dt, bands, segments, use_target=use_target,
                              target_segment=target_segment)


def _download_files(sat_source, remote_map, tmpdir, download_workers, decompress_workers):
    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        return download_goes_files(remote_map, tmpdir, download_workers)
    if sat_source == "mtg":
        return remote_map
    if sat_source in MTSAT_SAT_CONFIG:
        return download_mtsat_tars(remote_map, tmpdir)
    return download_and_decompress_all(remote_map, tmpdir, download_workers, decompress_workers)


def _resolve_latest_dt(sat_source, sat, segments, bands, use_target=False):
    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        return get_latest_available_dt_goes(sat)
    if sat_source == "mtg":
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        return now.replace(minute=(now.minute // 10) * 10, second=0,
                           microsecond=0) - datetime.timedelta(minutes=20)
    if sat_source in MTSAT_SAT_CONFIG:
        cfg = _mtsat_cfg(sat if sat in MTSAT_SAT_CONFIG else sat_source)
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        probe = now.replace(minute=0, second=0, microsecond=0)
        for _ in range(24):
            if mtsat_tar_exists(cfg, probe):
                return probe
            probe -= datetime.timedelta(hours=1)
        return None
    return get_latest_available_dt(sat, segments, bands, use_target=use_target)

def _eumetsat_creds_available():
    try:
        import eumdac
    except ImportError:
        return False
    if os.environ.get("EUMETSAT_CONSUMER_KEY") and os.environ.get("EUMETSAT_CONSUMER_SECRET"):
        return True
    base_dir = os.path.dirname(os.path.abspath(__file__))
    creds = os.path.join(base_dir, "creds.txt")
    if os.path.exists(creds):
        try:
            txt = open(creds, encoding="utf-8", errors="replace").read()
            if "EUMETSAT_CONSUMER_KEY" in txt and "EUMETSAT_CONSUMER_SECRET" in txt:
                return True
        except Exception:
            pass
    if os.path.exists(os.path.join(base_dir, "!!VPSIFT.py")):
        return True
    return False

def _auto_satellite_candidates(lat, lon):
    lon_n = ((float(lon) + 180.0) % 360.0) - 180.0
    lat_r = np.radians(float(lat))

    def _view_angle(sub_lon):
        c = np.cos(lat_r) * np.cos(np.radians(lon_n - sub_lon))
        c = max(-1.0, min(1.0, float(c)))
        return float(np.degrees(np.arccos(c)))

    entries = [
        ("him",    140.7, "Himawari-9/8", 0.0),
        ("gk2a",   128.2, "GK-2A",        0.0),
        ("goes19", -75.2, "GOES-19",      0.0),
        ("goes18", -137.0, "GOES-18",     0.0),
        ("goes16", -75.2, "GOES-16",      0.5),
        ("goes17", -137.0, "GOES-17",     0.5),
    ]
    if _eumetsat_creds_available():
        entries.append(("mtg", 0.0, "MTG-I1", 0.3))

    ranked = []
    for src, sub_lon, name, penalty in entries:
        a = _view_angle(sub_lon)
        if a >= 81.0:
            continue
        ranked.append((a + penalty, src, name))
    ranked.sort(key=lambda t: t[0])
    return [(src, name) for _, src, name in ranked]


def _auto_probe_satellite(sat_source, date_str, time_str, bands, use_target=False):
    if time_str:
        d = date_str or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        try:
            fixed_dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
        except ValueError:
            return None, None
    else:
        fixed_dt = None

    if sat_source == "him":
        segs = [f"S{i:02d}" for i in range(1, 11)]
        for bucket in ("noaa-himawari9", "noaa-himawari8"):
            try:
                probe_dt = fixed_dt or get_latest_available_dt(
                    bucket, segs, bands, use_target=use_target)
                if probe_dt is None:
                    continue
                if discover_ahi_files(bucket, probe_dt, bands, segs,
                                      use_target=use_target):
                    return probe_dt, bucket
            except Exception as e:
                logging.debug(f"  auto-probe {bucket}: {e}")
        return None, None

    if sat_source == "gk2a":
        try:
            probe_dt = fixed_dt or get_latest_available_dt_gk2a(bands)
            if probe_dt is None:
                return None, None
            if discover_gk2a_files(probe_dt, bands):
                return probe_dt, GK2A_BUCKET
        except Exception as e:
            logging.debug(f"  auto-probe gk2a: {e}")
        return None, None

    if sat_source in ("goes16", "goes17", "goes18", "goes19"):
        try:
            probe_dt = fixed_dt or get_latest_available_dt_goes(sat_source)
            if probe_dt is None:
                return None, None
            abi_bands = [AHI_TO_ABI[b] for b in bands]
            if discover_goes_files(sat_source, probe_dt, abi_bands):
                return probe_dt, GOES_SATELLITE_MAP[sat_source]
        except Exception as e:
            logging.debug(f"  auto-probe {sat_source}: {e}")
        return None, None

    if sat_source == "mtg":
        try:
            eumdac = _ensure_mtg_setup()
            key, secret = _load_eumetsat_creds()
            token = eumdac.AccessToken((key, secret))
            store = eumdac.DataStore(token)
            coll = store.get_collection(MTG_COLLECTION)
            probe_dt = fixed_dt
            if probe_dt is None:
                now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                probe_dt = now.replace(minute=(now.minute // 10) * 10,
                                       second=0, microsecond=0) - datetime.timedelta(minutes=20)
            hits = coll.search(dtstart=probe_dt - datetime.timedelta(minutes=5),
                               dtend=probe_dt + datetime.timedelta(minutes=5))
            if hits.first() is not None:
                return probe_dt, MTG_COLLECTION
        except Exception as e:
            logging.debug(f"  auto-probe mtg: {e}")
        return None, None

def _global_area(output_width):
    lon_min, lon_max = -180.0, 180.0
    lat_min, lat_max = -80.0, 80.0
    R = 6378137.0
    deg2rad = np.pi / 180.0
    width = output_width
    height = round(width * (lat_max - lat_min) / (lon_max - lon_min))
    x0, x1 = lon_min * R * deg2rad, lon_max * R * deg2rad
    y0, y1 = lat_min * R * deg2rad, lat_max * R * deg2rad
    return AreaDefinition("global", "Global", "eqc",
                          {"proj": "eqc", "lon_0": 0, "lat_ts": 0},
                          width, height, (x0, y0, x1, y1))


def _global_resolve_dt(latest, date_str, time_str):
    if time_str:
        d = date_str if date_str else datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        try:
            dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
            logging.info(f"Global mode: using specified time {dt.strftime('%Y-%m-%d %H:%M')}Z")
            return dt
        except ValueError:
            logging.error(f"Invalid --date {date_str} or --time {time_str}; use YYYYMMDD and HHMM")
            return None
    him_segs = [f"S{i:02d}" for i in range(1, 11)]
    dt_him = get_latest_available_dt("noaa-himawari9", him_segs, [13])
    if dt_him is None:
        dt_him = get_latest_available_dt("noaa-himawari8", him_segs, [13])
    dt_gk2a = get_latest_available_dt_gk2a([13])
    dt_goes = get_latest_available_dt_goes(GOES_DEFAULT)
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    dt_mtg = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0) \
        - datetime.timedelta(minutes=20)
    candidates = [x for x in (dt_him, dt_gk2a, dt_goes, dt_mtg) if x is not None]
    if not candidates:
        logging.error("Global mode: could not resolve a shared timestamp.")
        return None
    dt = min(candidates)
    logging.info(f"Global mode: using shared timestamp {dt.strftime('%Y-%m-%d %H:%M')}Z "
                 f"(HIM={dt_him and dt_him.strftime('%H%M')}, "
                 f"GK2A={dt_gk2a and dt_gk2a.strftime('%H%M')}, "
                 f"GOES={dt_goes and dt_goes.strftime('%H%M')}, MTG={dt_mtg.strftime('%H%M')})")
    return dt


def _global_sandwich_rgb(vis, ir, dt, area):
    from pyorbital.astronomy import sun_zenith_angle
    nan_mask = np.isnan(ir) | np.isnan(vis)
    vis = np.nan_to_num(vis, nan=0.0)
    if np.nanmax(vis) > 1.0:
        vis = vis / 100.0
    vis = np.clip(vis, 0.0, 1.0)
    ir = np.nan_to_num(ir, nan=300.0)
    lons, lats = area.get_lonlats()
    sza = sun_zenith_angle(dt, lons, lats)
    cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
    cos2_sza = np.clip(np.cos(np.radians(sza)), 0.40, 1.0)
    path_sun = 0.8 / cos2_sza
    path_sun_a = 1.0 / cos_sza
    vis_bright = vis * path_sun_a
    rayleigh_vis = 0.011 * path_sun
    vis_corr = np.clip(vis_bright - rayleigh_vis, 0.0, 1.0)
    day_weight = np.clip((90.0 - sza) / 5.0, 0.0, 1.0)
    night_weight = 1.0 - day_weight
    vis_day = vis_corr * day_weight
    ir_norm = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
    ir_layer = np.power(ir_norm, 1.5) * 2
    r = vis_day + (ir_layer * night_weight)
    g = vis_day + (ir_layer * night_weight)
    b = vis_day + (ir_layer * night_weight)
    saturation_factor = 1.33
    luminance = 0.2989 * r + 0.5870 * g + 0.1140 * b
    r = np.clip(luminance + saturation_factor * (r - luminance), 0.0, 1.0)
    g = np.clip(luminance + saturation_factor * (g - luminance), 0.0, 1.0)
    b = np.clip(luminance + saturation_factor * (b - luminance), 0.0, 1.0)
    ir_rgb = _sandwich_ir_lookup(ir)
    cold = ir < 248.15
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.where(cold[:, :, None], ir_rgb, rgb)
    rgb = np.clip(rgb, 0.0, 1.0) * 255
    rgb = rgb.astype(np.uint8)
    rgb[nan_mask] = 0
    return rgb


def _global_ir_rgb(ir, product):
    nan_mask = np.isnan(ir)
    ir = np.nan_to_num(ir, nan=300.0)
    if product == "dvorak":
        rgb = _dvorak_ir_lookup(ir)
    else:
        rgb = _sandwich_ir_lookup(ir)
    rgb = np.clip(rgb, 0.0, 1.0) * 255
    rgb = rgb.astype(np.uint8)
    rgb[nan_mask] = 0
    return rgb


def plot_global_stacked(panels, out_path, dt_obj, product_name, logo_path,
                        grid=False, grid_thick=0.4, grid_color="#00BFFF", grid_style="--",
                        no_coastlines=False, label=False,
                        export_formats=None, quiet=False):
    if export_formats is None:
        export_formats = ['avif']
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5.2 * n), dpi=160,
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    fig.patch.set_facecolor('black')
    if n == 1:
        axes = [axes]
    extent = [-180, 180, -80, 80]
    for ax, panel in zip(axes, panels):
        ax.set_facecolor('black')
        ax.imshow(panel['rgb'], extent=extent, transform=ccrs.PlateCarree(), origin='upper')
        if not no_coastlines:
            ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.6, edgecolor='#00FF00')
            ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.3, edgecolor='#00FF00', alpha=0.5)
        if grid:
            gl = ax.gridlines(draw_labels=False, linewidth=grid_thick,
                              color=grid_color, alpha=0.6, linestyle=grid_style)
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
            gl.ylocator = mticker.FixedLocator(np.arange(-80, 81, 20))
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(panel['name'], color='white', fontsize=13, fontweight='bold',
                     loc='left', pad=4)
        ax.axis('off')
    utc_str = dt_obj.strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle(f"Global Composite — {product_name} — {utc_str}",
                 color='white', fontsize=15, fontweight='bold', y=0.995)
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert('RGBA')
            fig.figimage(np.array(logo), xo=10, yo=10, origin='upper', zorder=50)
        except Exception as e:
            logging.warning(f"Failed to draw logo: {e}")

    base_path = _export_base_path(out_path)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=None, pad_inches=0)
    buf.seek(0)
    with Image.open(buf) as img:
        for fmt in export_formats:
            fmt = fmt.lower().strip()
            out_fmt_path = f"{base_path}.{fmt}"
            try:
                if os.path.exists(out_fmt_path):
                    try:
                        os.remove(out_fmt_path)
                    except OSError:
                        pass
                if fmt == 'avif':
                    img.save(out_fmt_path, format='AVIF', quality=95, subsampling="4:4:4")
                elif fmt == 'png':
                    img.save(out_fmt_path, format='PNG', compress_level=1)
                elif fmt == 'jpg' or fmt == 'jpeg':
                    img.save(out_fmt_path, format='JPEG', quality=95, subsampling=0)
                elif fmt == 'webp':
                    img.save(out_fmt_path, format='WEBP', quality=95, method=6)
                else:
                    continue
                if not quiet:
                    logging.info(f"Saved: {out_fmt_path}")
            except OSError as e:
                alt = f"{base_path}.png"
                logging.warning(f"Failed to save {fmt} to {out_fmt_path!r} ({e}); falling back to PNG: {alt}")
                try:
                    img.save(alt, format='PNG', compress_level=1)
                    if not quiet:
                        logging.info(f"Saved: {alt}")
                except OSError as e2:
                    logging.error(f"PNG fallback also failed for {alt}: {e2}")
    plt.close(fig)


def process_global(output_dir, output_width, product, logo_path,
                   latest=False, date_str=None, time_str=None,
                   grid=False, grid_thick=0.4, grid_color="#00BFFF", grid_style="--",
                   no_coastlines=False, label=False,
                   export_formats=None, download_workers=16, quiet=False):
    product = (product or "ir").lower()
    if product not in ("sandwich", "ir", "infrared", "dvorak", "bt0"):
        product = "ir"
        logging.warning(f"Global mode supports sandwich/ir/infrared/dvorak; using 'ir'.")
    want_vis = product == "sandwich"
    bands_ir = [13]
    bands_vis = [3, 13] if want_vis else bands_ir

    try:
        _ensure_mtg_setup()
    except RuntimeError as e:
        logging.error(f"Global: MTG setup required but missing — {e}")
        sys.exit(1)

    dt = _global_resolve_dt(latest, date_str, time_str)
    if dt is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    global_area = _global_area(output_width)
    panels = []
    
    him_sat = None
    for cand in ("noaa-himawari9", "noaa-himawari8"):
        try:
            remote = discover_ahi_files(cand, dt, bands_vis,
                                        [f"S{i:02d}" for i in range(1, 11)])
            if remote:
                him_sat = cand
                break
        except Exception as e:
            logging.warning(f"Himawari discovery failed for {cand}: {e}")
    if him_sat:
        tmpdir = tempfile.mkdtemp(prefix="automata_global_him_")
        try:
            local = download_and_decompress_all(remote, tmpdir, download_workers, 8)
            if local:
                if want_vis:
                    vis, ir, _, _ = process_vpsift_ahi_data(
                        local, global_area, dt, "sandwich", resample_type="nearest", sat_source="him")
                    rgb = _global_sandwich_rgb(vis, ir, dt, global_area)
                else:
                    ir, _, _, _ = process_vpsift_ahi_data(
                        local, global_area, dt, "infrared", resample_type="nearest", sat_source="him")
                    rgb = _global_ir_rgb(ir, product)
                panels.append({"name": "HIMAWARI-9 (AHI)" if "himawari9" in him_sat else "HIMAWARI-8 (AHI)", "rgb": rgb})
                logging.info(f"Global: HIMAWARI panel ready")
        except Exception as e:
            logging.error(f"Global: Himawari processing failed: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        logging.warning("Global: no Himawari data at shared timestamp; skipping panel.")

    try:
        remote = discover_gk2a_files(dt, bands_vis)
        if remote:
            tmpdir = tempfile.mkdtemp(prefix="automata_global_gk2a_")
            try:
                local = download_gk2a_files(remote, tmpdir, download_workers)
                if local:
                    if want_vis:
                        vis, ir, _, _ = process_gk2a_data(local, global_area, dt, "sandwich", "nearest")
                        rgb = _global_sandwich_rgb(vis, ir, dt, global_area)
                    else:
                        ir, _, _, _ = process_gk2a_data(local, global_area, dt, "infrared", "nearest")
                        rgb = _global_ir_rgb(ir, product)
                    panels.append({"name": "GK-2A (AMI)", "rgb": rgb})
                    logging.info(f"Global: GK2A panel ready")
            except Exception as e:
                logging.error(f"Global: GK2A processing failed: {e}")
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            logging.warning("Global: no GK2A data at shared timestamp; skipping panel.")
    except Exception as e:
        logging.warning(f"Global: GK2A skipped ({e})")

    try:
        remote = discover_goes_files(GOES_DEFAULT, dt, bands_vis)
        if remote:
            tmpdir = tempfile.mkdtemp(prefix="automata_global_goes_")
            try:
                local = download_goes_files(remote, tmpdir, download_workers)
                if local:
                    ir, vis = _global_read_goes(local, global_area, want_vis)
                    if want_vis and vis is not None:
                        rgb = _global_sandwich_rgb(vis, ir, dt, global_area)
                    else:
                        rgb = _global_ir_rgb(ir, product)
                    panels.append({"name": f"GOES-{GOES_DEFAULT[-2:].upper()} (ABI)", "rgb": rgb})
                    logging.info(f"Global: GOES panel ready")
            except Exception as e:
                logging.error(f"Global: GOES processing failed: {e}")
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            logging.warning("Global: no GOES data at shared timestamp; skipping panel.")
    except Exception as e:
        logging.warning(f"Global: GOES skipped ({e})")

    try:
        remote = discover_mtg_files(MTG_COLLECTION, dt, bands_ir, temp_dir=output_dir)
        if remote:
            local = remote
            ir, vis = _global_read_mtg(local, global_area, want_vis)
            if want_vis and vis is not None:
                rgb = _global_sandwich_rgb(vis, ir, dt, global_area)
            else:
                rgb = _global_ir_rgb(ir, product)
            panels.append({"name": "MTG-I1 (FCI)", "rgb": rgb})
            logging.info(f"Global: MTG panel ready")
        else:
            logging.warning("Global: no MTG data at shared timestamp; skipping panel.")
    except Exception as e:
        logging.warning(f"Global: MTG skipped ({e})")

    if not panels:
        logging.error("Global: no satellite panels could be built; nothing to render.")
        return

    product_name = {"sandwich": "Sandwich", "dvorak": "Dvorak",
                    "ir": "IR", "infrared": "IR", "bt0": "BT0"}.get(product, "IR")
    ts = dt.strftime('%Y%m%d_%H%M')
    out_path = os.path.join(output_dir, f"global_{ts}_{product_name.lower()}")
    logging.info(f"Global: rendering {len(panels)} stacked panel(s) -> {out_path}")
    plot_global_stacked(panels, out_path, dt, product_name, logo_path,
                        grid=grid, grid_thick=grid_thick, grid_color=grid_color,
                        grid_style=grid_style, no_coastlines=no_coastlines,
                        label=label, export_formats=export_formats, quiet=quiet)

def make_sandwich(local_files_map, target_area):
    all_files = []
    for paths in local_files_map.values():
        all_files.extend(paths)
    if not all_files:
        raise ValueError("No local files")
    scn = Scene(filenames=all_files, reader="ahi_hsd")
    scn.load(["B03", "B13"])
    res = scn.resample(target_area, resampler="nearest",
                       reduce_data=True, radius_of_influence=50000)
    vis_lazy = res["B03"].data
    ir_lazy = res["B13"].data
    vis, ir = dask.compute(vis_lazy, ir_lazy)
    vis = vis.astype(np.float32)
    ir = ir.astype(np.float32)

    vis = np.nan_to_num(vis, nan=0.0)
    if np.nanmax(vis) > 1.0:
        vis = vis / 100.0
    vis = np.clip(vis, 0.0, 1.0)

    ir = np.nan_to_num(ir, nan=300.0)
    rgb = np.stack([vis, vis, vis], axis=-1)
    cold = ir < 248.0
    rgb[cold, 0] = 1.0
    rgb[cold, 1] = 0.3
    rgb[cold, 2] = 0.0

    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

def _point_in_par(lon, lat):
    par_pts = [(115,5), (115,15), (120,21), (120,25), (135,25), (135,5)]
    inside = False
    j = len(par_pts) - 1
    for i in range(len(par_pts)):
        yi, xi = par_pts[i]
        yj, xj = par_pts[j]
        if ((xi > lat) != (xj > lat)) and (lon < (yj - yi) * (lat - xi) / (xj - xi) + yi):
            inside = not inside
        j = i
    return inside


def imported_by(ax, product):
    MAP_TEXT_BOX_X = 0.005
    MAP_TEXT_BOX_Y = 0.005
    MAP_TEXT_BOX_BG_COLOR = 'black'
    MAP_TEXT_BOX_BG_OPACITY = 0.5
    MAP_TEXT_BOX_OUTLINE_WIDTH = 1
    MAP_TEXT_BOX_OUTLINE_COLOR = 'black'
    MAP_TEXT_BOX_OUTLINE_OPACITY = 0.8
    MAP_TEXT_BOX_TEXT_COLOR = 'white'
    MAP_TEXT_BOX_FONT_SIZE = 12
    MAP_TEXT_BOX_FONT_WEIGHT = 'normal'

    prod = (product or '').upper()
    if prod.startswith("Z1-"):
        MAP_TEXT_BOX_CUSTOM_TEXT = "CMAP From @z136a1"
    elif prod.startswith("ALTHEA"):
        MAP_TEXT_BOX_CUSTOM_TEXT = "CMAP from Althea Kate (@thea_girl)"
    else:
        return ax

    ax.text(MAP_TEXT_BOX_X, MAP_TEXT_BOX_Y, MAP_TEXT_BOX_CUSTOM_TEXT,
            transform=ax.transAxes,
            fontsize=MAP_TEXT_BOX_FONT_SIZE,
            fontweight=MAP_TEXT_BOX_FONT_WEIGHT,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='square',
                      facecolor=MAP_TEXT_BOX_BG_COLOR,
                      alpha=MAP_TEXT_BOX_BG_OPACITY,
                      edgecolor=MAP_TEXT_BOX_OUTLINE_COLOR,
                      linewidth=MAP_TEXT_BOX_OUTLINE_WIDTH),
            color=MAP_TEXT_BOX_TEXT_COLOR)

    return ax


def _draw_geometries_platecarree(ax, feature, color, linewidth, alpha, center_lon, zorder=10):
    for geom in feature.geometries():
        if geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        elif geom.geom_type == 'LineString':
            lines = [geom]
        elif geom.geom_type == 'MultiPolygon':
            lines = [p.boundary for p in geom.geoms]
        elif geom.geom_type == 'Polygon':
            lines = [geom.boundary]
        else:
            continue
        for ln in lines:
            xs, ys = ln.xy
            xs_norm = (np.asarray(xs, dtype=float) - center_lon + 180.0) % 360.0 + center_lon - 180.0
            ax.plot(xs_norm, ys, color=color, linewidth=linewidth,
                    alpha=alpha, zorder=zorder)



def _export_base_path(out_path):
    if not out_path:
        return "output"
    root, ext = os.path.splitext(out_path)
    if ext.lower().lstrip(".") in ("avif", "png", "jpg", "jpeg", "webp", "mp4", "gif", "tif", "tiff"):
        return root if root else out_path
    return out_path


def plot_floater_image(img_data, out_path, metadata, cmap=None, vmin=None, vmax=None, logo_path=None, quiet=False, export_formats=None, return_png=False, dbz_cmap=None):
    sat_name = metadata.get('satellite_name', 'HIMAWARI-9')
    dt_obj = metadata.get('target_dt')
    lat = metadata.get('center_lat', 0.0)
    lon = metadata.get('center_lon', 0.0)
    crop_lon = metadata.get('crop_lon', metadata.get('crop_deg', 5.0))
    crop_lat = metadata.get('crop_lat', metadata.get('crop_deg', 5.0))
    product = metadata.get('product', '')
    storm_id = metadata.get('storm_id', '')
    lon_norm = (lon + 180) % 360 - 180

    if dt_obj:
        utc_str = dt_obj.strftime("%Y-%m-%d %H:%M UTC")
    else:
        utc_str = "Unknown"

    lon_min = lon_norm - crop_lon
    lon_max = lon_norm + crop_lon
    lat_min = lat - crop_lat
    lat_max = lat + crop_lat

    t_extent = metadata.get('target_extent')
    if t_extent:
        lon_min, lon_max, lat_min, lat_max = t_extent
        lon_norm = (lon_min + lon_max) / 2.0
    extent = [lon_min, lon_max, lat_min, lat_max]

    _lon_span = max(lon_max - lon_min, 1e-6)
    _lat_span = max(lat_max - lat_min, 1e-6)
    _mean_lat_fs = 0.5 * (lat_min + lat_max)
    _width_over_height = (_lon_span * max(float(np.cos(np.radians(_mean_lat_fs))), 0.05)) / _lat_span
    _fig_w = 10.0
    _fig_h = max(min(_fig_w / max(_width_over_height, 0.25), 16.0), 4.0)
    if _width_over_height > 1.0:
        _fig_w = min(max(_fig_h * _width_over_height, 6.0), 18.0)
        _fig_h = 10.0
    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h), dpi=200)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if cmap:
        im = ax.imshow(img_data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="upper")
    else:
        interp = 'nearest' if metadata.get('product', '').startswith('RADAR') else 'antialiased'
        im = ax.imshow(img_data, extent=extent, origin="upper",
                       interpolation=interp)

    if not metadata.get('no_coastlines', False):
        coast_color = metadata.get('coastline_color', '#00FF00')
        _draw_geometries_platecarree(ax, cfeature.COASTLINE.with_scale('10m'),
                                     coast_color, 0.5, 1.0, lon_norm)
        _draw_geometries_platecarree(ax, cfeature.BORDERS.with_scale('10m'),
                                     coast_color, 0.3, 0.5, lon_norm)

    crop_km = metadata.get('crop_km', 1000)
    grid_step = 5 if crop_km == 1000 else 10
    ax.grid(True, alpha=0.6, ls=metadata.get('grid_style', '--'),
            color=metadata.get('grid_color', '#00BFFF'),
            linewidth=metadata.get('grid_thick', 0.4))
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(np.ceil(lon_min / grid_step) * grid_step,
                            lon_max + grid_step, grid_step))
    ax.set_yticks(np.arange(np.ceil(lat_min / grid_step) * grid_step,
                            lat_max + grid_step, grid_step))

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    _mean_lat = 0.5 * (lat_min + lat_max)
    _geo_aspect = 1.0 / max(float(np.cos(np.radians(_mean_lat))), 0.05)
    try:
        ax.set_aspect(_geo_aspect, adjustable="box")
    except Exception:
        ax.set_aspect("equal")
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    ax.set_xlabel("Longitude (°E)", fontsize=9, color="white")
    ax.set_ylabel("Latitude (°N)", fontsize=9, color="white")
    ax.tick_params(colors="white", labelsize=8)

    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    _radar_ov = metadata.get("radar_overlay")
    _radar_tag = ""
    if _radar_ov:
        _rsrc = _radar_ov.get("source") or "Radar"
        _rtype = _radar_ov.get("type") or ""
        _radar_tag = f" + {_rsrc}"
        if _rtype:
            _radar_tag += f" {_rtype}"
    ax.set_title(f"{sat_name} {product.upper()}{_radar_tag}  {abs(lat):.2f}{lat_dir} "
                 f"{abs(lon):.2f}{lon_dir}  {utc_str}",
                 fontsize=11, fontweight="bold", color="white")

    astorms = metadata.get('active_storms', None)
    e_lon = [lon_min, lon_max]
    e_lat = [lat_min, lat_max]
    if astorms and (metadata.get('ico', False) or metadata.get('invest', False)):
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        ico_dir = os.path.join(script_dir, 'ico')
        for s in astorms:
            slat = s.get('latitude')
            slot = s.get('longitude')
            if slat is None or slot is None:
                continue
            if not (e_lon[0] <= slot <= e_lon[1] and e_lat[0] <= slat <= e_lat[1]):
                continue
            winds = s.get('winds')
            is_invest = winds is None or winds < 25
            if is_invest and metadata.get('invest', False):
                ax.plot(slot, slat, 'o', color='white', markersize=80,
                        markeredgewidth=2, markerfacecolor='none', zorder=25)
            if not is_invest and metadata.get('ico', False):
                if winds < 34: icon_name = 'td.png'
                elif winds < 48: icon_name = 'ts.png'
                elif winds < 64: icon_name = 'ty.png'
                else: icon_name = 'sty.png'
                icon_path = os.path.join(ico_dir, icon_name)
                if os.path.exists(icon_path):
                    try:
                        icon_img = Image.open(icon_path).convert('RGBA')
                        icon_arr = np.array(icon_img)
                        ax.imshow(icon_arr, extent=[slot - 0.8, slot + 0.8,
                                                    slat - 0.8, slat + 0.8],
                                  origin='upper', zorder=25)
                    except Exception as e:
                        logging.warning(f"Failed to load icon {icon_path}: {e}")

    if cmap:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.9)
        if vmin is not None and vmin < -50:
            bar_label = "BT (°C)"
        elif vmax is not None and vmax > 150:
            bar_label = "BT (K)"
        else:
            bar_label = ""
        cbar.set_label(bar_label, color="white")
        cbar.ax.tick_params(colors="white", labelsize=8)
        for spine in cbar.ax.spines.values():
            spine.set_color("#555555")
    elif dbz_cmap is not None:
        ncolors = dbz_cmap.N
        norm = mcolors.BoundaryNorm(np.arange(0, ncolors + 1), ncolors)
        sm = plt.cm.ScalarMappable(cmap=dbz_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, shrink=0.9)
        ticks = list(range(5, ncolors, 5))
        if ticks[-1] != ncolors:
            ticks.append(ncolors)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])
        cbar.set_label("Reflectivity (dBZ)", color="white")
        cbar.ax.tick_params(colors="white", labelsize=8)
        for spine in cbar.ax.spines.values():
            spine.set_color("#555555")

    if logo_path:
        found_path = None
        for cand in (logo_path, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), logo_path)):
            if os.path.exists(cand):
                found_path = cand
                break
        if found_path:
            try:
                logo_img = Image.open(found_path).convert('RGBA')
                lw_fig = 0.11
                lh_fig = lw_fig * logo_img.height / logo_img.width
                logo_ax = fig.add_axes([0.875, 0.875, lw_fig, lh_fig])
                logo_ax.axis("off")
                logo_ax.imshow(np.array(logo_img), origin="upper", zorder=25)
            except Exception as e:
                logging.warning(f"Logo failed: {e}")

    imported_by(ax, product)

    if export_formats is None:
        export_formats = ['avif']

    base_path = _export_base_path(out_path)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    buf.seek(0)
    with Image.open(buf) as img:
        return_bytes = None
        if return_png:
            out_buf = io.BytesIO()
            img.save(out_buf, format='PNG', compress_level=1)
            return_bytes = out_buf.getvalue()
        for fmt in export_formats:
            fmt = fmt.lower().strip()
            out_fmt_path = f"{base_path}.{fmt}"
            try:
                if os.path.exists(out_fmt_path):
                    try:
                        os.remove(out_fmt_path)
                    except OSError:
                        pass
                if fmt == 'avif':
                    img.save(out_fmt_path, format='AVIF', quality=95, subsampling="4:4:4")
                elif fmt == 'png':
                    img.save(out_fmt_path, format='PNG', compress_level=1)
                elif fmt == 'jpg' or fmt == 'jpeg':
                    img.save(out_fmt_path, format='JPEG', quality=95, subsampling=0)
                elif fmt == 'webp':
                    img.save(out_fmt_path, format='WEBP', quality=95, method=6)
                elif fmt == 'mp4':
                    png_path = f"{base_path}.png"
                    if os.path.exists(png_path):
                        try:
                            os.remove(png_path)
                        except OSError:
                            pass
                    img.save(png_path, format='PNG', compress_level=1)
                    logging.info(f"Frame saved: {png_path}")
                    continue
                else:
                    logging.warning(f"Unknown export format: {fmt}, skipping")
                    continue
                if not quiet:
                    logging.info(f"Saved: {out_fmt_path}")
            except OSError as e:
                alt = f"{base_path}.png"
                logging.warning(f"Failed to save {fmt} to {out_fmt_path!r} ({e}); falling back to PNG: {alt}")
                try:
                    img.save(alt, format='PNG', compress_level=1)
                    if not quiet:
                        logging.info(f"Saved: {alt}")
                except OSError as e2:
                    logging.error(f"PNG fallback also failed for {alt}: {e2}")
    plt.close(fig)
    return return_bytes

def add_modern_info(ax, metadata, logo_path=None):
    fig = ax.figure
    storm_id = metadata.get('storm_id', 'UNKNOWN')
    storm_name = metadata.get('storm_name', '')
    dt_obj = metadata.get('target_dt')
    lat = metadata.get('center_lat', 0.0)
    lon = metadata.get('center_lon', 0.0)
    product = metadata.get('product', '')
    sat_name = metadata.get('satellite_name', 'HIMAWARI-9')
    winds = metadata.get('winds')
    pressure = metadata.get('pressure')
    crop_lon = metadata.get('crop_lon', metadata.get('crop_deg', 5.0))
    crop_lat = metadata.get('crop_lat', metadata.get('crop_deg', 5.0))
    active_storms = metadata.get('active_storms', None)

    crop_ratio = crop_lat / crop_lon
    if crop_ratio < 1:
        margin = (1 - crop_ratio) / 2
    else:
        margin = 0

    if dt_obj:
        pht = dt_obj + datetime.timedelta(hours=8)
        time_str = pht.strftime("%Y-%m-%d %I:%M %p PHT")
    else:
        time_str = "Unknown time"

    lines = []
    if storm_name and storm_id:
        lines.append(f"{storm_name} ({storm_id})")
    elif storm_id:
        lines.append(f"Storm {storm_id}")
    else:
        lines.append("Storm")
    lines.append(f"{time_str}")
    lines.append(f"{sat_name} • {product.upper()}")
    radar_ov = metadata.get("radar_overlay")
    if radar_ov:
        src = radar_ov.get("source") or "Radar"
        rtype = radar_ov.get("type") or ""
        rts = radar_ov.get("ts") or ""
        
        label = f"Radar: {src}"
        line = f"Radar: {src}"
        if rtype:
            line += f" ({rtype})"
        lines.append(line)

        if rts:
            try:
                radar_dt_utc = datetime.datetime.strptime(rts, "%Y%m%d%H%M").replace(
                    tzinfo=datetime.timezone.utc
                )
                radar_dt_pht = radar_dt_utc + datetime.timedelta(hours=8)

                target_dt = metadata.get('target_dt')
                same_date = False
                if target_dt is not None and isinstance(target_dt, datetime.datetime):
                    if target_dt.tzinfo is None:
                        target_dt_utc = target_dt.replace(tzinfo=datetime.timezone.utc)
                    else:
                        target_dt_utc = target_dt
                    target_dt_pht = target_dt_utc + datetime.timedelta(hours=8)
                    same_date = (radar_dt_pht.date() == target_dt_pht.date())

                if same_date:
                    time_str = radar_dt_pht.strftime("%I:%M %p PHT")
                else:
                    time_str = radar_dt_pht.strftime("%Y-%m-%d %I:%M %p PHT")

                lines.append(f"Radar Time: {time_str}")
            except ValueError:
                lines.append(f"Radar Time: {rts}")
    if winds is not None:
        lines.append(f"Winds: {winds:.0f} kt")
    if pressure is not None:
        lines.append(f"Pressure: {pressure:.0f} hPa")
    if storm_id not in ('PHL', 'WPAC', 'PMD', 'NL', 'SL', 'IPAR', 'FLDK', 'TARGET'):
        lines.append(f"Lat: {lat:.2f}°  Lon: {lon:.2f}°")

    if active_storms and storm_id in ('PHL', 'WPAC', 'PMD', 'NL', 'SL', 'IPAR'):
        lines.append("")
        lines.append("Storms:")
        x0, x1 = -crop_lon, crop_lon
        y0, y1 = lat - crop_lat, lat + crop_lat
        for s in active_storms:
            slat = s.get('latitude')
            slot = s.get('longitude')
            if slat is None or slot is None:
                continue
            sx = slot - lon
            if x0 <= sx <= x1 and y0 <= slat <= y1:
                sname = (s.get('storm_name') or '').upper()
                sid = s.get('atcf_id', '')
                label = f"{sname} ({sid})" if sname else sid
                inside_par = _point_in_par(slot, slat)
                label += " [PAR]" if inside_par else " [OUT]"
                lines.append(label)

    if logo_path:
        found_path = None
        if os.path.exists(logo_path):
            found_path = logo_path
        else:
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            alt_path = os.path.join(script_dir, logo_path)
            if os.path.exists(alt_path):
                found_path = alt_path
            else:
                base = os.path.basename(logo_path)
                alt_path2 = os.path.join(script_dir, base)
                if os.path.exists(alt_path2):
                    found_path = alt_path2

        if found_path:
            try:
                logo_img = Image.open(found_path).convert('RGBA')
                logo_array = np.array(logo_img)

                logo_width_fig = max(0.05, 0.15 * crop_ratio)
                h_orig, w_orig = logo_array.shape[:2]
                logo_height_fig = logo_width_fig * (h_orig / w_orig)

                logo_x = 0.99 - logo_width_fig
                logo_y = 1 - margin - 0.01 - logo_height_fig

                logo_ax = fig.add_axes([logo_x, logo_y, logo_width_fig, logo_height_fig])
                logo_ax.axis('off')
                logo_ax.imshow(logo_array, origin='upper', zorder=25)

                logging.info(f"Logo loaded from: {found_path}")

            except Exception as e:
                logging.warning(f"Failed to load logo from {found_path}: {e}")
        else:
            logging.warning(f"Logo file not found: {logo_path}")

    if metadata.get('info', True):
        text_str = "\n".join(lines)
        info_fs = 5 if storm_id in ('PHL', 'WPAC', 'PMD', 'NL', 'SL', 'IPAR') else 12
        props = dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='#02fbfb')
        fig.text(0.02, margin + 0.02, text_str,
                 fontsize=info_fs, color='white',
                 family='sans-serif', weight='bold',
                 verticalalignment='bottom', horizontalalignment='left',
                 bbox=props, zorder=20)

def _crop_radar_to_extent(radar_rgba, radar_bounds, center_lon, crop_lon, crop_lat, center_lat, extent_x_shift=0.0):
    min_lon, min_lat, max_lon, max_lat = radar_bounds
    h, w = radar_rgba.shape[:2]
    if w < 2 or h < 2 or not np.isfinite([min_lon, min_lat, max_lon, max_lat]).all() \
            or max_lon <= min_lon or max_lat <= min_lat:
        return radar_rgba

    target_lon_min = center_lon - crop_lon + extent_x_shift
    target_lon_max = center_lon + crop_lon + extent_x_shift
    target_lat_min = center_lat - crop_lat
    target_lat_max = center_lat + crop_lat

    x0 = (target_lon_min - min_lon) / (max_lon - min_lon) * (w - 1)
    x1 = (target_lon_max - min_lon) / (max_lon - min_lon) * (w - 1) + 1
    y0 = (max_lat - target_lat_max) / (max_lat - min_lat) * (h - 1)
    y1 = (max_lat - target_lat_min) / (max_lat - min_lat) * (h - 1) + 1

    x0i = max(0, min(w, int(round(x0))))
    x1i = max(0, min(w, int(round(x1))))
    y0i = max(0, min(h, int(round(y0))))
    y1i = max(0, min(h, int(round(y1))))
    if x1i <= x0i or y1i <= y0i:
        return None
    return radar_rgba[y0i:y1i, x0i:x1i]




def _sat_subpoint_lon(sat_source):
    if sat_source == "mtg":
        return -0.3
    if sat_source in ("goes", "goes16", "goes19"):
        return -75.0
    if sat_source in ("goes17", "goes18"):
        return -137.0
    if sat_source == "gk2a":
        return 128.2
    if sat_source in ("mtsat", "mtsat2"):
        return 145.0
    if sat_source == "mtsat1":
        return 140.0
    return 140.7


def _project_is_native_geos(name):
    key = (name or "flat").strip().lower().replace("-", "_").replace(" ", "_")
    return key in ("geos", "geostationary", "geo", "native", "nat")


def _geos_area_for_crop(lat, lon, half_lon, half_lat, sat_lon, width, height,
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
            xs.append(float(x))
            ys.append(float(y))
    if len(xs) < 3:
        cx, cy = p(lon_norm, lat)
        if not (np.isfinite(cx) and abs(cx) < 1e20):
            cx, cy = 0.0, 0.0
        m = max(half_lon, half_lat) * 1.2e5
        xs = [cx - m, cx + m]
        ys = [cy - m, cy + m]
    pad = 0.01 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid geos crop extent")
    proj_dict = {
        "proj": "geos",
        "lon_0": sat_lon,
        "h": sat_height,
        "a": a,
        "b": b,
        "sweep": "x",
        "units": "m",
    }
    return AreaDefinition(
        "storm_crop_geos", "Storm Crop (GEOS)", "geos", proj_dict,
        int(width), int(height),
        (x_min, y_min, x_max, y_max),
    )


def _standard_fulldisk_geos_area(sat_source, nx=None, ny=None):
    sat_lon = _sat_subpoint_lon(sat_source)
    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        h = 35786023.0
        half = 5434894.885056
        nx = int(nx or 5424)
        ny = int(ny or 5424)
        sweep = "x"
    elif sat_source == "mtg":
        h = 35786400.0
        half = 5568000.0
        nx = int(nx or 5568)
        ny = int(ny or 5568)
        sweep = "y"
    else:
        h = 35785831.0
        half = 5499999.901531426
        nx = int(nx or 5500)
        ny = int(ny or 5500)
        sweep = "x"
    proj_dict = {
        "proj": "geos",
        "lon_0": float(sat_lon),
        "h": h,
        "a": 6378137.0,
        "b": 6356752.31414,
        "sweep": sweep,
        "units": "m",
    }
    return AreaDefinition(
        "fldk_geos", "Full Disk (native GEOS)", "geos", proj_dict,
        nx, ny, (-half, -half, half, half),
    )


def _area_with_shape(area, shape):
    if area is None or shape is None or len(shape) < 2:
        return area
    ny, nx = int(shape[0]), int(shape[1])
    if area.x_size == nx and area.y_size == ny:
        return area
    return AreaDefinition(
        area.area_id, area.description, area.proj_id,
        area.proj_dict, nx, ny, area.area_extent,
    )


def _align_result_to_area(result, area):
    if result is None or area is None:
        return result
    target = (int(area.y_size), int(area.x_size))
    out = []
    for arr in result:
        if arr is None:
            out.append(None)
        elif getattr(arr, "ndim", 0) == 2 and arr.shape != target:
            out.append(_resize_like(arr, target))
        else:
            out.append(arr)
    return tuple(out)


def _resolve_map_projection(name, lon_norm=0.0, lat=0.0, sat_lon=None):
    key = (name or "flat").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "flat": "flat", "platecarree": "flat", "plate_carree": "flat",
        "eqc": "flat", "equirectangular": "flat", "cylindrical": "flat",
        "disk": "disk", "ortho": "disk", "orthographic": "disk", "globe": "disk",
        "equal_earth": "equal_earth", "equalearth": "equal_earth", "ee": "equal_earth",
        "robinson": "robinson", "mollweide": "mollweide", "moll": "mollweide",
        "mercator": "mercator", "merc": "mercator",
        "sinusoidal": "sinusoidal", "sinu": "sinusoidal",
        "geos": "geos", "geostationary": "geos", "geo": "geos",
        "native": "geos", "nat": "geos",
    }
    kind = aliases.get(key, key)
    if kind == "flat":
        return ccrs.PlateCarree(central_longitude=lon_norm), True
    if kind == "disk":
        return ccrs.Orthographic(central_longitude=lon_norm, central_latitude=lat), False
    if kind == "equal_earth":
        return ccrs.EqualEarth(central_longitude=lon_norm), False
    if kind == "robinson":
        return ccrs.Robinson(central_longitude=lon_norm), False
    if kind == "mollweide":
        return ccrs.Mollweide(central_longitude=lon_norm), False
    if kind == "mercator":
        return ccrs.Mercator(central_longitude=lon_norm), False
    if kind == "sinusoidal":
        return ccrs.Sinusoidal(central_longitude=lon_norm), False
    if kind == "geos":
        sat = float(sat_lon) if sat_lon is not None else lon_norm
        return ccrs.Geostationary(central_longitude=sat, satellite_height=35785831.0), False
    logging.warning(f"Unknown --project '{name}'; falling back to flat (PlateCarree)")
    return ccrs.PlateCarree(central_longitude=lon_norm), True


def plot_image(img_data, out_path, metadata, cmap=None, vmin=None, vmax=None, logo_path=None, quiet=False, export_formats=None, return_png=False):
    sat_name = metadata.get('satellite_name', 'HIMAWARI-9')
    dt_obj = metadata.get('target_dt')
    lat = metadata.get('center_lat', 0.0)
    lon = metadata.get('center_lon', 0.0)
    crop_lon = metadata.get('crop_lon', metadata.get('crop_deg', 5.0))
    crop_lat = metadata.get('crop_lat', metadata.get('crop_deg', 5.0))
    product = metadata.get('product', '')
    storm_id = metadata.get('storm_id', '')
    lon_norm = (lon + 180) % 360 - 180

    project_name = metadata.get('project', 'flat')
    sat_lon_hint = metadata.get('sat_lon')
    data_is_geos = bool(metadata.get('data_is_geos'))
    area_extent = metadata.get('area_extent')
    data_crs = ccrs.PlateCarree()
    sat_h = 35785831.0

    extent_x_shift = 0.0
    if 'PWARDS' in storm_id:
        extent_x_shift = (2.0 * 2.0 * crop_lon) / 2560.0
    geo_extent = [lon_norm - crop_lon + extent_x_shift,
                  lon_norm + crop_lon + extent_x_shift,
                  lat - crop_lat, lat + crop_lat]
    flat_extent = [-crop_lon + extent_x_shift, crop_lon + extent_x_shift,
                   lat - crop_lat, lat + crop_lat]

    if data_is_geos and area_extent is not None and len(area_extent) == 4:
        sat = float(sat_lon_hint) if sat_lon_hint is not None else lon_norm
        proj = ccrs.Geostationary(central_longitude=sat, satellite_height=sat_h)
        is_flat = False
        x0, y0, x1, y1 = (float(area_extent[0]), float(area_extent[1]),
                          float(area_extent[2]), float(area_extent[3]))
        geos_imshow_extent = (x0, x1, y0, y1)
    else:
        proj, is_flat = _resolve_map_projection(
            project_name, lon_norm=lon_norm, lat=lat, sat_lon=sat_lon_hint)
        geos_imshow_extent = None

    fig = plt.figure(figsize=(10, 10), dpi=256)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj, facecolor='black')

    polygon = metadata.get('polygon')
    if polygon is not None and not data_is_geos:
        from matplotlib.path import Path as MplPath
        h, w = img_data.shape[0], img_data.shape[1]
        lon_flat = np.linspace(lon_norm - crop_lon, lon_norm + crop_lon, w)
        lat_flat = np.linspace(lat + crop_lat, lat - crop_lat, h)
        lon_grid, lat_grid = np.meshgrid(lon_flat, lat_flat)
        pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        inside = MplPath(polygon).contains_points(pts).reshape(h, w)
        img_data[~inside] = 0

    if data_is_geos and geos_imshow_extent is not None:
        if cmap:
            im = ax.imshow(img_data, extent=geos_imshow_extent, origin='upper',
                           cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img_data, extent=geos_imshow_extent, origin='upper')
    else:
        extent = flat_extent if is_flat else geo_extent
        im_transform = proj if is_flat else data_crs
        if cmap:
            im = ax.imshow(img_data, extent=extent, transform=im_transform,
                           cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        else:
            im = ax.imshow(img_data, extent=extent, transform=im_transform, origin='upper')

    if not metadata.get('no_coastlines', False):
        coast_color = metadata.get('coastline_color', '#00FF00')
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, edgecolor=coast_color, alpha=1)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.3, edgecolor=coast_color, alpha=0.5)

    radar_ov = metadata.get('radar_overlay')
    if radar_ov:
        try:
            rb = radar_ov['bounds']
            rgb = radar_ov['rgb']
            if rgb.ndim != 3 or rgb.shape[-1] != 4:
                raise ValueError("radar overlay must be RGBA")

            if np.all(rgb[..., 3] > 0.99):
                gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                bg = gray < 0.1
                rgb[..., 3] = np.where(bg, 0.0, 1.0)

            if (not is_flat) or data_is_geos:
                rmin_lon, rmin_lat, rmax_lon, rmax_lat = rb
                ax.imshow(rgb, extent=[rmin_lon, rmax_lon, rmin_lat, rmax_lat],
                          transform=data_crs, origin='upper', zorder=5, interpolation='nearest')
                logging.info("Radar overlay drawn in geographic CRS (projection-aware).")
            else:
                if cmap is not None:
                    data = img_data.astype(np.float32)
                    vmin_ = vmin if vmin is not None else np.nanmin(data)
                    vmax_ = vmax if vmax is not None else np.nanmax(data)
                    if vmax_ <= vmin_:
                        vmax_ = vmin_ + 1e-6
                    normed = np.clip((data - vmin_) / (vmax_ - vmin_), 0, 1)
                    sat_rgba = cmap(normed)
                else:
                    sat_rgba = img_data.astype(np.float32)
                    if sat_rgba.max() > 1.0:
                        sat_rgba = sat_rgba / 255.0
                    if sat_rgba.ndim == 2:
                        sat_rgba = np.stack([sat_rgba]*3, axis=-1)
                        sat_rgba = np.concatenate([sat_rgba, np.ones_like(sat_rgba[..., :1])], axis=-1)
                    elif sat_rgba.shape[-1] == 3:
                        sat_rgba = np.concatenate([sat_rgba, np.ones_like(sat_rgba[..., :1])], axis=-1)

                radar_crop = _crop_radar_to_extent(
                    rgb, rb, lon_norm, crop_lon, crop_lat, lat, extent_x_shift)
                if radar_crop is not None:
                    sat_h, sat_w = sat_rgba.shape[:2]
                    if (radar_crop.shape[0], radar_crop.shape[1]) != (sat_h, sat_w):
                        radar_pil = Image.fromarray((radar_crop * 255).astype(np.uint8), mode='RGBA')
                        radar_pil = radar_pil.resize((sat_w, sat_h), Image.NEAREST)
                        radar_resized = np.asarray(radar_pil).astype(np.float32) / 255.0
                    else:
                        radar_resized = radar_crop

                    alpha = radar_resized[..., 3:4]
                    blended = radar_resized[..., :3] * alpha + sat_rgba[..., :3] * (1 - alpha)
                    blended_rgba = np.concatenate([blended, np.ones_like(alpha)], axis=-1)

                    im.set_data((blended_rgba * 255).astype(np.uint8))
                    logging.info("Radar composited (cropped to satellite extent).")
        except Exception as e:
            logging.warning(f"Failed to composite radar: {e}")
            import traceback
            traceback.print_exc()
        
    try:
        if data_is_geos and metadata.get('fulldisk') and area_extent is not None:
            x0, y0, x1, y1 = area_extent
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
        elif data_is_geos:
            ax.set_extent(geo_extent, crs=data_crs)
        elif is_flat:
            ax.set_extent(flat_extent, crs=proj)
        else:
            ax.set_extent(geo_extent, crs=data_crs)
    except Exception as e:
        logging.warning(f"set_extent failed for project={project_name}: {e}")
        if data_is_geos and area_extent is not None:
            try:
                x0, y0, x1, y1 = area_extent
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
            except Exception:
                pass
    ax.axis('off')

    crop_km = metadata.get('crop_km', 1000)
    if metadata.get('fulldisk'):
        grid_step = 10
    elif crop_km == 1000:
        grid_step = 5
    else:
        grid_step = 10
    if metadata.get('grid', False):
        gl = ax.gridlines(draw_labels=False, linewidth=metadata.get('grid_thick', 0.4),
                          color=metadata.get('grid_color', '#00BFFF'), alpha=0.6,
                          linestyle=metadata.get('grid_style', '--'))
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, grid_step))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, grid_step))

    if metadata.get('label', False):
        e_lon = [lon_norm - crop_lon, lon_norm + crop_lon]
        e_lat = [lat - crop_lat, lat + crop_lat]
        lon_vals = np.arange(np.ceil(e_lon[0] / grid_step) * grid_step, np.floor(e_lon[1] / grid_step) * grid_step + grid_step, grid_step)
        lat_vals = np.arange(np.ceil(e_lat[0] / grid_step) * grid_step, np.floor(e_lat[1] / grid_step) * grid_step + grid_step, grid_step)
        fs = 4 if storm_id in ('PHL', 'WPAC', 'PMD', 'NL', 'SL', 'IPAR') else 8
        bbox_w = dict(facecolor='white', alpha=1.0, edgecolor='none', pad=1, boxstyle='round,pad=0.3')
        px = (e_lat[1] - e_lat[0]) * 0.0012
        px_lr = (e_lon[1] - e_lon[0]) * 0.0008
        for lat_val in lat_vals:
            ax.text(e_lon[0] + px_lr, lat_val, f"{lat_val:.0f}°",
                    transform=ccrs.PlateCarree(), fontsize=fs, color='black', fontweight='bold',
                    ha='left', va='bottom', bbox=bbox_w, zorder=15)
            ax.text(e_lon[1] - px_lr, lat_val, f"{lat_val:.0f}°",
                    transform=ccrs.PlateCarree(), fontsize=fs, color='black', fontweight='bold',
                    ha='right', va='bottom', bbox=bbox_w, zorder=15)
        for lon_val in lon_vals:
            ax.text(lon_val, e_lat[1] + px, f"{lon_val:.0f}°",
                    transform=ccrs.PlateCarree(), fontsize=fs, color='black', fontweight='bold',
                    ha='left', va='top', bbox=bbox_w, zorder=15)
            ax.text(lon_val, e_lat[0] - px, f"{lon_val:.0f}°",
                    transform=ccrs.PlateCarree(), fontsize=fs, color='black', fontweight='bold',
                    ha='left', va='bottom', bbox=bbox_w, zorder=15)

    poly_lw = metadata.get('grid_thick', 0.4)
    if metadata.get('par', False):
        pts = [(115,5), (115,15), (120,21), (120,25), (135,25), (135,5)]
        lons = [p[0] for p in pts] + [pts[0][0]]
        lats = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(lons, lats, color='#e5321d', linewidth=poly_lw, transform=ccrs.PlateCarree(), zorder=20)
    if metadata.get('tcad', False):
        pts = [(114,4), (114,27), (145,27), (145,4)]
        lons = [p[0] for p in pts] + [pts[0][0]]
        lats = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(lons, lats, color='#f1a408', linewidth=poly_lw, transform=ccrs.PlateCarree(), zorder=20)
    if metadata.get('tcid', False):
        pts = [(110,0), (110,27), (155,27), (155,0)]
        lons = [p[0] for p in pts] + [pts[0][0]]
        lats = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(lons, lats, color='#d0d514', linewidth=poly_lw, transform=ccrs.PlateCarree(), zorder=20)

    astorms = metadata.get('active_storms', None)
    if astorms and (metadata.get('ico', False) or metadata.get('invest', False)):
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        ico_dir = os.path.join(script_dir, 'ico')
        for s in astorms:
            slat = s.get('latitude')
            slot = s.get('longitude')
            if slat is None or slot is None:
                continue
            sx = slot - lon_norm
            if not (-crop_lon <= sx <= crop_lon and lat - crop_lat <= slat <= lat + crop_lat):
                continue
            winds = s.get('winds')
            is_invest = winds is None or winds < 25
            if is_invest and metadata.get('invest', False):
                ax.plot(slot, slat, 'o', color='white', markersize=100,
                        markeredgewidth=1, markerfacecolor='none',
                        transform=ccrs.PlateCarree(), zorder=25)
            if not is_invest and metadata.get('ico', False):
                if winds < 34:
                    icon_name = 'td.png'
                elif winds < 48:
                    icon_name = 'ts.png'
                elif winds < 64:
                    icon_name = 'ty.png'
                else:
                    icon_name = 'sty.png'
                icon_path = os.path.join(ico_dir, icon_name)
                if os.path.exists(icon_path):
                    try:
                        icon_img = Image.open(icon_path).convert('RGBA')
                        icon_arr = np.array(icon_img)
                        ax.imshow(icon_arr, extent=[slot - 0.8, slot + 0.8,
                                                     slat - 0.8, slat + 0.8],
                                  transform=ccrs.PlateCarree(), origin='upper', zorder=25)
                    except Exception as e:
                        logging.warning(f"Failed to load icon {icon_path}: {e}")
                        
    add_modern_info(ax, metadata, logo_path)

    if product.upper().startswith("Z1-"):
        bbox_props = dict(facecolor='black', alpha=0.66, edgecolor='none', pad=4.0)
        ax.text(0.01, 0.99, "CMAP From @z136a1", transform=ax.transAxes, color='white',
                fontsize=10, family='monospace', va='top', ha='left', bbox=bbox_props, zorder=25)
    elif product.upper().startswith("ALTHEA"):
        bbox_props = dict(facecolor='black', alpha=0.66, edgecolor='none', pad=4.0)
        ax.text(0.01, 0.99, "CMAP from Althea Kate (@thea_girl)", transform=ax.transAxes, color='white',
                fontsize=10, family='monospace', va='top', ha='left', bbox=bbox_props, zorder=25)

    if export_formats is None:
        export_formats = ['avif']
    
    base_path = _export_base_path(out_path)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=None, pad_inches=0)
    buf.seek(0)
    with Image.open(buf) as img:
        if is_flat and crop_lon > 0:
            target_h = round(img.width * (crop_lat / crop_lon))
            if target_h != img.height and target_h > 0:
                top = (img.height - target_h) // 2
                img = img.crop((0, top, img.width, top + target_h))
        return_bytes = None
        if return_png:
            out_buf = io.BytesIO()
            img.save(out_buf, format='PNG', compress_level=1)
            return_bytes = out_buf.getvalue()
        for fmt in export_formats:
            fmt = fmt.lower().strip()
            out_fmt_path = f"{base_path}.{fmt}"
            if fmt == 'avif':
                img.save(out_fmt_path, format='AVIF', quality=95, subsampling="4:4:4")
            elif fmt == 'png':
                img.save(out_fmt_path, format='PNG', compress_level=1)
            elif fmt == 'jpg' or fmt == 'jpeg':
                img.save(out_fmt_path, format='JPEG', quality=95, subsampling=0)
            elif fmt == 'webp':
                img.save(out_fmt_path, format='WEBP', quality=95, method=6)
            elif fmt == 'mp4':
                png_path = f"{base_path}.png"
                img.save(png_path, format='PNG', compress_level=1)
                logging.info(f"MP4 export requested - saved as PNG: {png_path} (convert to MP4 externally)")
                continue
            else:
                logging.warning(f"Unknown export format: {fmt}, skipping")
                continue
            if not quiet:
                logging.info(f"Saved: {out_fmt_path}")
    plt.close(fig)
    return return_bytes

def _group_target_files_by_segment(local_dat_map):
    grouped = {}
    for band, paths in local_dat_map.items():
        for path in paths:
            match = re.search(r'_R3(\d{2})_', path)
            if match:
                segment = f"R3{match.group(1)}"
                grouped.setdefault(segment, {}).setdefault(band, []).append(path)
    return grouped


def _native_target_area(local_files_map):
    all_files = []
    for paths in local_files_map.values():
        all_files.extend(paths)
    if not all_files:
        raise ValueError("No local files")
    try:
        import satpy
        from satpy.readers import load_reader
        satpy_dir = os.path.dirname(os.path.abspath(satpy.__file__))
        config_candidates = glob.glob(os.path.join(satpy_dir, 'etc', 'readers', 'ahi_hsd.yaml'))
        if not config_candidates:
            config_candidates = glob.glob(os.path.join(satpy_dir, 'etc', 'readers', '*ahi_hsd*.yaml'))
        if not config_candidates:
            logging.warning("Could not locate ahi_hsd reader config; cannot read target projection")
            return None
        reader = load_reader([config_candidates[0]])
        reader.create_filehandlers(all_files)
        for fhs in reader.file_handlers.values():
            for fh in fhs:
                area = getattr(fh, 'area', None)
                if area is not None:
                    return area
        return None
    except Exception as e:
        logging.warning(f"Failed to read target projection header: {e}")
        return None


def _upscaled_target_area(native_area, output_width):
    return AreaDefinition(
        "target_area", "Target Area", native_area.proj_id,
        native_area.proj_dict, output_width, output_width,
        native_area.area_extent
    )


def _native_res_area(native_area):
    return AreaDefinition(
        "target_area_native", "Target Area (native)", native_area.proj_id,
        native_area.proj_dict, native_area.x_size, native_area.y_size,
        native_area.area_extent
    )


def _upscale_to_width(arr, output_width):
    from PIL import Image
    h, w = arr.shape
    if h == output_width and w == output_width:
        return arr
    mask = np.isfinite(arr)
    fill = np.nanmin(arr) if np.any(mask) else 0.0
    data = np.where(mask, arr, fill)
    img = Image.fromarray(data.astype(np.float32))
    img = img.resize((output_width, output_width))
    out = np.asarray(img).astype(np.float32)
    if np.any(~mask):
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img = mask_img.resize((output_width, output_width))
        out_mask = np.asarray(mask_img) > 0.5
        out = np.where(out_mask, out, np.nan)
    return out


def _target_area_for_segment(seg_map, output_width):
    native = _native_target_area(seg_map)
    if native is None:
        return None, None, None
    upscaled = _upscaled_target_area(native, output_width)
    lons, lats = native.get_lonlats()
    center_lat = float(np.nanmean(lats))
    center_lon = float(np.nanmean(lons))
    return upscaled, center_lat, center_lon


def _target_scan_extent(seg_map):
    native = _native_target_area(seg_map)
    if native is None:
        return None
    lons, lats = native.get_lonlats()
    return [float(np.nanmin(lons)), float(np.nanmax(lons)),
            float(np.nanmin(lats)), float(np.nanmax(lats))]

TARGET_SEGMENT_OFFSETS = {
    "R301": datetime.timedelta(minutes=2, seconds=30),
    "R302": datetime.timedelta(minutes=5),
    "R303": datetime.timedelta(minutes=7, seconds=30),
    "R304": datetime.timedelta(minutes=10),
}

def segment_observation_dt(base_dt, segment):
    offset = TARGET_SEGMENT_OFFSETS.get(segment)
    if offset is None:
        return base_dt
    return base_dt + offset


def resolve_target_scan(dt):
    base = dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)
    frac_min = (dt - base).total_seconds() / 60.0
    best_segment = None
    best_diff = float("inf")
    for segment, offset in TARGET_SEGMENT_OFFSETS.items():
        diff = abs(frac_min - offset.total_seconds() / 60.0)
        if diff < best_diff:
            best_diff = diff
            best_segment = segment
    return base, best_segment


def _generate_single_product(product, ir, ir_celsius, ir_norm_ott, ir_norm_infrared, ir_norm_dvorak,
                                metadata, out_path, logo_path,
                                cmap_infrared, cmap_ott, cmap_dvorak, cmap_dvorak_pwards,
                                cmap_sandwich=None, cmap_dvorak_ir=None, cmap_ott2=None,
                                export_formats=None,
                                local_dat_map=None, seg_area=None, target_dt=None,
                                use_target=False, output_width=0, resample_type="nearest",
                                sat_source="him"):
    try:
        use_floater = metadata.get('floater', False)
        plot_func = plot_floater_image if use_floater else plot_image
        
        if product == "ir":
            metadata['product'] = "BT (PWARDS)"
            plot_func(ir_celsius, out_path, metadata,
                       cmap=cmap_sandwich, vmin=-100, vmax=50, logo_path=logo_path, quiet=True, export_formats=export_formats)

        elif product == "infrared":
            metadata['product'] = "Infrared"
            plot_func(ir_celsius, out_path, metadata,
                       cmap=cmap_infrared, vmin=-100, vmax=50, logo_path=logo_path, quiet=True, export_formats=export_formats)

        elif product == "z1-ir":
            metadata['product'] = "Z1-IR"
            plot_func(ir_celsius, out_path, metadata,
                       cmap=cmap_ott, vmin=-100, vmax=50, logo_path=logo_path, quiet=True, export_formats=export_formats)

        elif product == "althea-ott2":
            metadata['product'] = "ALTHEA-OTT2"
            plot_func(ir_celsius, out_path, metadata,
                       cmap=cmap_ott2, vmin=-100, vmax=50, logo_path=logo_path, quiet=True, export_formats=export_formats)

        elif product == "z1-dvorak":
            metadata['product'] = "Z1-DVORAK"
            plot_func(ir_celsius, out_path, metadata,
                       cmap=cmap_dvorak, vmin=-100, vmax=50, logo_path=logo_path, quiet=True, export_formats=export_formats)

        elif product == "bt0":
            metadata['product'] = "BT0 (PWARDS)"
            plot_func(ir_celsius, out_path, metadata,
                       cmap=cmap_dvorak_ir, vmin=-100, vmax=50, logo_path=logo_path, quiet=True, export_formats=export_formats)

        elif product == "dvorak":
            metadata['product'] = "DVORAK (PWARDS)"
            plot_func(ir_norm_dvorak, out_path, metadata,
                       cmap=cmap_dvorak_pwards, vmin=-100, vmax=50, logo_path=logo_path, quiet=True, export_formats=export_formats)

        elif product == "fire":
            if local_dat_map is None or target_dt is None:
                return f"Unknown IR product: {product}"
            b07, _, _, _ = process_vpsift_ahi_data(local_dat_map, seg_area, target_dt, "b07", sat_source=sat_source)
            if b07 is None:
                return f"Failed to load B07 for fire product"
            b07_celsius = b07 - 273.15
            cmap_fire = mcolors.LinearSegmentedColormap.from_list("hotspot_SIR", hotspot_SIR_nodes)
            metadata['product'] = "Fire (3.9um)"
            plot_func(b07_celsius, out_path, metadata,
                       cmap=cmap_fire, vmin=273.15-273.15, vmax=353.15-273.15, logo_path=logo_path, quiet=True, export_formats=export_formats)

        elif product in RGB_COMPOSITES:
            if local_dat_map is None or target_dt is None:
                return f"Unknown IR product: {product}"
            composite = RGB_COMPOSITES[product]
            if use_target:
                res = process_vpsift_ahi_data(local_dat_map, None, target_dt, composite, sat_source=sat_source)
                res = tuple(_upscale_to_width(a, output_width) if a is not None else None
                            for a in res)
            else:
                res = process_vpsift_ahi_data(local_dat_map, seg_area, target_dt, composite,
                                              resample_type=resample_type, sat_source=sat_source)
            if product == "sandwich":
                vis, ir_s, _, _ = res
                vis = np.nan_to_num(vis, nan=0.0)
                if np.nanmax(vis) > 1.0:
                    vis = vis / 100.0
                vis = np.clip(vis, 0.0, 1.0)
                ir_s = np.nan_to_num(ir_s, nan=300.0)
                from pyorbital.astronomy import sun_zenith_angle
                lons, lats = seg_area.get_lonlats()
                sza = sun_zenith_angle(target_dt, lons, lats)
                cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
                cos2_sza = np.clip(np.cos(np.radians(sza)), 0.40, 1.0)
                path_sun = 0.8 / cos2_sza
                path_sun_a = 1.0 / cos_sza
                vis_bright = vis * path_sun_a
                rayleigh_vis = 0.011 * path_sun
                vis_corr = np.clip(vis_bright - rayleigh_vis, 0.0, 1.0)
                day_weight = np.clip((90.0 - sza) / 5.0, 0.0, 1.0)
                night_weight = 1.0 - day_weight
                vis_day = vis_corr * day_weight
                ir_norm = np.clip((313.15 - ir_s) / (313.15 - 173.15), 0.0, 1.0)
                ir_layer = np.power(ir_norm, 1.5) * 2
                r_final = vis_day + (ir_layer * night_weight)
                g_final = vis_day + (ir_layer * night_weight)
                b_final = vis_day + (ir_layer * night_weight)
                saturation_factor = 1.33
                luminance = 0.2989 * r_final + 0.5870 * g_final + 0.1140 * b_final
                r_final = np.clip(luminance + saturation_factor * (r_final - luminance), 0.0, 1.0)
                g_final = np.clip(luminance + saturation_factor * (g_final - luminance), 0.0, 1.0)
                b_final = np.clip(luminance + saturation_factor * (b_final - luminance), 0.0, 1.0)
                ir_rgb = _sandwich_ir_lookup(ir_s)
                cold_mask = ir_s < 248.15
                rgb = np.stack([r_final, g_final, b_final], axis=-1)
                rgb = np.where(cold_mask[:, :, None], ir_rgb, rgb)
                rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                r, g, b, _ = res
                rgb = _stack_rgb(r, g, b)
            metadata['product'] = RGB_DISPLAY_NAMES[product]
            plot_func(rgb, out_path, metadata, logo_path=logo_path, quiet=True, export_formats=export_formats)

        else:
            return f"Unknown IR product: {product}"

        return f"Done: {product}"
    except Exception as e:
        return f"Failed {product}: {e}"

IR_PRODUCTS = ["ir", "infrared", "z1-ir", "althea-ott2", "z1-dvorak", "bt0", "dvorak", "fire"]

BATCH_PRODUCTS = IR_PRODUCTS + ["sandwich", "falsecolor", "falsecoloradv", "firetemp", "dayconv"]

PRODUCT_BANDS = {
    "sandwich": [3, 13],
    "true": [1, 3, 4, 13],
    "z1-true": [1, 3, 4, 13],
    "dvorak": [13],
    "z1-ir": [13],
    "althea-ott2": [13],
    "z1-dvorak": [13],
    "ir": [13],
    "infrared": [13],
    "bt0": [13],
    "b03": [3],
    "irv": [3, 13],
    "falsecolor": [3, 13],
    "falsecoloradv": [3, 13],
    "firetemp": [6, 7, 9],
    "fire": [7],
    "dayconv": [3, 5, 7, 8, 10, 13],
}

RGB_COMPOSITES = {
    "sandwich": "sandwich",
    "falsecolor": "falsecolor",
    "falsecoloradv": "falsecoloradv",
    "firetemp": "firetemp",
    "dayconv": "dayconv",
}

RGB_DISPLAY_NAMES = {
    "sandwich": "Sandwich (PWARDS)",
    "falsecolor": "False Color",
    "falsecoloradv": "False Color Adv",
    "firetemp": "Fire Temp",
    "dayconv": "Day Convection",
}


def process_storm_batch(storm, crop_km, products, output_dir, output_width,
                        download_workers, decompress_workers, logo_path,
                        latest=False, date_str=None, time_str=None,
                        grid=False, grid_thick=0.4, grid_color="#00BFFF", grid_style="--",
                        no_coastlines=False, label=False,
                        par=False, tcad=False, tcid=False,
                        ico=False, invest=False, peak=False, active_storms=None,
                        data_dir=None, export_formats=None,
                        date_from=None, date_to=None, time_from=None, time_to=None,
                        use_target=False, floater=False, fps=4, nopng=False,
                         track=None, sat_source="him", fulldisk=False, info=False, radar_overlay=None,
                         project="flat"):
    sat_tag = ("GK2A" if sat_source == "gk2a"
               else "GOES" if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19")
               else "MTG" if sat_source == "mtg"
               else "MTS2" if sat_source in ("mtsat", "mtsat2")
               else "MTS1" if sat_source == "mtsat1"
               else "HIM9")
    storm_id = storm.get("atcf_id") or storm.get("storm_name", "UNKNOWN")
    storm_name = storm.get("storm_name", "")
    lat = storm.get("latitude")
    lon = storm.get("longitude")
    if lat is None or lon is None:
        logging.warning(f"Storm {storm_id} has no lat/lon; skipping.")
        return

    logging.info(f"Processing {storm_id} at ({lat:.1f}, {lon:.1f}) for products: {products}")

    peak_vmax = None
    peak_pressure = None
    dt = None
    if peak and track is not None:
        _peak_fix = track_peak_fix(track)
        if _peak_fix is not None:
            dt = _peak_fix["dt"]
            lat, lon = _peak_fix["lat"], _peak_fix["lon"]
            peak_vmax = _peak_fix.get("wind")
            peak_pressure = _peak_fix.get("pres")
            logging.info(f"  IBTrACS peak intensity: {peak_vmax if peak_vmax is not None else 'n/a'} "
                         f"at {dt:%Y-%m-%d %H:%MZ} ({lat:.2f}, {lon:.2f})")
    elif peak and storm_id not in ('PHL', 'WPAC'):
        atcf_id = storm.get("atcf_id", "").strip()
        if atcf_id:
            url = f"https://api.knackwx.com/atcf/v2/track/archive?stormID={atcf_id}"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    best_vmax = -1
                    best_pres = None
                    best_dt_str = None
                    best_lat = None
                    best_lon = None
                    for line in r.text.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 9:
                            continue
                        try:
                            vmax = int(parts[8])
                            if vmax > best_vmax:
                                best_vmax = vmax
                                best_dt_str = parts[2]
                                try:
                                    best_pres = int(parts[9])
                                except (ValueError, IndexError):
                                    best_pres = None
                                if len(parts) >= 8:
                                    best_lat = parts[6]
                                    best_lon = parts[7]
                        except (ValueError, IndexError):
                            continue
                    if best_dt_str and best_lat and best_lon:
                        ymdh = best_dt_str.strip()
                        dt = datetime.datetime.strptime(ymdh, "%Y%m%d%H")
                        peak_vmax = best_vmax
                        peak_pressure = best_pres
                        lat_str = best_lat.strip()
                        if lat_str:
                            lat_val = float(lat_str[:-1]) / 10.0
                            if lat_str[-1] == 'S':
                                lat_val = -lat_val
                            lat = lat_val
                        lon_str = best_lon.strip()
                        if lon_str:
                            lon_val = float(lon_str[:-1]) / 10.0
                            if lon_str[-1] == 'W':
                                lon_val = -lon_val
                            lon = lon_val
                        logging.info(f"  Peak intensity: {best_vmax} kt at {dt.strftime('%Y-%m-%d %H:%M')}Z ({lat:.1f}, {lon:.1f})")
            except Exception as e:
                logging.warning(f"Failed to fetch archive for {atcf_id}: {e}")

    use_fulldisk_resolution = False
    if fulldisk:
        lat = 0.0
        lon = _sat_subpoint_lon(sat_source)
        half_deg = 90.0
        half_lon = half_lat = half_deg
        bounds = (lat, lon, half_deg)
        use_fulldisk_resolution = True
        logging.info(f"  Fulldisk mode: native GEOS, no 2 km eqc resample (center lon={lon})")

    explicit_bounds = all(storm.get(k) is not None for k in ("lon_min", "lon_max", "lat_min", "lat_max"))
    if use_fulldisk_resolution:
        pass
    elif explicit_bounds:
        lon_min, lon_max = storm["lon_min"], storm["lon_max"]
        lat_min, lat_max = storm["lat_min"], storm["lat_max"]
        half_lon = (lon_max - lon_min) / 2.0
        half_lat = (lat_max - lat_min) / 2.0
        half_deg = max(half_lon, half_lat)
        bounds = (lat, lon, half_deg)
        out_h = round(output_width * (half_lat / half_lon))
    else:
        R_earth = 6371.0
        lat_deg_per_km = 1.0 / 111.32
        lon_deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat)))
        half_km = crop_km / 2.0
        half_deg = max(half_km * lat_deg_per_km, half_km * lon_deg_per_km)
        half_lon = half_lat = half_deg
        bounds = (lat, lon, half_deg)
        out_h = output_width

    if sat_source == "gk2a":
        segments = []
        logging.info("  GK2A full-disk mode (no segments)")
    elif sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        segments = []
        logging.info("  GOES full-disk mode (no segments)")
    elif sat_source == "mtg":
        segments = []
        logging.info("  MTG full-disk mode (no segments)")
    elif sat_source in ("mtsat", "mtsat2", "mtsat1"):
        segments = []
        logging.info("  MTSAT full-disk mode (no segments)")
    elif use_target:
        segments = ["R3"]
        logging.info(f"  Target area mode: using dynamic R3xx observation sequences (not fixed segments)")
    else:
        segments = get_required_segments(bounds, buffer=False)
        logging.info(f"  Required segments: {segments}")

    bands = []
    for p in products:
        for b in PRODUCT_BANDS.get(p, [13]):
            if b not in bands:
                bands.append(b)
    bands.sort()

    if sat_source == "gk2a" and "firetemp" in products and bands == [6, 7, 9]:
        bands = [5, 7, 9]

    time_slots = []
    if date_from or (time_from or time_to):
        if not date_from:
            today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
            date_from = today
            date_to = today
            logging.info(f"Using today's date for time range: {today}")
        time_slots = generate_time_slots(date_from, date_to, time_from, time_to)
        if not time_slots:
            logging.warning("No valid time slots generated from date/time range")
            return
    else:
        time_slots = [None]

    prefetch_dir = None
    prefetched = None
    if any(s is not None for s in time_slots) and sat_source in ("him", "gk2a"):
        if sat_source == "gk2a":
            prefetch_dir, prefetched = prefetch_all_slots_gk2a(bands, time_slots, download_workers)
        else:
            prefetch_sats = ("noaa-himawari9",) if use_target else ("noaa-himawari9", "noaa-himawari8")
            prefetch_dir, prefetched = prefetch_all_slots(
                prefetch_sats, bands, segments, time_slots, use_target,
                download_workers, decompress_workers)

    for dt_slot in time_slots:
        if dt_slot is not None:
            dt = dt_slot
            logging.info(f"  Processing time slot: {dt.strftime('%Y-%m-%d %H:%M')}Z")
        else:
            pass

        local_dat_map = None
        tmpdir = None
        sat_used = None
        sat_name = "HIMAWARI-9"
        requested_target_segment = None

        if dt is None and time_str:
            d = date_str if date_str else datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
            try:
                dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
                if use_target:
                    dt, requested_target_segment = resolve_target_scan(dt)
                    logging.info(f"  --time {time_str} -> target scan {requested_target_segment} at {dt.strftime('%H:%M')}Z slot")
                logging.info(f"  Using specified time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
            except ValueError:
                logging.error(f"Invalid --date {date_str} or --time {time_str}; use YYYYMMDD and HHMM")
                continue
                
        if track is not None and dt is not None:
            ilat, ilon = interpolate_track_position(track, dt)
            if ilat is not None and ilon is not None:
                lat, lon = ilat, ilon
                storm["latitude"], storm["longitude"] = lat, lon
                logging.info(f"  IBTrACS position @ {dt:%Y-%m-%d %H:%MZ}: ({lat:.2f}, {lon:.2f})")

        if sat_source == "him" and data_dir and os.path.exists(data_dir) and dt is not None:
            dt_str = dt.strftime("%Y%m%d_%H%M")
            found = {}
            complete = True
            for band in bands:
                band_files = []
                if use_target:
                    seg_filter = f"*{requested_target_segment}*" if requested_target_segment else "*R3*"
                    matches = glob.glob(os.path.join(data_dir, f"*_{dt_str}_B{band:02d}_{seg_filter}.DAT"))
                    if not matches:
                        complete = False
                        break
                    band_files.extend(matches)
                else:
                    for seg in segments:
                        matches = glob.glob(os.path.join(data_dir, f"*_{dt_str}_B{band:02d}_*{seg}*.DAT"))
                        if not matches:
                            complete = False
                            break
                        band_files.extend(matches)
                if not complete:
                    break
                found[band] = sorted(band_files)
            if complete and found:
                local_dat_map = found
                sat_used = "noaa-himawari9"
                logging.info(f"  Using pre-downloaded data from {data_dir} for {dt_str} (bands {sorted(found.keys())})")
            else:
                logging.warning(f"  Incomplete cache in data_dir for {dt_str} (need segments {segments}, bands {bands}); will download")

        if sat_used is None and prefetched is not None and dt is not None and dt in prefetched and prefetched[dt]:
            local_dat_map = prefetched[dt]
            sat_used = "noaa-himawari9"
            logging.info(f"  Using bulk-prefetched data for {dt.strftime('%Y-%m-%d %H:%M')}Z (bands {sorted(local_dat_map.keys())})")

        if sat_used is None and sat_source == "gk2a":
            logging.info("  Satellite: GK-2A (AMI, NOAA PDS)")
            if dt is None:
                if time_str:
                    d = date_str if date_str else datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
                    try:
                        dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
                    except ValueError:
                        logging.error(f"Invalid --date {date_str} or --time {time_str}; use YYYYMMDD and HHMM")
                        continue
                    logging.info(f"  Using specified time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                else:
                    dt = get_latest_available_dt_gk2a(bands)
                    if dt is None:
                        logging.warning(f"No complete GK2A dataset found for {storm_id} within search window.")
                        continue
                    logging.info(f"  Using time: {dt.strftime('%Y-%m-%d %H:%M')}Z")

            remote_map = discover_gk2a_files(dt, bands)
            if not remote_map:
                logging.warning(f"No GK2A remote files found for {storm_id}; skipping.")
                continue

            tmpdir = tempfile.mkdtemp(prefix="automata_gk2a_")
            local_dat_map = download_gk2a_files(remote_map, tmpdir, download_workers)
            if local_dat_map is None:
                logging.warning(f"GK2A download failed for {storm_id}; falling back to prior slot")
                shutil.rmtree(tmpdir, ignore_errors=True)
                dt -= datetime.timedelta(minutes=10)
                logging.info(f"  Fallback time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                remote_map = discover_gk2a_files(dt, bands)
                if not remote_map:
                    logging.warning(f"No fallback GK2A files found for {storm_id}; skipping.")
                    continue
                tmpdir = tempfile.mkdtemp(prefix="automata_gk2a_")
                local_dat_map = download_gk2a_files(remote_map, tmpdir, download_workers)
                if local_dat_map is None:
                    logging.warning(f"GK2A fallback download also failed for {storm_id}; skipping.")
                    continue

            missing_bands = [b for b in bands if GK2A_BAND_CHANNEL.get(b) and (b not in local_dat_map or not local_dat_map[b])]
            if missing_bands:
                logging.warning(f"Missing GK2A bands {missing_bands} for {storm_id}; falling back to prior slot")
                shutil.rmtree(tmpdir, ignore_errors=True)
                dt -= datetime.timedelta(minutes=10)
                logging.info(f"  Fallback time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                remote_map = discover_gk2a_files(dt, bands)
                if not remote_map:
                    logging.warning(f"No fallback GK2A files found for {storm_id}; skipping.")
                    continue
                tmpdir = tempfile.mkdtemp(prefix="automata_gk2a_")
                local_dat_map = download_gk2a_files(remote_map, tmpdir, download_workers)
                if local_dat_map is None:
                    logging.warning(f"GK2A fallback download also failed for {storm_id}; skipping.")
                    continue

            sat_used = "noaa-gk2a-pds"

        if sat_used is None:
            if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
                sat_source = _resolve_goes_source(sat_source, lon)
                satellites = _goes_candidate_buckets(sat_source)
            elif sat_source == "mtg":
                satellites = (MTG_COLLECTION,)
            elif sat_source in ("mtsat", "mtsat2"):
                satellites = ("mtsat", "mtsat1")
            elif sat_source == "mtsat1":
                satellites = ("mtsat1",)
            else:
                satellites = ("noaa-himawari9",) if use_target else ("noaa-himawari9", "noaa-himawari8")
            for slot_attempt in range(2):
                for sat in satellites:
                    logging.info(f"  Satellite: {sat}")
                    if dt is None:
                        if time_str:
                            d = date_str if date_str else datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
                            try:
                                dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
                            except ValueError:
                                logging.error(f"Invalid --date {date_str} or --time {time_str}; use YYYYMMDD and HHMM")
                                continue
                            if use_target:
                                dt, requested_target_segment = resolve_target_scan(dt)
                            logging.info(f"  Using specified time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                        else:
                            dt = _resolve_latest_dt(sat_source, sat, segments, bands, use_target=use_target)
                            if dt is None:
                                logging.warning(f"No complete dataset found for {storm_id} on {sat} within search window; trying next satellite.")
                                continue
                            logging.info(f"  Using time: {dt.strftime('%Y-%m-%d %H:%M')}Z")

                    remote_map = _discover_files(sat_source, sat, dt, bands, segments,
                                                 use_target=use_target,
                                                 target_segment=requested_target_segment,
                                                 center_lat=lat, center_lon=lon,
                                                 roi_deg=half_deg)
                    if not remote_map:
                        logging.warning(f"No remote files found for {storm_id} on {sat}; trying next satellite.")
                        continue

                    tmpdir = tempfile.mkdtemp(prefix="automata_")
                    local_dat_map = _download_files(sat_source, remote_map, tmpdir,
                                                    download_workers, decompress_workers)

                    if local_dat_map is None:
                        logging.warning(f"Download failed for {storm_id} on {sat}; falling back to prior slot")
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        dt -= datetime.timedelta(minutes=10)
                        logging.info(f"  Fallback time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                        remote_map = _discover_files(sat_source, sat, dt, bands, segments,
                                                     use_target=use_target,
                                                     target_segment=requested_target_segment)
                        if not remote_map:
                            logging.warning(f"No fallback files found for {storm_id} on {sat}; trying next satellite.")
                            continue
                        tmpdir = tempfile.mkdtemp(prefix="automata_")
                        local_dat_map = _download_files(sat_source, remote_map, tmpdir,
                                                        download_workers, decompress_workers)
                        if local_dat_map is None:
                            logging.warning(f"Fallback download also failed for {storm_id} on {sat}; trying next satellite.")
                            continue

                    missing_bands = [b for b in bands if b not in local_dat_map or not local_dat_map[b]]
                    if missing_bands:
                        logging.warning(f"Missing bands {missing_bands} for {storm_id} on {sat}; falling back to prior slot")
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        dt -= datetime.timedelta(minutes=10)
                        logging.info(f"  Fallback time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                        remote_map = _discover_files(sat_source, sat, dt, bands, segments,
                                                     use_target=use_target,
                                                     target_segment=requested_target_segment)
                        if not remote_map:
                            logging.warning(f"No fallback files found for {storm_id} on {sat}; trying next satellite.")
                            continue
                        tmpdir = tempfile.mkdtemp(prefix="automata_")
                        local_dat_map = _download_files(sat_source, remote_map, tmpdir,
                                                        download_workers, decompress_workers)
                        if local_dat_map is None:
                            logging.warning(f"Fallback download also failed for {storm_id} on {sat}; trying next satellite.")
                            continue

                    sat_used = sat
                    break
                if sat_used is not None:
                    break
                if slot_attempt == 0:
                    dt -= datetime.timedelta(minutes=10)
                    logging.info(f"  No data on any satellite at latest slot; falling back one slot: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                else:
                    logging.warning(f"No data on any candidate satellite for {storm_id} at latest or fallback slot; marking as missing.")
                    break

        if sat_used is None or local_dat_map is None:
            logging.warning(f"No usable satellite data found for {storm_id}; skipping.")
            continue
        sat = sat_used
        if sat in MTSAT_SAT_CONFIG:
            sat_source = sat
        if sat_source in ("mtsat", "mtsat2", "mtsat1"):
            sat_name = _mtsat_cfg(sat_source)["name"]
            sat_tag = "MTS2" if sat_source in ("mtsat", "mtsat2") else "MTS1"
        elif sat_source == "gk2a":
            sat_name = "GK-2A"
        elif sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
            _gnum = sat.replace("noaa-goes", "") if isinstance(sat, str) and "goes" in sat else sat_source.replace("goes", "")
            sat_name = f"GOES-{_gnum}"
            sat_tag = f"G{_gnum}"
        elif sat_source == "mtg":
            sat_name = "MTG-I1"
        else:
            sat_name = "HIMAWARI-8" if "himawari8" in sat else "HIMAWARI-9"

        lon_norm = (lon + 180) % 360 - 180
        sat_lon = _sat_subpoint_lon(sat_source)
        use_geos_area = _project_is_native_geos(project) and not use_target
        R = 6378137.0
        deg2rad = np.pi / 180.0
        half_m_x = half_lon * R * deg2rad
        half_m_y = half_lat * R * deg2rad
        x_min, x_max = -half_m_x, half_m_x
        y_min = (lat - half_lat) * R * deg2rad
        y_max = (lat + half_lat) * R * deg2rad

        if use_fulldisk_resolution:
            area_def = _standard_fulldisk_geos_area(sat_source)
            final_width = area_def.x_size
            final_height = area_def.y_size
            data_is_geos = True
            area_extent_meta = tuple(area_def.area_extent)
            use_geos_area = True
            logging.info(f"  Native full-disk GEOS {final_width}x{final_height} "
                         f"(IR already 2 km, no eqc resample)")
        else:
            if use_target:
                target_km = 1000.0
                target_pixels = output_width
                half_target_m = (target_km / 2.0) * 1000.0
                x_min_t = -half_target_m
                x_max_t = half_target_m
                y_min_t = -half_target_m
                y_max_t = half_target_m
                half_lon = half_lat = target_km / 111.32
                out_h = target_pixels
                final_width = target_pixels
                final_height = target_pixels
                x_min, y_min, x_max, y_max = x_min_t, y_min_t, x_max_t, y_max_t
            else:
                final_width = output_width
                final_height = out_h

            data_is_geos = False
            area_extent_meta = None
            if use_geos_area:
                try:
                    area_def = _geos_area_for_crop(
                        lat, lon, half_lon, half_lat, sat_lon, final_width, final_height)
                    data_is_geos = True
                    area_extent_meta = tuple(area_def.area_extent)
                    logging.info(f"  GEOS/native area: sat_lon={sat_lon} extent={area_extent_meta}")
                except Exception as e:
                    logging.warning(f"  GEOS area failed ({e}); falling back to eqc")
                    use_geos_area = False
            if not use_geos_area:
                proj_dict = {"proj": "eqc", "lon_0": lon_norm, "lat_ts": 0}
                area_def = AreaDefinition(
                    "storm_crop", "Storm Crop", "eqc", proj_dict,
                    final_width, final_height,
                    (x_min, y_min, x_max, y_max)
                )

        metadata = {
            'satellite_name': sat_name,
            'target_dt': dt,
            'center_lat': lat,
            'center_lon': lon,
            'crop_deg': max(half_lon, half_lat),
            'crop_lon': half_lon,
            'crop_lat': half_lat,
            'product': '',
            'storm_id': storm_id,
            'storm_name': storm_name,
            'winds': peak_vmax if peak_vmax is not None else storm.get('winds'),
            'pressure': peak_pressure if peak_pressure is not None else storm.get('pressure'),
            'grid': grid,
            'grid_thick': grid_thick,
            'grid_color': grid_color,
            'grid_style': grid_style,
            'no_coastlines': no_coastlines,
            'coastline_color': GK2A_DEFAULT_COASTLINE_COLOR if sat_source == "gk2a" else "#00FF00",
            'label': label,
            'par': par,
            'tcad': tcad,
            'tcid': tcid,
            'ico': ico,
            'invest': invest,
            'active_storms': active_storms,
            'crop_km': crop_km,
            'polygon': storm.get('polygon'),
            'floater': floater,
            'info': info,
            'radar_overlay': radar_overlay,
            'project': project,
            'sat_lon': sat_lon,
            'data_is_geos': data_is_geos,
            'area_extent': area_extent_meta,
            'fulldisk': bool(use_fulldisk_resolution),
        }

        cmap_infrared = mcolors.LinearSegmentedColormap.from_list("Infrared_Him", INFRARED_HIM_nodes)
        cmap_ott = mcolors.LinearSegmentedColormap.from_list("OTT", OTT_nodes)
        cmap_dvorak = mcolors.LinearSegmentedColormap.from_list("Dvorak", DVORAK_nodes)
        cmap_dvorak_pwards = _dvorak_cmap()
        cmap_sandwich = mcolors.ListedColormap(_SANDWICH_IR_LUT, name="sandwich_ir")
        cmap_dvorak_ir = mcolors.ListedColormap(_DVORAK_IR_LUT, name="dvorak_ir")
        cmap_ott2 = mcolors.LinearSegmentedColormap.from_list("OTT2", OTT2_nodes).reversed()

        want_mp4 = any(f.lower() == "mp4" for f in (export_formats or []))
        frame_tmpdir = tempfile.mkdtemp(prefix="automata_frames_")

        product_dirs = {}
        for product in products:
            product_dir = os.path.join(output_dir, product)
            os.makedirs(product_dir, exist_ok=True)
            product_dirs[product] = product_dir

        if use_target:
            grouped = _group_target_files_by_segment(local_dat_map)
            if not grouped:
                logging.warning("No R3xx segments found in target files; skipping.")
            if requested_target_segment and requested_target_segment not in grouped:
                logging.warning(f"  Requested {requested_target_segment} not present; available: {sorted(grouped.keys())}")
            frame_paths = {p: [] for p in products}
            for segment, seg_map in sorted(grouped.items()):
                if requested_target_segment and segment != requested_target_segment:
                    continue
                seg_dt = segment_observation_dt(dt, segment)
                native = _native_target_area(seg_map)
                if native is None:
                    logging.warning(f"  {segment}: could not read projection header; using static area")
                    seg_area = area_def
                    seg_lat = seg_lon = None
                    seg_extent = None
                else:
                    seg_area = _upscaled_target_area(native, output_width)
                    lons, lats = native.get_lonlats()
                    seg_lat = float(np.nanmean(lats))
                    seg_lon = float(np.nanmean(lons))
                    seg_extent = [float(np.nanmin(lons)), float(np.nanmax(lons)),
                                  float(np.nanmin(lats)), float(np.nanmax(lats))]
                meta_lat_bak = metadata.get('center_lat')
                meta_lon_bak = metadata.get('center_lon')
                meta_ext_bak = metadata.get('target_extent')
                if seg_lat is not None and seg_lon is not None:
                    metadata['center_lat'] = seg_lat
                    metadata['center_lon'] = seg_lon
                if seg_extent:
                    metadata['target_extent'] = seg_extent
                seg_ir, _, _, _ = process_vpsift_ahi_data(seg_map, None, seg_dt, "infrared", sat_source=sat_source)
                seg_ir = _upscale_to_width(seg_ir, output_width)
                seg_ir = np.nan_to_num(seg_ir, nan=300.0)
                seg_celsius = seg_ir - 273.15
                ir_norm_ott = np.clip((seg_ir - 173.15) / (323.15 - 173.15), 0.0, 1.0)
                ir_norm_infrared = np.power(np.clip((seg_ir - 173.15) / (323.15 - 173.15), 0.0, 1.0), 0.5)
                ir_norm_dvorak = seg_celsius
                logging.info(f"  Processing {segment} ({seg_dt.strftime('%H:%M:%S')}Z)"
                             + (f" center ({seg_lat:.2f}, {seg_lon:.2f})" if seg_lat is not None else ""))
                meta_backup = metadata.get('target_dt')
                metadata['target_dt'] = seg_dt
                ts = seg_dt.strftime('%Y%m%d_%H%M')
                with ThreadPoolExecutor(max_workers=len(products), thread_name_prefix='ProductGen') as executor:
                    futures = {}
                    for product in products:
                        metadata['product'] = product
                        out_name = f"{storm_id}_{ts}_{segment}_{product}_{sat_tag}.avif"
                        out_path = os.path.join(product_dirs[product], out_name)
                        future = executor.submit(
                            _generate_single_product,
                            product, seg_ir, seg_celsius, ir_norm_ott, ir_norm_infrared, ir_norm_dvorak,
                            metadata, out_path, logo_path,
                            cmap_infrared, cmap_ott, cmap_dvorak, cmap_dvorak_pwards,
                            cmap_sandwich, cmap_dvorak_ir, cmap_ott2,
                            export_formats,
                            local_dat_map=seg_map, seg_area=seg_area, target_dt=seg_dt,
                            use_target=True, output_width=output_width, sat_source=sat_source
                        )
                        futures[future] = product
                    for future in as_completed(futures):
                        product = futures[future]
                        try:
                            result = future.result()
                            print(result)
                        except Exception as e:
                            print(f"Failed {product}: {e}")
                if want_mp4:
                    for product in products:
                        frame_png = os.path.join(frame_tmpdir, f"{segment}_{product}.png")
                        metadata['product'] = product
                        _generate_single_product(
                            product, seg_ir, seg_celsius, ir_norm_ott, ir_norm_infrared, ir_norm_dvorak,
                            metadata, frame_png, logo_path,
                            cmap_infrared, cmap_ott, cmap_dvorak, cmap_dvorak_pwards,
                            cmap_sandwich, cmap_dvorak_ir, cmap_ott2,
                            ['png'],
                            local_dat_map=seg_map, seg_area=seg_area, target_dt=seg_dt,
                            use_target=True, output_width=output_width, sat_source=sat_source
                        )
                        frame_paths[product].append(frame_png)
                metadata['target_dt'] = meta_backup
                metadata['center_lat'] = meta_lat_bak
                metadata['center_lon'] = meta_lon_bak
                metadata['target_extent'] = meta_ext_bak

            if want_mp4:
                for product in products:
                    frames = sorted(frame_paths[product])
                    if len(frames) < 1:
                        continue
                    mp4_name = f"{storm_id}_{dt.strftime('%Y%m%d_%H%M')}_{product}_{sat_tag}.mp4"
                    mp4_path = os.path.join(product_dirs[product], mp4_name)
                    if create_mp4_from_frames(frames, mp4_path, fps=fps):
                        logging.info(f"  Created MP4: {mp4_path} ({len(frames)} frames)")
            shutil.rmtree(frame_tmpdir, ignore_errors=True)
            shutil.rmtree(tmpdir, ignore_errors=True) if tmpdir else None
            gc.collect()
            continue

        if use_fulldisk_resolution:
            logging.info("  Loading native B13 full disk (already 2 km GEOS, no resample)...")
            target_area_for_vpsift = None
        else:
            logging.info("  Loading and resampling B13 data (shared across all IR products)...")
            target_area_for_vpsift = None if use_target else area_def
        ir, _, _, _ = process_vpsift_ahi_data(local_dat_map, target_area_for_vpsift, dt, "infrared", sat_source=sat_source)
        if use_fulldisk_resolution and ir is not None:
            area_def = _area_with_shape(area_def, ir.shape)
            metadata['area_extent'] = tuple(area_def.area_extent)
            metadata['data_is_geos'] = True

        ir = np.nan_to_num(ir, nan=300.0)
        ir_celsius = ir - 273.15

        ir_norm_ott = np.clip((ir - 173.15) / (323.15 - 173.15), 0.0, 1.0)
        ir_norm_infrared = np.power(np.clip((ir - 173.15) / (323.15 - 173.15), 0.0, 1.0), 0.5)
        ir_norm_dvorak = ir_celsius

        with ThreadPoolExecutor(max_workers=len(products), thread_name_prefix='ProductGen') as executor:
            futures = {}
            for product in products:
                metadata['product'] = product
                product_dir = os.path.join(output_dir, product)
                os.makedirs(product_dir, exist_ok=True)
                out_name = f"{storm_id}_{dt.strftime('%Y%m%d_%H%M')}_{product}_{sat_tag}.avif"
                out_path = os.path.join(product_dir, out_name)

                future = executor.submit(
                    _generate_single_product,
                    product, ir, ir_celsius, ir_norm_ott, ir_norm_infrared, ir_norm_dvorak,
                    metadata, out_path, logo_path,
                    cmap_infrared, cmap_ott, cmap_dvorak, cmap_dvorak_pwards,
                    cmap_sandwich, cmap_dvorak_ir, cmap_ott2,
                    export_formats,
                    local_dat_map=local_dat_map, seg_area=area_def, target_dt=dt,
                    use_target=False, resample_type="nearest", sat_source=sat_source
                )
                futures[future] = product

            for future in as_completed(futures):
                product = futures[future]
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Failed {product}: {e}")

        shutil.rmtree(frame_tmpdir, ignore_errors=True)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()

        continue

    if prefetch_dir:
        shutil.rmtree(prefetch_dir, ignore_errors=True)



def process_storm(storm, crop_km, product, output_dir, output_width,
                  download_workers, decompress_workers, logo_path,
                  latest=False, date_str=None, time_str=None,
                  grid=False, grid_thick=0.4, grid_color="#00BFFF", grid_style="--",
                  no_coastlines=False, label=False,
                  par=False, tcad=False, tcid=False,
                  ico=False, invest=False, peak=False, active_storms=None,
                  data_dir=None, export_formats=None,
                   date_from=None, date_to=None, time_from=None, time_to=None,
                    use_target=False, floater=False, fps=4, nopng=False,
                    sat_source="him", fulldisk=False, track=None, info=False, radar_overlay=None,
                    project="flat", jpss_product=None, jpss_sat=None):
    _jpss_families = {"VIIRS-SDR", "VIIRS-EDR", "JPSS-GRAN", "VIIRSI-EDR",
                      "ATMS-SDR", "ATMS-TDR", "CRIS-SDR", "OMPS-SDR", "OMPS-RDR",
                      "JPSS-OZONE", "JPSS-NGRN", "JPSS-OCL2", "JPSS-SND"}
    if sat_source and (sat_source.upper() in {f.upper() for f in _jpss_families}
                       or sat_source.upper().startswith(("VIIRS", "JPSS", "ATMS", "OMPS", "CRIS"))):
        return process_jpss_storm(
            storm, crop_km, product, output_dir, output_width,
            family=sat_source, jpss_product=jpss_product, jpss_sat=jpss_sat,
            date_str=date_str, time_str=time_str,
            download_workers=download_workers, logo_path=logo_path,
            grid=grid, grid_thick=grid_thick, grid_color=grid_color, grid_style=grid_style,
            no_coastlines=no_coastlines, label=label,
            export_formats=export_formats, floater=floater, info=info, project=project)
    sat_tag = ("GK2A" if sat_source == "gk2a"
               else "GOES" if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19")
               else "MTG" if sat_source == "mtg"
               else "MTS2" if sat_source in ("mtsat", "mtsat2")
               else "MTS1" if sat_source == "mtsat1"
               else "HIM9")
    storm_id = storm.get("atcf_id") or storm.get("storm_name", "UNKNOWN")
    storm_name = storm.get("storm_name", "")
    lat = storm.get("latitude")
    lon = storm.get("longitude")
    if lat is None or lon is None:
        logging.warning(f"Storm {storm_id} has no lat/lon; skipping.")
        return

    if fulldisk:
        lat = 0.0
        lon = _sat_subpoint_lon(sat_source)
        half_deg = 90.0
        half_lon = half_lat = half_deg
        bounds = (lat, lon, half_deg)
        use_fulldisk_resolution = True
        logging.info(f"  Fulldisk mode: native GEOS, no 2 km eqc resample (center lon={lon})")
    else:
        use_fulldisk_resolution = False
        logging.info(f"Processing {storm_id} at ({lat:.1f}, {lon:.1f})")

    peak_vmax = None
    peak_pressure = None
    dt = None
    if peak and track is not None:
        _peak_fix = track_peak_fix(track)
        if _peak_fix is not None:
            dt = _peak_fix["dt"]
            lat, lon = _peak_fix["lat"], _peak_fix["lon"]
            peak_vmax = _peak_fix.get("wind")
            peak_pressure = _peak_fix.get("pres")
            logging.info(f"  IBTrACS peak intensity: {peak_vmax if peak_vmax is not None else 'n/a'} "
                         f"at {dt:%Y-%m-%d %H:%MZ} ({lat:.2f}, {lon:.2f})")
    elif peak and storm_id not in ('PHL', 'WPAC'):
        atcf_id = storm.get("atcf_id", "").strip()
        if atcf_id:
            url = f"https://api.knackwx.com/atcf/v2/track/archive?stormID={atcf_id}"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    best_vmax = -1
                    best_pres = None
                    best_dt_str = None
                    best_lat = None
                    best_lon = None
                    for line in r.text.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 9:
                            continue
                        try:
                            vmax = int(parts[8])
                            if vmax > best_vmax:
                                best_vmax = vmax
                                best_dt_str = parts[2]
                                try:
                                    best_pres = int(parts[9])
                                except (ValueError, IndexError):
                                    best_pres = None
                                if len(parts) >= 8:
                                    best_lat = parts[6]
                                    best_lon = parts[7]
                        except (ValueError, IndexError):
                            continue
                    if best_dt_str and best_lat and best_lon:
                        ymdh = best_dt_str.strip()
                        dt = datetime.datetime.strptime(ymdh, "%Y%m%d%H")
                        peak_vmax = best_vmax
                        peak_pressure = best_pres
                        lat_str = best_lat.strip()
                        if lat_str:
                            lat_val = float(lat_str[:-1]) / 10.0
                            if lat_str[-1] == 'S':
                                lat_val = -lat_val
                            lat = lat_val
                        lon_str = best_lon.strip()
                        if lon_str:
                            lon_val = float(lon_str[:-1]) / 10.0
                            if lon_str[-1] == 'W':
                                lon_val = -lon_val
                            lon = lon_val
                        logging.info(f"  Peak intensity: {best_vmax} kt at {dt.strftime('%Y-%m-%d %H:%M')}Z ({lat:.1f}, {lon:.1f})")
            except Exception as e:
                logging.warning(f"Failed to fetch archive for {atcf_id}: {e}")

    explicit_bounds = False
    if not use_fulldisk_resolution:
        explicit_bounds = all(storm.get(k) is not None for k in ("lon_min", "lon_max", "lat_min", "lat_max"))
        if explicit_bounds:
            lon_min, lon_max = storm["lon_min"], storm["lon_max"]
            lat_min, lat_max = storm["lat_min"], storm["lat_max"]
            half_lon = (lon_max - lon_min) / 2.0
            half_lat = (lat_max - lat_min) / 2.0
            half_deg = max(half_lon, half_lat)
            bounds = (lat, lon, half_deg)
            out_h = round(output_width * (half_lat / half_lon))
        else:
            R_earth = 6371.0
            lat_deg_per_km = 1.0 / 111.32
            lon_deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat)))
            half_km = crop_km / 2.0
            half_deg = max(half_km * lat_deg_per_km, half_km * lon_deg_per_km)
            half_lon = half_lat = half_deg
            bounds = (lat, lon, half_deg)
            out_h = output_width
            
    if sat_source == "gk2a":
        segments = []
        logging.info("  GK2A full-disk mode (no segments)")
    elif sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        segments = []
        logging.info("  GOES full-disk mode (no segments)")
    elif sat_source == "mtg":
        segments = []
        logging.info("  MTG full-disk mode (no segments)")
    elif sat_source in ("mtsat", "mtsat2", "mtsat1"):
        segments = []
        logging.info("  MTSAT full-disk mode (no segments)")
    elif use_target:
        segments = ["R3"]
        logging.info(f"  Target area mode: using dynamic R3xx observation sequences (not fixed segments)")
    else:
        segments = get_required_segments(bounds, buffer=False)
        logging.info(f"  Required segments: {segments}")

    if product == "sandwich":
        bands = [3, 13]
    elif product in ("true", "z1-true"):
        bands = [1, 3, 4, 13]
    elif product in ("dvorak", "z1-ir", "althea-ott2", "z1-dvorak", "ir", "infrared", "bt0"):
        bands = [13]
    elif product == "b03":
        bands = [3]
    elif product == "irv":
        bands = [3, 13]
    elif product in ("falsecolor", "falsecoloradv"):
        bands = [3, 13]
    elif product == "firetemp":
        bands = [6, 7, 9]
    elif product == "fire":
        bands = [7]
    elif product == "dayconv":
        bands = [3, 5, 7, 8, 10, 13]
    else:
        logging.error(f"Unsupported product: {product}")
        return

    time_slots = []
    if date_from or (time_from or time_to):
        if not date_from:
            today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
            date_from = today
            date_to = today
            logging.info(f"Using today's date for time range: {today}")
        time_slots = generate_time_slots(date_from, date_to, time_from, time_to)
        if not time_slots:
            logging.warning("No valid time slots generated from date/time range")
            return
    else:
        time_slots = [None]

    prefetch_dir = None
    prefetched = None
    if any(s is not None for s in time_slots) and sat_source in ("him", "gk2a"):
        if sat_source == "gk2a":
            prefetch_dir, prefetched = prefetch_all_slots_gk2a(bands, time_slots, download_workers)
        else:
            prefetch_sats = ("noaa-himawari9",) if use_target else ("noaa-himawari9", "noaa-himawari8")
            prefetch_dir, prefetched = prefetch_all_slots(
                prefetch_sats, bands, segments, time_slots, use_target,
                download_workers, decompress_workers)

    want_mp4 = any(f.lower() == "mp4" for f in (export_formats or []))
    all_mp4_frames = []
    all_mp4_png_bytes = []
    range_start_dt = None
    range_end_dt = None
    if not nopng:
        if date_from and date_to:
            frame_tmpdir = os.path.join(output_dir, f"temp_{date_from}_{date_to}")
        else:
            frame_tmpdir = tempfile.mkdtemp(prefix="automata_frames_")
        os.makedirs(frame_tmpdir, exist_ok=True)
    else:
        frame_tmpdir = None

    for dt_slot in time_slots:
        if dt_slot is not None:
            dt = dt_slot
            logging.info(f"  Processing time slot: {dt.strftime('%Y-%m-%d %H:%M')}Z")
        else:
            pass

        local_dat_map = None
        tmpdir = None
        sat_used = None
        sat_name = "HIMAWARI-9"
        requested_target_segment = None

        if dt is None and time_str:
            d = date_str if date_str else datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
            try:
                dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
                if use_target:
                    dt, requested_target_segment = resolve_target_scan(dt)
                    logging.info(f"  --time {time_str} -> target scan {requested_target_segment} at {dt.strftime('%H:%M')}Z slot")
                logging.info(f"  Using specified time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
            except ValueError:
                logging.error(f"Invalid --date {date_str} or --time {time_str}; use YYYYMMDD and HHMM")
                continue 

        if track is not None and dt is not None:
            ilat, ilon = interpolate_track_position(track, dt)
            if ilat is not None and ilon is not None:
                lat, lon = ilat, ilon
                storm["latitude"], storm["longitude"] = lat, lon
                logging.info(f"  IBTrACS position @ {dt:%Y-%m-%d %H:%MZ}: ({lat:.2f}, {lon:.2f})")

        if sat_source == "him" and data_dir and os.path.exists(data_dir) and dt is not None:
            dt_str = dt.strftime("%Y%m%d_%H%M")
            found = {}
            complete = True
            for band in bands:
                band_files = []
                if use_target:
                    seg_filter = f"*{requested_target_segment}*" if requested_target_segment else "*R3*"
                    matches = glob.glob(os.path.join(data_dir, f"*_{dt_str}_B{band:02d}_{seg_filter}.DAT"))
                    if not matches:
                        complete = False
                        break
                    band_files.extend(matches)
                else:
                    for seg in segments:
                        matches = glob.glob(os.path.join(data_dir, f"*_{dt_str}_B{band:02d}_*{seg}*.DAT"))
                        if not matches:
                            complete = False
                            break
                        band_files.extend(matches)
                if not complete:
                    break
                found[band] = sorted(band_files)
            if complete and found:
                local_dat_map = found
                sat_used = "noaa-himawari9"
                logging.info(f"  Using pre-downloaded data from {data_dir} for {dt_str} (bands {sorted(found.keys())})")
            else:
                logging.warning(f"  Incomplete cache in data_dir for {dt_str} (need segments {segments}, bands {bands}); will download")

        if sat_used is None and prefetched is not None and dt is not None and dt in prefetched and prefetched[dt]:
            local_dat_map = prefetched[dt]
            sat_used = "noaa-himawari9"
            logging.info(f"  Using bulk-prefetched data for {dt.strftime('%Y-%m-%d %H:%M')}Z (bands {sorted(local_dat_map.keys())})")

        if sat_used is None and sat_source == "gk2a":
            logging.info("  Satellite: GK-2A (AMI, NOAA PDS)")
            if dt is None:
                if time_str:
                    d = date_str if date_str else datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
                    try:
                        dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
                    except ValueError:
                        logging.error(f"Invalid --date {date_str} or --time {time_str}; use YYYYMMDD and HHMM")
                        continue
                    logging.info(f"  Using specified time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                else:
                    dt = get_latest_available_dt_gk2a(bands)
                    if dt is None:
                        logging.warning(f"No complete GK2A dataset found for {storm_id} within search window.")
                        continue
                    logging.info(f"  Using time: {dt.strftime('%Y-%m-%d %H:%M')}Z")

            remote_map = discover_gk2a_files(dt, bands)
            if not remote_map:
                logging.warning(f"No GK2A remote files found for {storm_id}; skipping.")
                continue

            tmpdir = tempfile.mkdtemp(prefix="automata_gk2a_")
            local_dat_map = download_gk2a_files(remote_map, tmpdir, download_workers)
            if local_dat_map is None:
                logging.warning(f"GK2A download failed for {storm_id}; falling back to prior slot")
                shutil.rmtree(tmpdir, ignore_errors=True)
                dt -= datetime.timedelta(minutes=10)
                logging.info(f"  Fallback time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                remote_map = discover_gk2a_files(dt, bands)
                if not remote_map:
                    logging.warning(f"No fallback GK2A files found for {storm_id}; skipping.")
                    continue
                tmpdir = tempfile.mkdtemp(prefix="automata_gk2a_")
                local_dat_map = download_gk2a_files(remote_map, tmpdir, download_workers)
                if local_dat_map is None:
                    logging.warning(f"GK2A fallback download also failed for {storm_id}; skipping.")
                    continue

            missing_bands = [b for b in bands if GK2A_BAND_CHANNEL.get(b) and (b not in local_dat_map or not local_dat_map[b])]
            if missing_bands:
                logging.warning(f"Missing GK2A bands {missing_bands} for {storm_id}; falling back to prior slot")
                shutil.rmtree(tmpdir, ignore_errors=True)
                dt -= datetime.timedelta(minutes=10)
                logging.info(f"  Fallback time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                remote_map = discover_gk2a_files(dt, bands)
                if not remote_map:
                    logging.warning(f"No fallback GK2A files found for {storm_id}; skipping.")
                    continue
                tmpdir = tempfile.mkdtemp(prefix="automata_gk2a_")
                local_dat_map = download_gk2a_files(remote_map, tmpdir, download_workers)
                if local_dat_map is None:
                    logging.warning(f"GK2A fallback download also failed for {storm_id}; skipping.")
                    continue

            sat_used = "noaa-gk2a-pds"

        if sat_used is None:
            if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
                sat_source = _resolve_goes_source(sat_source, lon)
                satellites = _goes_candidate_buckets(sat_source)
            elif sat_source == "mtg":
                satellites = (MTG_COLLECTION,)
            elif sat_source in ("mtsat", "mtsat2"):
                satellites = ("mtsat", "mtsat1")
            elif sat_source == "mtsat1":
                satellites = ("mtsat1",)
            else:
                satellites = ("noaa-himawari9",) if use_target else ("noaa-himawari9", "noaa-himawari8")
            for slot_attempt in range(2):
                for sat in satellites:
                    logging.info(f"  Satellite: {sat}")
                    if dt is None:
                        if time_str:
                            d = date_str if date_str else datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
                            try:
                                dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
                            except ValueError:
                                logging.error(f"Invalid --date {date_str} or --time {time_str}; use YYYYMMDD and HHMM")
                                continue
                            if use_target:
                                dt, requested_target_segment = resolve_target_scan(dt)
                            logging.info(f"  Using specified time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                        else:
                            dt = _resolve_latest_dt(sat_source, sat, segments, bands, use_target=use_target)
                            if dt is None:
                                logging.warning(f"No complete dataset found for {storm_id} on {sat} within search window; trying next satellite.")
                                continue
                            logging.info(f"  Using time: {dt.strftime('%Y-%m-%d %H:%M')}Z")

                    remote_map = _discover_files(sat_source, sat, dt, bands, segments,
                                                 use_target=use_target,
                                                 target_segment=requested_target_segment,
                                                 center_lat=lat, center_lon=lon,
                                                 roi_deg=half_deg)
                    if not remote_map:
                        logging.warning(f"No remote files found for {storm_id} on {sat}; trying next satellite.")
                        continue

                    tmpdir = tempfile.mkdtemp(prefix="automata_")
                    local_dat_map = _download_files(sat_source, remote_map, tmpdir,
                                                    download_workers, decompress_workers)

                    if local_dat_map is None:
                        logging.warning(f"Download failed for {storm_id} on {sat}; falling back to prior slot")
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        dt -= datetime.timedelta(minutes=10)
                        logging.info(f"  Fallback time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                        remote_map = _discover_files(sat_source, sat, dt, bands, segments,
                                                     use_target=use_target,
                                                     target_segment=requested_target_segment)
                        if not remote_map:
                            logging.warning(f"No fallback files found for {storm_id} on {sat}; trying next satellite.")
                            continue
                        tmpdir = tempfile.mkdtemp(prefix="automata_")
                        local_dat_map = _download_files(sat_source, remote_map, tmpdir,
                                                        download_workers, decompress_workers)
                        if local_dat_map is None:
                            logging.warning(f"Fallback download also failed for {storm_id} on {sat}; trying next satellite.")
                            continue

                    missing_bands = [b for b in bands if b not in local_dat_map or not local_dat_map[b]]
                    if missing_bands:
                        logging.warning(f"Missing bands {missing_bands} for {storm_id} on {sat}; falling back to prior slot")
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        dt -= datetime.timedelta(minutes=10)
                        logging.info(f"  Fallback time: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                        remote_map = _discover_files(sat_source, sat, dt, bands, segments,
                                                     use_target=use_target,
                                                     target_segment=requested_target_segment)
                        if not remote_map:
                            logging.warning(f"No fallback files found for {storm_id} on {sat}; trying next satellite.")
                            continue
                        tmpdir = tempfile.mkdtemp(prefix="automata_")
                        local_dat_map = _download_files(sat_source, remote_map, tmpdir,
                                                        download_workers, decompress_workers)
                        if local_dat_map is None:
                            logging.warning(f"Fallback download also failed for {storm_id} on {sat}; trying next satellite.")
                            continue

                    sat_used = sat
                    break
                if sat_used is not None:
                    break
                if slot_attempt == 0:
                    dt -= datetime.timedelta(minutes=10)
                    logging.info(f"  No data on any satellite at latest slot; falling back one slot: {dt.strftime('%Y-%m-%d %H:%M')}Z")
                else:
                    logging.warning(f"No data on any candidate satellite for {storm_id} at latest or fallback slot; marking as missing.")
                    break

        if sat_used is None or local_dat_map is None:
            logging.warning(f"No usable satellite data found for {storm_id}; skipping.")
            continue 
        sat = sat_used
        if sat in MTSAT_SAT_CONFIG:
            sat_source = sat
        if sat_source in ("mtsat", "mtsat2", "mtsat1"):
            sat_name = _mtsat_cfg(sat_source)["name"]
            sat_tag = "MTS2" if sat_source in ("mtsat", "mtsat2") else "MTS1"
        elif sat_source == "gk2a":
            sat_name = "GK-2A"
        elif sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
            _gnum = sat.replace("noaa-goes", "") if isinstance(sat, str) and "goes" in sat else sat_source.replace("goes", "")
            sat_name = f"GOES-{_gnum}"
            sat_tag = f"G{_gnum}"
        elif sat_source == "mtg":
            sat_name = "MTG-I1"
        else:
            sat_name = "HIMAWARI-8" if "himawari8" in sat else "HIMAWARI-9"

        lon_norm = (lon + 180) % 360 - 180
        sat_lon = _sat_subpoint_lon(sat_source)
        use_geos_area = _project_is_native_geos(project) and not use_target
        R = 6378137.0
        deg2rad = np.pi / 180.0
        half_m_x = half_lon * R * deg2rad
        half_m_y = half_lat * R * deg2rad
        x_min, x_max = -half_m_x, half_m_x
        y_min = (lat - half_lat) * R * deg2rad
        y_max = (lat + half_lat) * R * deg2rad

        if use_fulldisk_resolution:
            area_def = _standard_fulldisk_geos_area(sat_source)
            final_width = area_def.x_size
            final_height = area_def.y_size
            data_is_geos = True
            area_extent_meta = tuple(area_def.area_extent)
            use_geos_area = True
            logging.info(f"  Native full-disk GEOS {final_width}x{final_height} "
                         f"(IR already 2 km, no eqc resample)")
        else:
            final_width = output_width
            final_height = out_h
        
            data_is_geos = False
            area_extent_meta = None
            if use_geos_area:
                try:
                    area_def = _geos_area_for_crop(
                        lat, lon, half_lon, half_lat, sat_lon, final_width, final_height)
                    data_is_geos = True
                    area_extent_meta = tuple(area_def.area_extent)
                    logging.info(f"  GEOS/native area: sat_lon={sat_lon} extent={area_extent_meta}")
                except Exception as e:
                    logging.warning(f"  GEOS area failed ({e}); falling back to eqc")
                    use_geos_area = False
            if not use_geos_area:
                proj_dict = {"proj": "eqc", "lon_0": lon_norm, "lat_ts": 0}
                area_def = AreaDefinition(
                    "storm_crop", "Storm Crop", "eqc", proj_dict,
                    final_width, final_height,
                    (x_min, y_min, x_max, y_max)
                )

        metadata = {
            'satellite_name': sat_name,
            'target_dt': dt,
            'center_lat': lat,
            'center_lon': lon,
            'crop_deg': max(half_lon, half_lat),
            'crop_lon': half_lon,
            'crop_lat': half_lat,
            'product': product,
            'storm_id': storm_id,
            'storm_name': storm_name,
            'winds': peak_vmax if peak_vmax is not None else storm.get('winds'),
            'pressure': peak_pressure if peak_pressure is not None else storm.get('pressure'),
            'grid': grid,
            'grid_thick': grid_thick,
            'grid_color': grid_color,
            'grid_style': grid_style,
            'no_coastlines': no_coastlines,
            'coastline_color': GK2A_DEFAULT_COASTLINE_COLOR if sat_source == "gk2a" else "#00FF00",
            'label': label,
            'par': par,
            'tcad': tcad,
            'tcid': tcid,
            'ico': ico,
            'invest': invest,
            'active_storms': active_storms,
            'crop_km': crop_km,
            'polygon': storm.get('polygon'),
            'floater': floater,
            'info': info,
            'radar_overlay': radar_overlay,
            'project': project,
            'sat_lon': sat_lon,
            'data_is_geos': data_is_geos,
            'area_extent': area_extent_meta,
            'fulldisk': bool(use_fulldisk_resolution),
        }

        plot_func = plot_floater_image if floater else plot_image

        target_area_for_vpsift = area_def
        resample_type = "bilinear" if use_target else "nearest"

        def _render_product(seg_map, frame_png_only=False, out_dir=None, seg_dt=None, seg_label="", frame_name=None, fmt=None, return_png=False):
            local_fmt = ['png'] if frame_png_only else (fmt if fmt is not None else export_formats)
            out_dir = out_dir or output_dir
            use_dt = seg_dt if seg_dt is not None else dt
            meta_dt_backup = metadata.get('target_dt')
            metadata['target_dt'] = use_dt
            ts = use_dt.strftime('%Y%m%d_%H%M')
            seg_tag = f"_{seg_label}" if seg_label else ""
            _last_plot = [None]

            def _plot(*args, **kwargs):
                if return_png:
                    kwargs['return_png'] = True
                _last_plot[0] = plot_func(*args, **kwargs)
                return _last_plot[0]
            seg_area = target_area_for_vpsift
            meta_lat_bak = metadata.get('center_lat')
            meta_lon_bak = metadata.get('center_lon')
            meta_ext_bak = metadata.get('target_extent')
            if use_target:
                native = _native_target_area(seg_map)
                if native is None:
                    seg_area = target_area_for_vpsift
                    seg_lat = seg_lon = None
                    seg_extent = None
                else:
                    seg_area = _upscaled_target_area(native, output_width)
                    lons, lats = native.get_lonlats()
                    seg_lat = float(np.nanmean(lats))
                    seg_lon = float(np.nanmean(lons))
                    seg_extent = [float(np.nanmin(lons)), float(np.nanmax(lons)),
                                  float(np.nanmin(lats)), float(np.nanmax(lats))]
                if seg_area is None:
                    seg_area = target_area_for_vpsift
                elif seg_lat is not None and seg_lon is not None:
                    metadata['center_lat'] = seg_lat
                    metadata['center_lon'] = seg_lon
                if seg_extent:
                    metadata['target_extent'] = seg_extent

            def _read(composite):
                if use_fulldisk_resolution:
                    res = process_vpsift_ahi_data(seg_map, None, use_dt, composite,
                                                    resample_type=resample_type, sat_source=sat_source)
                    native_arr = next((a for a in res if a is not None), None)
                    if native_arr is not None:
                        new_area = _area_with_shape(seg_area, native_arr.shape[:2])
                        if new_area is not None:
                            metadata['area_extent'] = tuple(new_area.area_extent)
                            metadata['data_is_geos'] = True
                            metadata['center_lat'] = 0.0
                            metadata['center_lon'] = sat_lon
                        return _align_result_to_area(res, new_area or seg_area)
                    return res
                if not use_target:
                    return process_vpsift_ahi_data(seg_map, seg_area, use_dt, composite,
                                                    resample_type=resample_type, sat_source=sat_source)
                res = process_vpsift_ahi_data(seg_map, None, use_dt, composite, sat_source=sat_source)
                return tuple(_upscale_to_width(a, output_width) if a is not None else None
                             for a in res)

            try:
                if product == "sandwich":
                    vis, ir, _, _ = _read("sandwich")
                    vis = np.nan_to_num(vis, nan=0.0)
                    if np.nanmax(vis) > 1.0:
                        vis = vis / 100.0
                    vis = np.clip(vis, 0.0, 1.0)
                    ir = np.nan_to_num(ir, nan=300.0)
                    from pyorbital.astronomy import sun_zenith_angle
                    lons, lats = seg_area.get_lonlats()
                    sza = sun_zenith_angle(use_dt, lons, lats)
                    cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
                    cos2_sza = np.clip(np.cos(np.radians(sza)), 0.40, 1.0)
                    path_sun = 0.8 / cos2_sza
                    path_sun_a = 1.0 / cos_sza
                    vis_bright = vis * path_sun_a
                    rayleigh_vis = 0.011 * path_sun
                    vis_corr = np.clip(vis_bright - rayleigh_vis, 0.0, 1.0)
                    day_weight = np.clip((90.0 - sza) / 5.0, 0.0, 1.0)
                    night_weight = 1.0 - day_weight
                    vis_day = vis_corr * day_weight
                    ir_norm = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
                    ir_layer = np.power(ir_norm, 1.5) * 2
                    r_final = vis_day + (ir_layer * night_weight)
                    g_final = vis_day + (ir_layer * night_weight)
                    b_final = vis_day + (ir_layer * night_weight)
                    saturation_factor = 1.33
                    luminance = 0.2989 * r_final + 0.5870 * g_final + 0.1140 * b_final
                    r_final = np.clip(luminance + saturation_factor * (r_final - luminance), 0.0, 1.0)
                    g_final = np.clip(luminance + saturation_factor * (g_final - luminance), 0.0, 1.0)
                    b_final = np.clip(luminance + saturation_factor * (b_final - luminance), 0.0, 1.0)
                    ir_rgb = _sandwich_ir_lookup(ir)
                    cold_mask = ir < 248.15
                    rgb = np.stack([r_final, g_final, b_final], axis=-1)
                    rgb = np.where(cold_mask[:, :, None], ir_rgb, rgb)
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
                    display_name = "Sandwich (PWARDS)"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_sandwich_{sat_tag}.avif"
                    _plot(rgb, os.path.join(out_dir, out_name), metadata, logo_path=logo_path, export_formats=local_fmt)

                elif product == "true":
                    r, g, b, _ = _read("true")
                    rgb = np.stack([r, g, b], axis=-1)
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
                    display_name = "True Color"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_true_{sat_tag}.avif"
                    _plot(rgb, os.path.join(out_dir, out_name), metadata, logo_path=logo_path, export_formats=local_fmt)

                elif product == "ir":
                    ir, _, _, _ = _read("infrared")
                    ir = np.nan_to_num(ir, nan=300.0)
                    ir_celsius = ir - 273.15
                    cmap = mcolors.ListedColormap(_SANDWICH_IR_LUT, name="sandwich_ir")
                    display_name = "BT (PWARDS)"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_bt_{sat_tag}.avif"
                    _plot(ir_celsius, os.path.join(out_dir, out_name), metadata,
                               cmap=cmap, vmin=-100, vmax=50, logo_path=logo_path, export_formats=local_fmt)

                elif product == "infrared":
                    ir, _, _, _ = _read("infrared")
                    ir = np.nan_to_num(ir, nan=300.0)
                    ir_celsius = ir - 273.15
                    cmap = mcolors.LinearSegmentedColormap.from_list("Infrared_Him", INFRARED_HIM_nodes)
                    display_name = "Infrared"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_infrared_{sat_tag}.avif"
                    _plot(ir_celsius, os.path.join(out_dir, out_name), metadata,
                               cmap=cmap, vmin=-100, vmax=50, logo_path=logo_path, export_formats=local_fmt)

                elif product == "z1-ir":
                    ir, _, _, _ = _read("infrared")
                    ir = np.nan_to_num(ir, nan=300.0)
                    ir_celsius = ir - 273.15
                    cmap = mcolors.LinearSegmentedColormap.from_list("OTT", OTT_nodes)
                    display_name = "Z1-IR"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_z1-ir_{sat_tag}.avif"
                    _plot(ir_celsius, os.path.join(out_dir, out_name), metadata,
                               cmap=cmap, vmin=-100, vmax=50, logo_path=logo_path, export_formats=local_fmt)

                elif product == "althea-ott2":
                    ir, _, _, _ = _read("infrared")
                    ir = np.nan_to_num(ir, nan=300.0)
                    ir_celsius = ir - 273.15
                    cmap = mcolors.LinearSegmentedColormap.from_list("OTT2", OTT2_nodes).reversed()
                    display_name = "ALTHEA-OTT2"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_althea-ott2_{sat_tag}.avif"
                    _plot(ir_celsius, os.path.join(out_dir, out_name), metadata,
                               cmap=cmap, vmin=-100, vmax=50, logo_path=logo_path, export_formats=local_fmt)

                elif product == "z1-true":
                    r, g, b, _ = _read("true")
                    rgb = np.stack([r, g, b], axis=-1)
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
                    display_name = "Z1-True Color"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_z1-true_{sat_tag}.avif"
                    _plot(rgb, os.path.join(out_dir, out_name), metadata, logo_path=logo_path, export_formats=local_fmt)

                elif product == "z1-dvorak":
                    ir, _, _, _ = _read("dvorak")
                    ir = np.nan_to_num(ir, nan=300.0)
                    ir_celsius = ir - 273.15
                    cmap = mcolors.LinearSegmentedColormap.from_list("Dvorak", DVORAK_nodes)
                    display_name = "Z1-DVORAK"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_z1-dvorak_{sat_tag}.avif"
                    _plot(ir_celsius, os.path.join(out_dir, out_name), metadata,
                               cmap=cmap, vmin=-100, vmax=50, logo_path=logo_path, export_formats=local_fmt)

                elif product == "dvorak":
                    ir, _, _, _ = _read("dvorak")
                    ir = np.nan_to_num(ir, nan=300.0)
                    ir_celsius = ir - 273.15
                    cmap = mcolors.LinearSegmentedColormap.from_list("Dvorak", DVORAK_nodes)
                    display_name = "DVORAK (PWARDS)"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_dvorak-pwards_{sat_tag}.avif"
                    _plot(ir_celsius, os.path.join(out_dir, out_name), metadata,
                               cmap=cmap, vmin=-100, vmax=50, logo_path=logo_path, export_formats=local_fmt)

                elif product == "bt0":
                    ir, _, _, _ = _read("infrared")
                    ir = np.nan_to_num(ir, nan=300.0)
                    ir_celsius = ir - 273.15
                    cmap = mcolors.ListedColormap(_DVORAK_IR_LUT, name="dvorak_ir")
                    display_name = "BT0 (PWARDS)"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_bt0-pwards_{sat_tag}.avif"
                    _plot(ir_celsius, os.path.join(out_dir, out_name), metadata,
                               cmap=cmap, vmin=-100, vmax=50, logo_path=logo_path, export_formats=local_fmt)

                elif product == "b03":
                    b03, _, _, _ = _read("b03")
                    b03 = np.nan_to_num(b03, nan=0.0)
                    b03_norm = np.clip(b03 / 100.0 if np.nanmax(b03) > 1.0 else b03, 0.0, 1.0)
                    display_name = "B03"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_b03_{sat_tag}.avif"
                    _plot(b03_norm, os.path.join(out_dir, out_name), metadata,
                               cmap='gray', vmin=0, vmax=1, logo_path=logo_path, export_formats=local_fmt)

                elif product == "irv":
                    irv, _, _, _ = _read("irv")
                    irv = np.nan_to_num(irv, nan=0.0)
                    display_name = "IRV"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_irv_{sat_tag}.avif"
                    _plot(irv, os.path.join(out_dir, out_name), metadata,
                               cmap='gray', vmin=0, vmax=1, logo_path=logo_path, export_formats=local_fmt)

                elif product == "falsecolor":
                    r, g, b, _ = _read("falsecolor")
                    rgb = _stack_rgb(r, g, b)
                    display_name = "False Color"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_falsecolor_{sat_tag}.avif"
                    _plot(rgb, os.path.join(out_dir, out_name), metadata, logo_path=logo_path, export_formats=local_fmt)

                elif product == "falsecoloradv":
                    r, g, b, _ = _read("falsecoloradv")
                    rgb = _stack_rgb(r, g, b)
                    display_name = "False Color Adv"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_falsecoloradv_{sat_tag}.avif"
                    _plot(rgb, os.path.join(out_dir, out_name), metadata, logo_path=logo_path, export_formats=local_fmt)

                elif product == "firetemp":
                    r, g, b, _ = _read("firetemp")
                    rgb = _stack_rgb(r, g, b)
                    display_name = "Fire Temp"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_firetemp_{sat_tag}.avif"
                    _plot(rgb, os.path.join(out_dir, out_name), metadata, logo_path=logo_path, export_formats=local_fmt)

                elif product == "fire":
                    b07, _, _, _ = _read("b07")
                    b07_celsius = b07 - 273.15
                    cmap_fire = mcolors.LinearSegmentedColormap.from_list("hotspot_SIR", hotspot_SIR_nodes)
                    display_name = "Fire (3.9um)"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_fire_{sat_tag}.avif"
                    _plot(b07_celsius, os.path.join(out_dir, out_name), metadata,
                           cmap=cmap_fire, vmin=273.15-273.15, vmax=353.15-273.15, logo_path=logo_path, export_formats=local_fmt)

                elif product == "dayconv":
                    r, g, b, _ = _read("dayconv")
                    rgb = _stack_rgb(r, g, b)
                    display_name = "Day Convection"
                    metadata['product'] = display_name
                    if frame_name:
                        out_name = frame_name
                    else:
                        out_name = f"{storm_id}_{ts}{seg_tag}_dayconv_{sat_tag}.avif"
                    _plot(rgb, os.path.join(out_dir, out_name), metadata, logo_path=logo_path, export_formats=local_fmt)

                return _last_plot[0] if return_png else os.path.join(out_dir, out_name)
            finally:
                metadata['target_dt'] = meta_dt_backup
                metadata['center_lat'] = meta_lat_bak
                metadata['center_lon'] = meta_lon_bak
                metadata['target_extent'] = meta_ext_bak

        if use_target:
            grouped = _group_target_files_by_segment(local_dat_map)
            if requested_target_segment and requested_target_segment not in grouped:
                logging.warning(f"  Requested {requested_target_segment} not present; available: {sorted(grouped.keys())}")
            for segment, seg_map in sorted(grouped.items()):
                if requested_target_segment and segment != requested_target_segment:
                    continue
                seg_dt = segment_observation_dt(dt, segment)
                logging.info(f"  Processing target segment {segment} ({seg_dt.strftime('%H:%M:%S')}Z)...")
                if want_mp4:
                    fmt = [f for f in export_formats if f != 'mp4']
                    png_bytes = _render_product(seg_map, seg_dt=seg_dt, seg_label=segment,
                                                fmt=fmt, return_png=True)
                    if nopng:
                        if png_bytes:
                            all_mp4_png_bytes.append(png_bytes)
                    else:
                        frame_tag = f"{dt.strftime('%H%M')}_{segment}"
                        frame_png = os.path.join(frame_tmpdir, f"{frame_tag}.png")
                        if png_bytes:
                            with open(frame_png, "wb") as f:
                                f.write(png_bytes)
                            all_mp4_frames.append(frame_png)
                    if range_start_dt is None or dt < range_start_dt:
                        range_start_dt = dt
                    if range_end_dt is None or dt > range_end_dt:
                        range_end_dt = dt
                else:
                    _render_product(seg_map, seg_dt=seg_dt, seg_label=segment)
        else:
            _render_product(local_dat_map)

        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()

        continue

    if want_mp4 and (all_mp4_png_bytes if nopng else all_mp4_frames):
        if range_start_dt is not None and range_end_dt is not None:
            mp4_name = (f"{storm_id}_{range_start_dt.strftime('%Y%m%d_%H%M')}-"
                        f"{range_end_dt.strftime('%H%M')}_{product}_{sat_tag}.mp4")
        else:
            mp4_name = f"{storm_id}_{product}_{sat_tag}.mp4"
        mp4_path = os.path.join(output_dir, mp4_name)
        if nopng:
            if create_mp4_from_png_bytes(all_mp4_png_bytes, mp4_path, fps=fps):
                logging.info(f"  Created MP4: {mp4_path} ({len(all_mp4_png_bytes)} frames)")
        else:
            if create_mp4_from_frames(all_mp4_frames, mp4_path, fps=fps):
                logging.info(f"  Created MP4: {mp4_path} ({len(all_mp4_frames)} frames)")

    if prefetch_dir:
        shutil.rmtree(prefetch_dir, ignore_errors=True)

NAMED_REGIONS = {
    "conus": {
        "storm_name": "CONUS", "satellite": "goes19",
        "latitude": 39.0, "longitude": -98.0,
        "lat_min": 20.0, "lat_max": 52.0,
        "lon_min": -130.0, "lon_max": -65.0,
    },
    "goes-full": {
        "storm_name": "GOES Full Disk", "satellite": "goes19",
        "latitude": 0.0, "longitude": -75.0,
        "lat_min": -60.0, "lat_max": 60.0,
        "lon_min": -140.0, "lon_max": -10.0,
    },
    "eastcoast": {
        "storm_name": "US East Coast", "satellite": "goes19",
        "latitude": 38.0, "longitude": -75.0,
        "lat_min": 24.0, "lat_max": 50.0,
        "lon_min": -85.0, "lon_max": -60.0,
    },
    "westcoast": {
        "storm_name": "US West Coast", "satellite": "goes18",
        "latitude": 38.0, "longitude": -122.0,
        "lat_min": 24.0, "lat_max": 50.0,
        "lon_min": -135.0, "lon_max": -110.0,
    },
    "gulf": {
        "storm_name": "Gulf of Mexico", "satellite": "goes19",
        "latitude": 28.0, "longitude": -90.0,
        "lat_min": 18.0, "lat_max": 33.0,
        "lon_min": -100.0, "lon_max": -78.0,
    },
    "caribbean": {
        "storm_name": "Caribbean", "satellite": "goes19",
        "latitude": 16.0, "longitude": -70.0,
        "lat_min": 5.0, "lat_max": 28.0,
        "lon_min": -90.0, "lon_max": -50.0,
    },
    "mexico": {
        "storm_name": "Mexico", "satellite": "goes19",
        "latitude": 20.0, "longitude": -100.0,
        "lat_min": 8.0, "lat_max": 32.0,
        "lon_min": -118.0, "lon_max": -82.0,
    },
    "southamerica": {
        "storm_name": "South America", "satellite": "goes19",
        "latitude": -15.0, "longitude": -60.0,
        "lat_min": -56.0, "lat_max": 24.0,
        "lon_min": -86.0, "lon_max": -34.0,
    },
    "atlantic": {
        "storm_name": "Atlantic", "satellite": "goes19",
        "latitude": 22.0, "longitude": -50.0,
        "lat_min": 0.0, "lat_max": 45.0,
        "lon_min": -80.0, "lon_max": -20.0,
    },
    "epac": {
        "storm_name": "Eastern Pacific", "satellite": "goes18",
        "latitude": 20.0, "longitude": -130.0,
        "lat_min": -5.0, "lat_max": 45.0,
        "lon_min": -160.0, "lon_max": -90.0,
    },
    "europe": {
        "storm_name": "Europe", "satellite": "mtg",
        "latitude": 52.0, "longitude": 15.0,
        "lat_min": 34.0, "lat_max": 70.0,
        "lon_min": -10.0, "lon_max": 40.0,
    },
    "mediterranean": {
        "storm_name": "Mediterranean", "satellite": "mtg",
        "latitude": 38.0, "longitude": 15.0,
        "lat_min": 28.0, "lat_max": 48.0,
        "lon_min": -10.0, "lon_max": 40.0,
    },
    "africa": {
        "storm_name": "Africa", "satellite": "mtg",
        "latitude": 5.0, "longitude": 20.0,
        "lat_min": -35.0, "lat_max": 45.0,
        "lon_min": -20.0, "lon_max": 60.0,
    },
    "middleeast": {
        "storm_name": "Middle East", "satellite": "mtg",
        "latitude": 25.0, "longitude": 45.0,
        "lat_min": 5.0, "lat_max": 45.0,
        "lon_min": 25.0, "lon_max": 70.0,
    },
    "northatlantic": {
        "storm_name": "North Atlantic", "satellite": "mtg",
        "latitude": 42.0, "longitude": -35.0,
        "lat_min": 20.0, "lat_max": 65.0,
        "lon_min": -70.0, "lon_max": 0.0,
    },
    "philippines": {
        "storm_name": "Philippines", "satellite": "him",
        "latitude": 12.0, "longitude": 125.0,
        "lat_min": -3.0, "lat_max": 27.0,
        "lon_min": 100.0, "lon_max": 150.0,
    },
    "westpac": {
        "storm_name": "WestPac", "satellite": "him",
        "latitude": 13.5, "longitude": 140.0,
        "lat_min": -3.0, "lat_max": 30.0,
        "lon_min": 100.0, "lon_max": 180.0,
    },
}


GARBIN_RADAR_BASE = "https://data.garbinwx.org/raw"
GARBIN_TIME_REFERENCE_URL = "https://data.garbinwx.org/latest/timeReference10.json"
GARBIN_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/152.0.0.0 Safari/537.36")
GARBIN_BROWSER_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://garbinwx.org",
    "referer": "https://garbinwx.org/",
    "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": GARBIN_USER_AGENT,
}
GARBIN_RADAR_BOUNDS = [115.41549141305251, 3.801613036809332,
                       129.51730887177652, 22.45850950564088]

HEX_COLORS_DBZ = [
    '#535353',  # 1 dBZ
    '#5b5b5b',  # 2
    '#606060',  # 3
    '#6e6e6e',  # 4
    '#797979',  # 5
    '#828282',  # 6
    '#8a8a8a',  # 7
    '#939393',  # 8
    '#9b9b9b',  # 9
    '#a1a1a1',  # 10
    '#aaaaaa',  # 11
    '#b9b9b9',  # 12
    '#c1c1c1',  # 13
    '#c8c8c8',  # 14
    '#cecece',  # 15
    '#00ff00',  # 16
    '#00f500',  # 17
    '#00e600',  # 18
    '#00dc00',  # 19
    '#00d200',  # 20
    '#00c800',  # 21
    '#00be00',  # 22
    '#00b400',  # 23
    '#00aa00',  # 24
    '#00a000',  # 25
    '#009600',  # 26
    '#32aa00',  # 27
    '#64be00',  # 28
    '#96d200',  # 29
    '#cdeb00',  # 30
    '#ffff00',  # 31
    '#fff500',  # 32
    '#ffe600',  # 33
    '#ffdc00',  # 34
    '#ffd200',  # 35
    '#ffc800',  # 36
    '#ffb900',  # 37
    '#ffaa00',  # 38
    '#ff9600',  # 39
    '#ff8700',  # 40
    '#ff7800',  # 41
    '#ff5f00',  # 42
    '#ff4600',  # 43
    '#ff3200',  # 44
    '#ff1900',  # 45
    '#ff0000',  # 46
    '#ff0000',  # 47
    '#e60000',  # 48
    '#dc0000',  # 49
    '#d20000',  # 50
    '#c80000',  # 51
    '#be0000',  # 52
    '#b40000',  # 53
    '#aa0000',  # 54
    '#a00000',  # 55
    '#960000',  # 56
    '#aa0032',  # 57
    '#be0064',  # 58
    '#d70096',  # 59
    '#eb00cd',  # 60
    '#ff00ff',  # 61
    '#eb00ff',  # 62
    '#d200ff',  # 63
    '#be00ff',  # 64
    '#aa00ff',  # 65
    '#9600ff',  # 66+ dBZ
]


def _garbin_dbz_cmap():
    return mcolors.ListedColormap(HEX_COLORS_DBZ, name="garbin_dbz")


def _load_dotenv():
    candidates = [".env", os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), ".env")]
    for env_path in candidates:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception as e:
            logging.warning(f"Failed to load .env from {env_path}: {e}")
        return


def _garbin_identity():
    gid = (
        os.environ.get("GARBINWXID")
        or os.environ.get("GARBINWX_ID")
        or os.environ.get("garbinwxid")
    )
    ua = (
        os.environ.get("GARBINWXUSER")
        or os.environ.get("GARBINWX_USER")
        or GARBIN_USER_AGENT
    )
    return (gid.strip() if gid else None), (ua.strip() if ua else GARBIN_USER_AGENT)

def _garbin_time_reference():
    url = GARBIN_TIME_REFERENCE_URL
    try:
        resp = _download_session.get(url, headers=GARBIN_BROWSER_HEADERS, timeout=30)
        if resp.status_code == 200:
            refs = resp.json().get("timeReference", [])
            return [r[:-4] for r in refs if isinstance(r, str) and r.endswith(".png")]
    except Exception as e:
        logging.warning(f"Could not fetch GarbinWx time reference: {e}")
    return []


def _fetch_garbin_radar(radar_type, ts, gid, output_dir):
    _, user_agent = _garbin_identity()
    url = f"{GARBIN_RADAR_BASE}/{radar_type}-{ts}.png"
    headers = {"garbinwxid": gid, "user-agent": user_agent}
    out_path = os.path.join(output_dir, f"radar_{radar_type}_{ts}.png")
    try:
        resp = _download_session.get(url, headers=headers, stream=True, timeout=30)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(1024):
                    if chunk:
                        f.write(chunk)
            logging.info(f"Saved GarbinWx radar: {out_path}")
            return True
        logging.error(f"Failed to fetch GarbinWx radar {ts}: status {resp.status_code}")
        return False
    except Exception as e:
        logging.error(f"Failed to fetch GarbinWx radar {ts}: {e}")
        return False


def process_garbin_radar(output_dir, radar_type="DBZ", date_str=None, time_str=None):
    gid, _ua = _garbin_identity()
    if not gid:
        logging.error("--garbinradar requires a garbinwxid. Set GARBINWX_ID in a .env file "
                      "(e.g. GARBINWX_ID=YOUR-ID).")
        return False

    radar_type = (radar_type or "DBZ").upper()

    if date_str and time_str:
        try:
            ts = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M").strftime("%Y%m%d%H%M")
        except ValueError:
            logging.error(f"Invalid --date/--time for --garbinradar: {date_str} {time_str}")
            return False
        return _fetch_garbin_radar(radar_type, ts, gid, output_dir)

    refs = _garbin_time_reference()
    if not refs:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
        now = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
        refs = [now.strftime("%Y%m%d%H%M")]
    max_attempts = min(len(refs), 18)
    for ts in refs[:max_attempts]:
        if _fetch_garbin_radar(radar_type, ts, gid, output_dir):
            return True
    return False


def _garbin_radar_bytes(radar_type, date_str, time_str, gid):
    radar_type = (radar_type or "DBZ").upper()
    ts_list = []
    if radar_type in ("RAIN", "RAINRATE"):
        radar_type = "RR"
    if time_str:
        if date_str:
            raw = date_str + time_str
        else:
            raw = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d") + time_str
        try:
            dt_utc = datetime.datetime.strptime(raw, "%Y%m%d%H%M").replace(tzinfo=datetime.timezone.utc)
            dt_pht = dt_utc + datetime.timedelta(hours=8)
            ts_list = [dt_pht.strftime("%Y%m%d%H%M")]
        except ValueError:
            logging.error(f"Invalid --date/--time for radar: date={date_str} time={time_str}")
            return None, None
    else:
        ts_list = _garbin_time_reference()
        if not ts_list:
            now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
            now = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
            ts_list = [now.strftime("%Y%m%d%H%M")]

    _gid, user_agent = _garbin_identity()
    headers = {"garbinwxid": gid, "user-agent": user_agent}
    for ts in ts_list[:18]:
        url = f"{GARBIN_RADAR_BASE}/{radar_type}-{ts}.png"
        try:
            resp = _download_session.get(url, headers=headers, stream=True, timeout=30)
            if resp.status_code == 200:
                return resp.content, ts[:12]
            logging.warning(f"GarbinWx radar {ts}: status {resp.status_code}")
        except Exception as e:
            logging.warning(f"GarbinWx radar {ts}: {e}")
    return None, None


def _garbin_radar_overlay(radar_type, date_str, time_str):
    gid, _ua = _garbin_identity()
    if not gid:
        logging.error("--garbinradar overlay requires a garbinwxid. Set GARBINWX_ID in a .env file.")
        return None
    png_bytes, ts = _garbin_radar_bytes(radar_type, date_str, time_str, gid)
    if png_bytes is None:
        logging.error("No GarbinWx radar composite available for overlay.")
        return None
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        logging.error(f"Failed to decode radar composite: {e}")
        return None
    arr = np.asarray(img).astype(np.float32) / 255.0
    radar_type = (radar_type or "DBZ").upper()
    logging.info(f"GarbinWx radar overlay fetched (timestamp: {ts})")
    return {
        "rgb": arr,
        "bounds": list(GARBIN_RADAR_BOUNDS),
        "ts": ts,
        "source": "GarbinWx",
        "type": radar_type,
    }


def process_garbin_radar_viewer(storm, output_dir, output_width, radar_type="DBZ",
                                date_str=None, time_str=None, crop_km=1000,
                                info=False, logo_path=None,
                                grid=False, grid_thick=0.4, grid_color="#00BFFF", grid_style="--",
                                no_coastlines=False, label=False, export_formats=None,
                                floater=False):
    gid, _ua = _garbin_identity()
    if not gid:
        logging.error("--garbinradar requires a garbinwxid. Set GARBINWX_ID in a .env file "
                      "(e.g. GARBINWX_ID=YOUR-ID).")
        return False

    png_bytes, ts = _garbin_radar_bytes(radar_type, date_str, time_str, gid)
    if png_bytes is None:
        logging.error("No GarbinWx radar composite available.")
        return False

    try:
        radar_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        logging.error(f"Failed to decode radar composite: {e}")
        return False
    arr = np.asarray(radar_img).astype(np.float32) / 255.0
    if arr.shape[-1] == 4:
        alpha = arr[..., 3:4]
        radar_rgb = arr[..., :3] * alpha
    else:
        radar_rgb = arr[..., :3]

    lat = storm.get("latitude")
    lon = storm.get("longitude")
    if lat is None or lon is None:
        logging.warning("Radar viewer: storm has no lat/lon; using full composite bounds.")
        lat = (GARBIN_RADAR_BOUNDS[3] + GARBIN_RADAR_BOUNDS[1]) / 2.0
        lon = (GARBIN_RADAR_BOUNDS[2] + GARBIN_RADAR_BOUNDS[0]) / 2.0
    storm_id = storm.get("atcf_id") or storm.get("storm_name", "RADAR").upper()
    storm_name = storm.get("storm_name", "")

    explicit = all(storm.get(k) is not None for k in ("lon_min", "lon_max", "lat_min", "lat_max"))
    if explicit:
        half_lon = (storm["lon_max"] - storm["lon_min"]) / 2.0
        half_lat = (storm["lat_max"] - storm["lat_min"]) / 2.0
        out_h = round(output_width * (half_lat / half_lon))
    else:
        lat_deg_per_km = 1.0 / 111.32
        lon_deg_per_km = 1.0 / (111.32 * max(np.cos(np.radians(lat)), 0.05))
        half_km = crop_km / 2.0
        half_deg = max(half_km * lat_deg_per_km, half_km * lon_deg_per_km)
        half_lon = half_lat = half_deg
        out_h = output_width

    ts_dt = None
    target_dt = None
    if ts:
        try:
            ts_dt = datetime.datetime.strptime(ts, "%Y%m%d%H%M")
            target_dt = ts_dt - datetime.timedelta(hours=8)
        except ValueError:
            pass

    if floater:
        radar_min_lon, radar_min_lat, radar_max_lon, radar_max_lat = GARBIN_RADAR_BOUNDS
        target_lon_min = lon - half_lon
        target_lon_max = lon + half_lon
        target_lat_min = lat - half_lat
        target_lat_max = lat + half_lat

        height, width = radar_rgb.shape[0], radar_rgb.shape[1]

        x0 = int(round((target_lon_min - radar_min_lon) / (radar_max_lon - radar_min_lon) * (width - 1)))
        x1 = int(round((target_lon_max - radar_min_lon) / (radar_max_lon - radar_min_lon) * (width - 1))) + 1
        y0 = int(round((radar_max_lat - target_lat_max) / (radar_max_lat - radar_min_lat) * (height - 1)))
        y1 = int(round((radar_max_lat - target_lat_min) / (radar_max_lat - radar_min_lat) * (height - 1))) + 1

        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))

        if x1 <= x0 or y1 <= y0:
            logging.error("No overlap between radar and target bounds.")
            return False

        radar_cropped = radar_rgb[y0:y1, x0:x1]

        metadata = {
            'satellite_name': 'GARBINWX',
            'target_dt': target_dt,
            'center_lat': lat,
            'center_lon': lon,
            'crop_deg': max(half_lon, half_lat),
            'crop_lon': half_lon,
            'crop_lat': half_lat,
            'product': f'RADAR {radar_type}',
            'storm_id': storm_id,
            'storm_name': storm_name,
            'winds': None,
            'pressure': None,
            'grid': grid,
            'grid_thick': grid_thick,
            'grid_color': grid_color,
            'grid_style': grid_style,
            'no_coastlines': no_coastlines,
            'label': label,
            'par': False, 'tcad': False, 'tcid': False,
            'ico': False, 'invest': False,
            'active_storms': None,
            'crop_km': crop_km,
            'polygon': storm.get('polygon'),
            'floater': True,
            'info': info,
            'target_extent': [target_lon_min, target_lon_max, target_lat_min, target_lat_max],
        }

        ts_part = ts if ts else "now"
        safe_id = storm_id.replace('/', '_').replace(' ', '_')
        base_name = f"{safe_id}_{ts_part}_radar-{radar_type}_floater"
        base_path = os.path.join(output_dir, base_name)

        plot_floater_image(radar_cropped, base_path, metadata,
                           cmap=None, vmin=None, vmax=None,
                           logo_path=logo_path, quiet=False,
                           export_formats=export_formats, return_png=False,
                           dbz_cmap=_garbin_dbz_cmap())
        return True

    fig = plt.figure(figsize=(10, 10), dpi=256, facecolor='black')
    proj = ccrs.PlateCarree()
    ax = fig.add_axes([0, 0, 1, 1], projection=proj, facecolor='black')

    min_lon, min_lat, max_lon, max_lat = GARBIN_RADAR_BOUNDS
    ax.imshow(radar_rgb, extent=[min_lon, max_lon, min_lat, max_lat],
              origin='upper', transform=proj, interpolation='nearest')

    if not no_coastlines:
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5,
                       edgecolor='#00FF00', alpha=1.0)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.3,
                       edgecolor='#00FF00', alpha=0.5)

    ax.set_extent([lon - half_lon, lon + half_lon, lat - half_lat, lat + half_lat], crs=proj)
    ax.axis('off')

    if grid or label:
        grid_step = 1 if (half_lon + half_lat) < 6 else 5
        gl = ax.gridlines(draw_labels=label, linewidth=grid_thick, color=grid_color,
                          alpha=0.6, linestyle=grid_style)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, grid_step))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, grid_step))
        gl.top_labels = False
        gl.right_labels = False

    metadata = {
        'satellite_name': 'GARBINWX',
        'target_dt': target_dt,
        'center_lat': lat,
        'center_lon': lon,
        'crop_deg': max(half_lon, half_lat),
        'crop_lon': half_lon,
        'crop_lat': half_lat,
        'product': f'RADAR {radar_type}',
        'storm_id': storm_id,
        'storm_name': storm_name,
        'winds': None,
        'pressure': None,
        'grid': grid,
        'grid_thick': grid_thick,
        'grid_color': grid_color,
        'grid_style': grid_style,
        'no_coastlines': no_coastlines,
        'label': label,
        'par': False, 'tcad': False, 'tcid': False,
        'ico': False, 'invest': False,
        'active_storms': None,
        'crop_km': crop_km,
        'polygon': storm.get('polygon'),
        'floater': False,
        'info': info,
    }
    add_modern_info(ax, metadata, logo_path)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=None, pad_inches=0)
    buf.seek(0)
    safe_id = storm_id.replace('/', '_').replace(' ', '_')
    base_name = f"{safe_id}_{ts or 'now'}_radar-{radar_type}"
    base_path = os.path.join(output_dir, base_name)
    with Image.open(buf) as out_img:
        out_img = out_img.convert('RGB')
        target_h = round(out_img.width * (half_lat / half_lon))
        if target_h != out_img.height:
            top = (out_img.height - target_h) // 2
            out_img = out_img.crop((0, top, out_img.width, top + target_h))
        w, h = out_img.size
        if w % 2 or h % 2:
            out_img = out_img.resize((w - (w % 2), h - (h % 2)))
        for fmt in (export_formats or ['avif']):
            fmt = fmt.lower().strip()
            out_fmt = f"{base_path}.{fmt}"
            try:
                if fmt == 'avif':
                    out_img.save(out_fmt, format='AVIF', quality=95, subsampling="4:4:4")
                elif fmt == 'png':
                    out_img.save(out_fmt, format='PNG', compress_level=1)
                elif fmt in ('jpg', 'jpeg'):
                    out_img.save(out_fmt, format='JPEG', quality=95, subsampling=0)
                elif fmt == 'webp':
                    out_img.save(out_fmt, format='WEBP', quality=95, method=6)
                else:
                    continue
                logging.info(f"Saved: {out_fmt}")
            except OSError as e:
                alt = f"{base_path}.png"
                logging.warning(f"Failed to save {fmt} ({e}); falling back to PNG: {alt}")
                try:
                    out_img.save(alt, format='PNG', compress_level=1)
                except OSError:
                    logging.error(f"PNG fallback also failed for {alt}")
    plt.close(fig)
    return True


PHRADAR_ORIGIN = "https://www.panahon.gov.ph"
PHRADAR_TOKEN_FALLBACK = "bH2qMl5ZJsRZEcgo32fk8VQlRN5X6K6eEGBVcOCm"
PHRADAR_BOUNDS = list(GARBIN_RADAR_BOUNDS)
PHRADAR_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
)



def _phradar_colorize_la(png_bytes, scale=None, radar_type="DBZ"):
    try:
        img = Image.open(io.BytesIO(png_bytes))
    except Exception as e:
        logging.error(f"Panahon: cannot open radar image: {e}")
        return None
    if img.mode not in ("LA", "L", "RGBA", "RGB"):
        img = img.convert("LA")
    arr = np.asarray(img)
    if arr.ndim == 2:
        L = arr.astype(np.float32)
        A = np.full(L.shape, 255, dtype=np.float32)
    elif arr.shape[-1] == 2:
        L = arr[..., 0].astype(np.float32)
        A = arr[..., 1].astype(np.float32)
    elif arr.shape[-1] >= 3:
        rgba = arr.astype(np.float32)
        if rgba.shape[-1] == 3:
            alpha = np.ones(rgba.shape[:2] + (1,), dtype=np.float32) * 255.0
            rgba = np.concatenate([rgba, alpha], axis=-1)
        return np.clip(rgba / 255.0, 0.0, 1.0)
    else:
        logging.error(f"Panahon: unexpected image array shape {arr.shape}")
        return None

    scale = scale or {}
    vmax = float(scale.get("max") or 80.0)
    vmin = float(scale.get("min") or 0.0)
    use_sqrt = bool(scale.get("sqrt"))
    r = L / 255.0
    if use_sqrt:
        dbz = (r * r) * vmax
    else:
        dbz = vmin + r * (vmax - vmin)

    colors = np.zeros((66, 4), dtype=np.float32)
    for i, hx in enumerate(HEX_COLORS_DBZ):
        hx = hx.lstrip("#")
        colors[i, 0] = int(hx[0:2], 16) / 255.0
        colors[i, 1] = int(hx[2:4], 16) / 255.0
        colors[i, 2] = int(hx[4:6], 16) / 255.0
        colors[i, 3] = 1.0
    idx = np.clip(np.floor(dbz).astype(np.int32) - 1, 0, 65)
    rgba = colors[idx].copy()
    alpha = np.clip(A / 255.0, 0.0, 1.0)
    alpha = np.where(dbz < 1.0, 0.0, alpha)
    rgba[..., 3] = alpha
    return rgba


def _phradar_hmac_sha256_hex(key: str, message: str) -> str:
    import hashlib
    import hmac as _hmac
    return _hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _phradar_session():
    import secrets
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": PHRADAR_USER_AGENT,
        "Accept": "*/*",
        "Referer": PHRADAR_ORIGIN + "/",
        "Origin": PHRADAR_ORIGIN,
    })
    home = sess.get(PHRADAR_ORIGIN + "/", timeout=30)
    home.raise_for_status()
    metas = {}
    for m in re.finditer(r"<meta[^>]+>", home.text, re.I):
        tag = m.group(0)
        name = re.search(r'name=["\']([^"\']+)["\']', tag, re.I)
        content = re.search(r'content=["\']([^"\']*)["\']', tag, re.I)
        if name and content:
            metas[name.group(1)] = content.group(1)
    csrf = metas.get("csrf-token")
    handle = metas.get("api-sig-handle")
    if not csrf or not handle:
        raise RuntimeError("Panahon: missing csrf-token / api-sig-handle meta tags")
    sig_url = f"{PHRADAR_ORIGIN}/api/v1/sig?token={csrf}"
    sr = sess.get(sig_url, headers={"X-Sig-Handle": handle, "X-Requested-With": "XMLHttpRequest"}, timeout=30)
    sr.raise_for_status()
    secret = (sr.json() or {}).get("secret")
    if not secret:
        raise RuntimeError("Panahon: /api/v1/sig returned no secret")
    sess._ph_csrf = csrf
    sess._ph_secret = secret

    def _signed_get(path, params=None, timeout=60):
        path_clean = path.strip("/")
        ts = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp()))
        nonce = secrets.token_hex(16)
        msg = f"GET\n{path_clean}\n{ts}\n{nonce}"
        headers = {
            "X-Ts": ts,
            "X-Nonce": nonce,
            "X-Sig": _phradar_hmac_sha256_hex(secret, msg),
            "X-Requested-With": "XMLHttpRequest",
        }
        url = f"{PHRADAR_ORIGIN}/{path_clean}"
        return sess.get(url, params=params or {}, headers=headers, timeout=timeout)

    sess.ph_get = _signed_get
    return sess


def _phradar_mode(radar_type):
    rt = (radar_type or "DBZ").upper()
    if rt in ("RR", "RAIN", "RAINRATE"):
        return "rain", "mosaic-rainrate"
    return "dbz", "mosaic-reflectivity"


def _phradar_timeline(sess, sublayer="mosaic-reflectivity"):
    token = sess._ph_csrf or PHRADAR_TOKEN_FALLBACK
    resp = sess.ph_get("api/v1/radar/timeline", {
        "token": token,
        "sublayer": sublayer,
    }, timeout=30)
    if resp.status_code != 200:
        logging.warning(f"Panahon timeline HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    try:
        payload = resp.json()
    except Exception as e:
        logging.warning(f"Panahon timeline JSON error: {e}")
        return None
    if not payload.get("success"):
        logging.warning(f"Panahon timeline unsuccessful: {payload}")
        return None
    return payload.get("data") or {}


def _phradar_pick_unix(data, date_str=None, time_str=None):
    timeline = list(data.get("timeline") or [])
    if not timeline:
        return None, None
    target = None
    if date_str and time_str:
        try:
            target = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M").replace(
                tzinfo=datetime.timezone.utc)
        except ValueError:
            target = None
    if target is not None:
        t_unix = int(target.timestamp())
        best = min(timeline, key=lambda e: abs(int(e.get("observed_at_unix") or 0) - t_unix))
        return int(best["observed_at_unix"]), best.get("observed_at")
    best = max(timeline, key=lambda e: int(e.get("observed_at_unix") or 0))
    return int(best["observed_at_unix"]), best.get("observed_at")


def _phradar_fetch_image_bytes(radar_type="DBZ", date_str=None, time_str=None, size=1536):
    mode, sublayer = _phradar_mode(radar_type)
    try:
        sess = _phradar_session()
    except Exception as e:
        logging.error(f"Panahon session bootstrap failed: {e}")
        return None, None, None, None
    data = _phradar_timeline(sess, sublayer=sublayer)
    if not data:
        return None, None, None, None
    bounds = data.get("bounds") or PHRADAR_BOUNDS
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4):
        bounds = PHRADAR_BOUNDS
    t_unix, observed_at = _phradar_pick_unix(data, date_str, time_str)
    if t_unix is None:
        logging.error("Panahon: no timeline frames available")
        return None, None, None, None
    token = sess._ph_csrf or PHRADAR_TOKEN_FALLBACK
    last_err = None
    for sz in (size, 1536, 1024, 512):
        try:
            resp = sess.ph_get("api/v1/radar-data-image", {
                "token": token,
                "t": t_unix,
                "mode": mode,
                "size": sz,
                "v": 5,
            }, timeout=90)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                logging.info(f"Panahon radar image: t={t_unix} ({observed_at}) mode={mode} size={sz} "
                             f"({len(resp.content) / 1024:.0f} KB)")
                scale = data.get("scale") or {"mode": mode, "max": 80, "sqrt": False, "unit": "dBZ"}
                return resp.content, t_unix, list(bounds), scale
            last_err = f"HTTP {resp.status_code} {resp.headers.get('content-type')} {resp.text[:120]}"
        except Exception as e:
            last_err = str(e)
    logging.error(f"Panahon radar image failed: {last_err}")
    return None, None, None, None


def _phradar_radar_bytes(radar_type, date_str, time_str):
    png, t_unix, bounds, scale = _phradar_fetch_image_bytes(radar_type, date_str, time_str)
    if png is None:
        return None, None, None, None
    try:
        ts = datetime.datetime.fromtimestamp(t_unix, tz=datetime.timezone.utc).strftime("%Y%m%d%H%M")
    except Exception:
        ts = str(t_unix)
    return png, ts, bounds, scale


def _phradar_radar_overlay(radar_type, date_str, time_str):
    png_bytes, ts, bounds, scale = _phradar_radar_bytes(radar_type, date_str, time_str)
    if png_bytes is None:
        logging.error("No Panahon radar composite available for overlay.")
        return None
    rgba = _phradar_colorize_la(png_bytes, scale=scale, radar_type=radar_type)
    if rgba is None:
        logging.error("Failed to colorize Panahon radar composite for overlay.")
        return None
    if not bounds or len(bounds) != 4:
        bounds = list(PHRADAR_BOUNDS)
    radar_type = (radar_type or "DBZ").upper()
    logging.info(f"Panahon radar overlay fetched (timestamp: {ts})")
    return {
        "rgb": rgba,
        "bounds": list(bounds),
        "ts": ts,
        "source": "PAGASA/Panahon",
        "type": radar_type,
    }


def process_phradar_viewer(storm, output_dir, output_width, radar_type="DBZ",
                           date_str=None, time_str=None, crop_km=1000,
                           info=False, logo_path=None,
                           grid=False, grid_thick=0.4, grid_color="#00BFFF", grid_style="--",
                           no_coastlines=False, label=False, export_formats=None,
                           floater=False):
    png_bytes, ts, bounds, scale = _phradar_radar_bytes(radar_type, date_str, time_str)
    if png_bytes is None:
        logging.error("No Panahon radar composite available.")
        return False
    if not bounds or len(bounds) != 4:
        bounds = list(PHRADAR_BOUNDS)
    rgba = _phradar_colorize_la(png_bytes, scale=scale, radar_type=radar_type)
    if rgba is None:
        logging.error("Failed to colorize Panahon radar composite.")
        return False
    alpha = rgba[..., 3:4]
    radar_rgb = rgba[..., :3] * alpha

    lat = storm.get("latitude")
    lon = storm.get("longitude")
    if lat is None or lon is None:
        logging.warning("Panahon radar: storm has no lat/lon; using full composite bounds.")
        lat = (bounds[3] + bounds[1]) / 2.0
        lon = (bounds[2] + bounds[0]) / 2.0
    storm_id = storm.get("atcf_id") or storm.get("storm_name", "PHRADAR").upper()
    storm_name = storm.get("storm_name", "")

    explicit = all(storm.get(k) is not None for k in ("lon_min", "lon_max", "lat_min", "lat_max"))
    if explicit:
        half_lon = (storm["lon_max"] - storm["lon_min"]) / 2.0
        half_lat = (storm["lat_max"] - storm["lat_min"]) / 2.0
    else:
        lat_deg_per_km = 1.0 / 111.32
        lon_deg_per_km = 1.0 / (111.32 * max(np.cos(np.radians(lat)), 0.05))
        half_km = crop_km / 2.0
        half_deg = max(half_km * lat_deg_per_km, half_km * lon_deg_per_km)
        half_lon = half_lat = half_deg

    ts_dt = None
    target_dt = None
    if ts:
        try:
            target_dt = datetime.datetime.strptime(ts, "%Y%m%d%H%M")
        except ValueError:
            pass

    radar_min_lon, radar_min_lat, radar_max_lon, radar_max_lat = bounds

    if floater:
        target_lon_min = lon - half_lon
        target_lon_max = lon + half_lon
        target_lat_min = lat - half_lat
        target_lat_max = lat + half_lat

        height, width = radar_rgb.shape[0], radar_rgb.shape[1]
        x0 = int(round((target_lon_min - radar_min_lon) / (radar_max_lon - radar_min_lon) * (width - 1)))
        x1 = int(round((target_lon_max - radar_min_lon) / (radar_max_lon - radar_min_lon) * (width - 1))) + 1
        y0 = int(round((radar_max_lat - target_lat_max) / (radar_max_lat - radar_min_lat) * (height - 1)))
        y1 = int(round((radar_max_lat - target_lat_min) / (radar_max_lat - radar_min_lat) * (height - 1))) + 1
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            logging.error("No overlap between Panahon radar and target bounds.")
            return False
        radar_cropped = radar_rgb[y0:y1, x0:x1]

        def _px_to_lon(px):
            return radar_min_lon + (px / max(width - 1, 1)) * (radar_max_lon - radar_min_lon)
        def _px_to_lat(py):
            return radar_max_lat - (py / max(height - 1, 1)) * (radar_max_lat - radar_min_lat)
        actual_lon_min = _px_to_lon(x0)
        actual_lon_max = _px_to_lon(x1 - 1 if x1 > x0 else x0)
        actual_lat_max = _px_to_lat(y0)
        actual_lat_min = _px_to_lat(y1 - 1 if y1 > y0 else y0)
        if actual_lon_max < actual_lon_min:
            actual_lon_min, actual_lon_max = actual_lon_max, actual_lon_min
        if actual_lat_max < actual_lat_min:
            actual_lat_min, actual_lat_max = actual_lat_max, actual_lat_min
        actual_half_lon = 0.5 * (actual_lon_max - actual_lon_min)
        actual_half_lat = 0.5 * (actual_lat_max - actual_lat_min)
        actual_lat = 0.5 * (actual_lat_min + actual_lat_max)
        actual_lon = 0.5 * (actual_lon_min + actual_lon_max)

        metadata = {
            'satellite_name': 'PAGASA/PANAHON',
            'target_dt': target_dt,
            'center_lat': actual_lat,
            'center_lon': actual_lon,
            'crop_deg': max(actual_half_lon, actual_half_lat),
            'crop_lon': actual_half_lon,
            'crop_lat': actual_half_lat,
            'product': f'PH RADAR {radar_type}',
            'storm_id': storm_id,
            'storm_name': storm_name,
            'winds': None,
            'pressure': None,
            'grid': grid,
            'grid_thick': grid_thick,
            'grid_color': grid_color,
            'grid_style': grid_style,
            'no_coastlines': no_coastlines,
            'label': label,
            'par': False, 'tcad': False, 'tcid': False,
            'ico': False, 'invest': False,
            'active_storms': None,
            'crop_km': crop_km,
            'polygon': storm.get('polygon'),
            'floater': True,
            'info': info,
            'target_extent': [actual_lon_min, actual_lon_max, actual_lat_min, actual_lat_max],
        }
        ts_part = ts if ts else "now"
        safe_id = storm_id.replace('/', '_').replace(' ', '_')
        base_name = f"{safe_id}_{ts_part}_phradar-{radar_type}_floater"
        base_path = os.path.join(output_dir, base_name)
        plot_floater_image(radar_cropped, base_path, metadata,
                           cmap=None, vmin=None, vmax=None,
                           logo_path=logo_path, quiet=False,
                           export_formats=export_formats, return_png=False,
                           dbz_cmap=_garbin_dbz_cmap())
        return True

    fig = plt.figure(figsize=(10, 10), dpi=256, facecolor='black')
    proj = ccrs.PlateCarree()
    ax = fig.add_axes([0, 0, 1, 1], projection=proj, facecolor='black')
    ax.imshow(radar_rgb, extent=[radar_min_lon, radar_max_lon, radar_min_lat, radar_max_lat],
              origin='upper', transform=proj, interpolation='nearest')
    if not no_coastlines:
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5,
                       edgecolor='#00FF00', alpha=1.0)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.3,
                       edgecolor='#00FF00', alpha=0.5)
    ax.set_extent([lon - half_lon, lon + half_lon, lat - half_lat, lat + half_lat], crs=proj)
    ax.axis('off')
    if grid or label:
        grid_step = 1 if (half_lon + half_lat) < 6 else 5
        gl = ax.gridlines(draw_labels=label, linewidth=grid_thick, color=grid_color,
                          alpha=0.6, linestyle=grid_style)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, grid_step))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, grid_step))
        gl.top_labels = False
        gl.right_labels = False
    metadata = {
        'satellite_name': 'PAGASA/PANAHON',
        'target_dt': target_dt,
        'center_lat': lat,
        'center_lon': lon,
        'crop_deg': max(half_lon, half_lat),
        'crop_lon': half_lon,
        'crop_lat': half_lat,
        'product': f'PH RADAR {radar_type}',
        'storm_id': storm_id,
        'storm_name': storm_name,
        'winds': None,
        'pressure': None,
        'grid': grid,
        'grid_thick': grid_thick,
        'grid_color': grid_color,
        'grid_style': grid_style,
        'no_coastlines': no_coastlines,
        'label': label,
        'par': False, 'tcad': False, 'tcid': False,
        'ico': False, 'invest': False,
        'active_storms': None,
        'crop_km': crop_km,
        'polygon': storm.get('polygon'),
        'floater': False,
        'info': info,
    }
    add_modern_info(ax, metadata, logo_path)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches=None, pad_inches=0)
    buf.seek(0)
    safe_id = storm_id.replace('/', '_').replace(' ', '_')
    base_name = f"{safe_id}_{ts or 'now'}_phradar-{radar_type}"
    base_path = os.path.join(output_dir, base_name)
    with Image.open(buf) as out_img:
        out_img = out_img.convert('RGB')
        for fmt in (export_formats or ['avif']):
            fmt = fmt.lower().strip()
            out_fmt = f"{base_path}.{fmt}"
            try:
                if os.path.exists(out_fmt):
                    try:
                        os.remove(out_fmt)
                    except OSError:
                        pass
                if fmt == 'avif':
                    out_img.save(out_fmt, format='AVIF', quality=95, subsampling="4:4:4")
                elif fmt == 'png':
                    out_img.save(out_fmt, format='PNG', compress_level=1)
                elif fmt in ('jpg', 'jpeg'):
                    out_img.save(out_fmt, format='JPEG', quality=95, subsampling=0)
                elif fmt == 'webp':
                    out_img.save(out_fmt, format='WEBP', quality=95, method=6)
                else:
                    continue
                logging.info(f"Saved: {out_fmt}")
            except OSError as e:
                alt = f"{base_path}.png"
                logging.warning(f"Failed to save {fmt} ({e}); falling back to PNG: {alt}")
                try:
                    out_img.save(alt, format='PNG', compress_level=1)
                except OSError:
                    logging.error(f"PNG fallback also failed for {alt}")
    plt.close(fig)
    return True


JPSS_CLASS_BASE = "https://data.class.noaa.gov/JPSS"

JPSS_FAMILY_DEFAULT_PRODUCTS = {
    "VIIRS-SDR": [
        "VIIRS-Moderate-Resolution-Band-15-SDR",
        "VIIRS-Imagery-Band-05-SDR",
        "VIIRS-Imagery-Band-01-SDR",
        "VIIRS-Moderate-Resolution-Band-05-SDR",
        "VIIRS-Moderate-Bands-SDR-Geo",
        "VIIRS-Day-Night-Band-SDR",
    ],
    "VIIRS-EDR": [
        "VIIRS-Cloud-Mask-EDR",
        "VIIRS-Surface-Reflectance-EDR",
    ],
    "JPSS-GRAN": [
        "VIIRS-Cloud-Mask-EDR",
        "VIIRS-Surface-Reflectance-EDR",
        "VIIRS-Daytime-Cloud-Optical-and-Microphysical-Properties-DCOMP-EDRs",
        "VIIRS-Aerosol-Optical-Depth-and-Aerosol-Particle-Size-EDRs",
        "VIIRS-Volcanic-Ash-Detection-and-Height-EDR",
    ],
    "VIIRSI-EDR": ["VIIRS-Imagery-EDR"],
}


def _jpss_list_dir(url):
    try:
        resp = _download_session.get(url if url.endswith("/") else url + "/", timeout=60)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logging.warning(f"JPSS list failed for {url}: {e}")
        return []
    entries = []
    for m in re.finditer(r'href="(/JPSS/[^"]+/)"', html):
        href = m.group(1)
        name = href.rstrip("/").split("/")[-1]
        if name and name not in (".", ".."):
            entries.append((name + "/", "https://data.class.noaa.gov" + href))
    for m in re.finditer(r'href="((?:/downloads)?/JPSS/[^"]+\.(?:tar|nc|h5|hdf5|xml)(?:\.gz)?)"', html, re.I):
        href = m.group(1)
        name = href.split("/")[-1]
        full = "https://data.class.noaa.gov" + href
        entries.append((name, full))
    if not entries:
        for m in re.finditer(r'href="([^"]+)"', html):
            href = m.group(1)
            if href.startswith("?") or href.startswith("#") or "javascript" in href:
                continue
            name = href.rstrip("/").split("/")[-1]
            if not name or name in (".", ".."):
                continue
            if href.startswith("http"):
                full = href
            elif href.startswith("/"):
                full = "https://data.class.noaa.gov" + href
            else:
                full = url.rstrip("/") + "/" + href
            if name.endswith("/") or href.endswith("/"):
                entries.append((name if name.endswith("/") else name + "/",
                                full if full.endswith("/") else full + "/"))
            else:
                entries.append((name, full))
    seen, out = set(), []
    for n, u in entries:
        if n not in seen:
            seen.add(n)
            out.append((n, u))
    return out


def jpss_list_available_dates():
    entries = _jpss_list_dir(JPSS_CLASS_BASE + "/")
    dates = []
    for name, _url in entries:
        m = re.match(r"^(\d{8})/?$", name.rstrip("/"))
        if m:
            dates.append(m.group(1))
    return sorted(dates)


def jpss_closest_date(target_date_str=None):
    dates = jpss_list_available_dates()
    if not dates:
        logging.error("JPSS: no date directories found on CLASS")
        return None
    if not target_date_str:
        return dates[-1]
    target_date_str = target_date_str.strip()[:8]
    if target_date_str in dates:
        return target_date_str
    try:
        target = datetime.datetime.strptime(target_date_str, "%Y%m%d").date()
    except ValueError:
        logging.warning(f"JPSS: invalid date {target_date_str}; using latest {dates[-1]}")
        return dates[-1]
    best = min(dates, key=lambda d: abs(
        (datetime.datetime.strptime(d, "%Y%m%d").date() - target).days))
    if best != target_date_str:
        logging.info(f"JPSS: requested {target_date_str} not present; closest available is {best}")
    return best


def jpss_list_families(date_str):
    entries = _jpss_list_dir(f"{JPSS_CLASS_BASE}/{date_str}/")
    out = []
    for n, _ in entries:
        if not n.endswith("/"):
            continue
        name = n.rstrip("/")
        if re.match(r"^\d{8}$", name):
            continue
        out.append(name)
    return out


def jpss_list_products(date_str, family):
    entries = _jpss_list_dir(f"{JPSS_CLASS_BASE}/{date_str}/{family}/")
    return [n.rstrip("/") for n, _ in entries if n.endswith("/")]


def jpss_list_sats(date_str, family, product):
    entries = _jpss_list_dir(f"{JPSS_CLASS_BASE}/{date_str}/{family}/{product}/")
    return [n.rstrip("/") for n, _ in entries if n.endswith("/")]


def jpss_list_tars(date_str, family, product, sat_id):
    entries = _jpss_list_dir(f"{JPSS_CLASS_BASE}/{date_str}/{family}/{product}/{sat_id}/")
    tars = []
    for name, url in entries:
        if name.lower().endswith(".tar") and "manifest" not in name.lower():
            tars.append((name, url))
    return tars


def _jpss_parse_granule_time_from_name(name):
    base = os.path.basename(name)
    m = re.search(r"_d(\d{8})_t(\d{6,7})", base)
    if m:
        d, t = m.group(1), m.group(2)[:6]
        try:
            return datetime.datetime.strptime(d + t, "%Y%m%d%H%M")
        except ValueError:
            pass
    m = re.search(r"_s(\d{14})", base)
    if m:
        s = m.group(1)[:12]
        try:
            return datetime.datetime.strptime(s, "%Y%m%d%H%M")
        except ValueError:
            pass
    m = re.search(r"_(\d{8})_", base)
    if m:
        try:
            return datetime.datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            pass
    return None


def jpss_download_tar(url, local_path, retries=3):
    thread_name = threading.current_thread().name
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"{thread_name}: downloading JPSS TAR {os.path.basename(local_path)} "
                         f"(attempt {attempt}/{retries})")
            resp = _download_session.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            sz = os.path.getsize(local_path) / (1024 * 1024)
            logging.info(f"{thread_name}: downloaded {os.path.basename(local_path)} ({sz:.1f} MB)")
            return True
        except Exception as e:
            logging.warning(f"{thread_name}: JPSS download attempt {attempt} failed: {e}")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
    return False


def jpss_extract_tar(tar_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    extracted = []
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for m in members:
                m.name = os.path.basename(m.name)
            tf.extractall(out_dir, members=members, filter="data")
            for m in members:
                p = os.path.join(out_dir, m.name)
                if os.path.exists(p):
                    extracted.append(p)
                    if p.endswith(".gz"):
                        plain = p[:-3]
                        try:
                            with gzip.open(p, "rb") as fi, open(plain, "wb") as fo:
                                shutil.copyfileobj(fi, fo, length=1024 * 1024)
                            os.remove(p)
                            extracted[-1] = plain
                        except Exception as e:
                            logging.warning(f"JPSS gzip decompress failed for {p}: {e}")
    except Exception as e:
        logging.error(f"JPSS extract failed for {tar_path}: {e}")
        return []
    logging.info(f"JPSS extracted {len(extracted)} file(s) from {os.path.basename(tar_path)}")
    return extracted


def jpss_select_closest_files(file_paths, target_dt, max_files=12):
    scored = []
    for p in file_paths:
        gt = _jpss_parse_granule_time_from_name(p)
        if gt is None:
            score = 1e12
        else:
            score = -gt.timestamp() if target_dt is None else abs((gt - target_dt).total_seconds())
        scored.append((score, gt, p))
    scored.sort(key=lambda x: x[0])
    selected = [p for _, _, p in scored[:max_files]]
    if scored and scored[0][1] is not None:
        logging.info(f"JPSS closest granule time: {scored[0][1].strftime('%Y-%m-%d %H:%M')}Z "
                     f"(requested {target_dt.strftime('%Y-%m-%d %H:%M') + 'Z' if target_dt else 'latest'})")
    return selected


def jpss_resolve_product(date_str, family, preferred_product=None):
    products = jpss_list_products(date_str, family)
    if not products:
        logging.warning(f"JPSS: no products under {date_str}/{family}")
        return None
    if preferred_product:
        for p in products:
            if p.lower() == preferred_product.lower() or preferred_product.lower() in p.lower():
                return p
        logging.warning(f"JPSS: product '{preferred_product}' not found under {family}; "
                        f"available: {products[:12]}...")
    defaults = JPSS_FAMILY_DEFAULT_PRODUCTS.get(family, [])
    for d in defaults:
        for p in products:
            if d.lower() in p.lower() or p.lower() in d.lower():
                return p
    return products[0]


def jpss_resolve_sat(date_str, family, product, preferred_sat=None):
    sats = jpss_list_sats(date_str, family, product)
    if not sats:
        logging.warning(f"JPSS: no satellite dirs under {date_str}/{family}/{product}")
        return None

    alias = {
        "NOAA-21": "J02", "NOAA21": "J02", "J02": "J02", "N21": "J02",
        "NOAA-20": "J01", "NOAA20": "J01", "J01": "J01", "N20": "J01",
        "NPP": "NPP", "SNPP": "NPP", "S-NPP": "NPP", "NOAA": None,
    }
    want = None
    if preferred_sat:
        key = preferred_sat.strip().upper().replace("_", "-")
        want = alias.get(key, key)
        if want is None:
            preferred_sat = None
        else:
            for s in sats:
                if s.upper() == want.upper():
                    logging.info(f"JPSS: selected satellite {s} ({preferred_sat})")
                    return s
            logging.warning(f"JPSS: requested {preferred_sat} ({want}) not under product; "
                            f"available {sats} — falling back to auto")

    for prefer, label in (("J02", "NOAA-21"), ("J01", "NOAA-20"), ("NPP", "S-NPP"), ("npp", "S-NPP")):
        for s in sats:
            if s.upper() == prefer.upper():
                logging.info(f"JPSS: auto-selected satellite {s} ({label})")
                return s
    logging.info(f"JPSS: using first available satellite dir {sats[0]}")
    return sats[0]


def discover_jpss_files(family, target_dt=None, date_str=None, time_str=None,
                        product=None, sat_id=None, center_lat=None, center_lon=None):
    if target_dt is None and date_str:
        try:
            if time_str:
                target_dt = datetime.datetime.strptime(date_str + time_str[:4], "%Y%m%d%H%M")
            else:
                target_dt = datetime.datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            target_dt = None

    seed_date = jpss_closest_date(date_str)
    if not seed_date:
        return None

    def _match_family(families, family):
        if family in families:
            return family
        m = next((f for f in families if f.upper() == family.upper()), None)
        if m is None:
            m = next((f for f in families if family.upper() in f.upper() or f.upper() in family.upper()), None)
        if m and re.match(r"^\d{8}$", m):
            return None
        return m

    all_dates = jpss_list_available_dates()
    try:
        target_d = datetime.datetime.strptime(seed_date, "%Y%m%d").date()
        ordered = sorted(all_dates, key=lambda d: abs(
            (datetime.datetime.strptime(d, "%Y%m%d").date() - target_d).days))
    except ValueError:
        ordered = list(reversed(all_dates))

    chosen_date = matched_family = None
    last_families = []
    for d in ordered[:21]:
        families = jpss_list_families(d)
        last_families = families
        m = _match_family(families, family)
        if m:
            chosen_date, matched_family = d, m
            break
    if not chosen_date:
        logging.error(
            f"JPSS family '{family}' not found on CLASS (checked {min(21, len(ordered))} day(s)). "
            f"Latest sample: {last_families}. Try --JPSS-GRAN or --date with an older day."
        )
        return None

    avail_date, family = chosen_date, matched_family
    logging.info(
        f"JPSS: using CLASS date {avail_date} / family {family}"
        + (f" (requested {date_str})" if date_str and date_str != avail_date else "")
        + (" [nearest day with this family]" if not date_str or date_str != avail_date else "")
    )

    prod = jpss_resolve_product(avail_date, family, product)
    if not prod:
        return None
    logging.info(f"JPSS: product = {prod}")

    sat = jpss_resolve_sat(avail_date, family, prod, preferred_sat=sat_id)
    if not sat:
        return None
    logging.info(f"JPSS: satellite = {sat}")

    tars = jpss_list_tars(avail_date, family, prod, sat)
    if not tars:
        logging.warning(f"JPSS: no TAR files under {avail_date}/{family}/{prod}/{sat}")
        return None
    logging.info(f"JPSS: found {len(tars)} TAR(s)")
    return {
        "date": avail_date, "family": family, "product": prod, "sat": sat,
        "tars": tars, "target_dt": target_dt,
    }


def download_jpss_and_extract(meta, work_dir, download_workers=4, max_tars=2):
    os.makedirs(work_dir, exist_ok=True)
    tars = meta["tars"][:max_tars]
    local_tars, tasks = [], []
    for name, url in tars:
        lpath = os.path.join(work_dir, name)
        local_tars.append(lpath)
        if not os.path.exists(lpath):
            tasks.append((url, lpath))
    if tasks:
        logging.info(f"JPSS: downloading {len(tasks)} TAR(s)...")
        with ThreadPoolExecutor(max_workers=min(download_workers, len(tasks)),
                                thread_name_prefix="JPSS") as ex:
            futures = [ex.submit(jpss_download_tar, u, p) for u, p in tasks]
            for f in as_completed(futures):
                f.result()
    all_files = []
    for lt in local_tars:
        if not os.path.exists(lt):
            continue
        extract_dir = os.path.join(work_dir, "extract_" + os.path.basename(lt).replace(".tar", ""))
        all_files.extend(jpss_extract_tar(lt, extract_dir))
    if not all_files:
        return None
    return jpss_select_closest_files(all_files, meta.get("target_dt"), max_files=24)


def _jpss_satpy_reader_for_files(files):
    names = " ".join(os.path.basename(f).lower() for f in files)
    if any(x in names for x in ("jrr-", "surfref", "lst_", "aod", "adp", "cloudmask", "cloudheight")):
        return "viirs_edr"
    if any(x in names for x in ("svm", "svi", "svdnb", "gimgo", "gitco", "gmodo", "gmtco", "gdnbo")):
        return "viirs_sdr"
    return "viirs_sdr"


def process_jpss_data(local_files, target_area, target_dt, composite_type, resample_type="nearest"):
    if not local_files:
        raise ValueError("No JPSS local files")
    reader = _jpss_satpy_reader_for_files(local_files)
    logging.info(f"JPSS: loading {len(local_files)} file(s) with satpy reader '{reader}'")
    try:
        scn = Scene(filenames=local_files, reader=reader)
    except Exception as e:
        alt = "viirs_edr" if reader == "viirs_sdr" else "viirs_sdr"
        logging.warning(f"JPSS: reader {reader} failed ({e}); trying {alt}")
        scn = Scene(filenames=local_files, reader=alt)

    ir_candidates = ["I05", "M15", "M16", "brightness_temperature_I5", "brightness_temperature_M15",
                     "BT", "BrightnessTemperature", "cloud_top_temperature"]
    vis_candidates = ["I01", "M05", "M03", "reflectance_I1", "reflectance_M5"]

    def _try_load(names):
        for n in names:
            try:
                scn.load([n])
                if n in scn:
                    return n
            except Exception:
                continue
        try:
            available = scn.available_dataset_names()
            logging.info(f"JPSS available datasets (sample): {available[:20]}")
            for n in names:
                if n in available:
                    scn.load([n])
                    return n
            for a in available:
                al = a.lower()
                if "i05" in al or "m15" in al or "brightness" in al or al.endswith("_bt"):
                    scn.load([a])
                    return a
        except Exception as e:
            logging.warning(f"JPSS available_dataset_names failed: {e}")
        return None

    if composite_type in ("infrared", "dvorak", "ir", "z1-ir", "althea-ott2", "bt0", "z1-dvorak"):
        key = _try_load(ir_candidates)
        if key is None:
            raise ValueError("JPSS: could not load an IR brightness-temperature dataset")
        data = scn[key]
        if target_area is not None:
            res = scn.resample(target_area, resampler=resample_type,
                               reduce_data=True, radius_of_influence=20000)
            data = res[key]
        arr = np.asarray(data.compute() if hasattr(data, "compute") else data, dtype=np.float32)
        if np.nanmax(arr) < 100:
            arr = arr + 273.15
        return arr, None, None, None

    if composite_type in ("sandwich", "irv", "falsecolor", "falsecoloradv", "true", "b03", "z1-true"):
        ir_key = _try_load(ir_candidates)
        vis_key = _try_load(vis_candidates)
        if ir_key is None and vis_key is None:
            raise ValueError("JPSS: no VIS/IR datasets available for this composite")
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=20000) if target_area is not None else scn
        ir = vis = None
        if ir_key and ir_key in res:
            ir = np.asarray(res[ir_key].compute() if hasattr(res[ir_key], "compute") else res[ir_key], dtype=np.float32)
            if np.nanmax(ir) < 100:
                ir = ir + 273.15
        if vis_key and vis_key in res:
            vis = np.asarray(res[vis_key].compute() if hasattr(res[vis_key], "compute") else res[vis_key], dtype=np.float32)
        if composite_type == "b03" and vis is not None:
            return vis, None, None, None
        if vis is not None and ir is not None:
            if composite_type in ("falsecolor", "falsecoloradv", "sandwich", "irv"):
                from pyorbital.astronomy import sun_zenith_angle
                if target_area is not None:
                    lons, lats = target_area.get_lonlats()
                    sza = sun_zenith_angle(target_dt or datetime.datetime.utcnow(), lons, lats)
                else:
                    sza = np.zeros(vis.shape, dtype=np.float32)
                r, g, b = _false_color_rgb(vis, ir, sza, advanced=(composite_type == "falsecoloradv"))
                return r, g, b, None
            if composite_type in ("true", "z1-true"):
                vis_n = _normalize_reflectance(vis)
                ir_n = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
                return vis_n, vis_n, vis_n * 0.85 + ir_n * 0.15, None
        if ir is not None:
            return ir, None, None, None
        if vis is not None:
            return vis, None, None, None

    key = _try_load(ir_candidates + vis_candidates)
    if key is None:
        raise ValueError(f"JPSS: unsupported composite {composite_type} / no datasets")
    data = scn[key]
    if target_area is not None:
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=20000)
        data = res[key]
    arr = np.asarray(data.compute() if hasattr(data, "compute") else data, dtype=np.float32)
    return arr, None, None, None


def process_jpss_storm(storm, crop_km, product, output_dir, output_width,
                       family, jpss_product=None, jpss_sat=None,
                       date_str=None, time_str=None,
                       download_workers=4, logo_path=None,
                       grid=False, grid_thick=0.4, grid_color="#00BFFF", grid_style="--",
                       no_coastlines=False, label=False,
                       export_formats=None, floater=False, info=False, project="flat"):
    from pyresample import create_area_def

    storm_id = storm.get("atcf_id") or storm.get("storm_name", "UNKNOWN")
    lat = storm.get("latitude")
    lon = storm.get("longitude")
    if lat is None or lon is None:
        logging.warning(f"JPSS: storm {storm_id} has no lat/lon; skipping")
        return

    target_dt = None
    if date_str:
        tpart = (time_str or "0000")[:4]
        try:
            target_dt = datetime.datetime.strptime(date_str + tpart, "%Y%m%d%H%M")
        except ValueError:
            target_dt = None

    meta = discover_jpss_files(
        family, target_dt=target_dt, date_str=date_str, time_str=time_str,
        product=jpss_product, sat_id=jpss_sat, center_lat=lat, center_lon=lon)
    if not meta:
        logging.error(f"JPSS: discovery failed for family={family}")
        return

    work_dir = tempfile.mkdtemp(prefix=f"jpss_{family}_")
    try:
        local_files = download_jpss_and_extract(meta, work_dir, download_workers=download_workers)
        if not local_files:
            logging.error("JPSS: no granules after download/extract")
            return

        half_km = (crop_km or 1000) / 2.0
        lat_deg = half_km / 111.32
        lon_deg = half_km / (111.32 * max(np.cos(np.radians(lat)), 0.05))
        if all(storm.get(k) is not None for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
            extent = [storm["lon_min"], storm["lat_min"], storm["lon_max"], storm["lat_max"]]
            width_m = abs(storm["lon_max"] - storm["lon_min"]) * 111320 * max(np.cos(np.radians(lat)), 0.05)
            height_m = abs(storm["lat_max"] - storm["lat_min"]) * 111320
        else:
            extent = [lon - lon_deg, lat - lat_deg, lon + lon_deg, lat + lat_deg]
            width_m = 2 * half_km * 1000
            height_m = 2 * half_km * 1000
        px = max(int(output_width or 2000), 512)
        py = max(int(round(px * (height_m / max(width_m, 1)))), 512)
        target_area = create_area_def(
            "jpss_crop", {"proj": "latlong", "datum": "WGS84"},
            area_extent=extent, shape=(py, px))

        obs_dt = (meta.get("target_dt")
                  or _jpss_parse_granule_time_from_name(local_files[0])
                  or datetime.datetime.strptime(meta["date"], "%Y%m%d"))

        composite = product
        if composite in ("ir", "infrared", "z1-ir", "althea-ott2", "bt0", "dvorak", "z1-dvorak"):
            ir, _, _, _ = process_jpss_data(local_files, target_area, obs_dt, "infrared")
            ir = np.nan_to_num(ir, nan=300.0)
            ir_c = ir - 273.15
            if composite in ("dvorak", "z1-dvorak"):
                cmap = mcolors.LinearSegmentedColormap.from_list("Dvorak", DVORAK_nodes)
                display = "DVORAK (VIIRS)"
            elif composite == "bt0":
                cmap = mcolors.ListedColormap(_DVORAK_IR_LUT, name="dvorak_ir")
                display = "BT0 VIIRS"
            else:
                cmap = mcolors.LinearSegmentedColormap.from_list("OTT", OTT_nodes)
                display = "IR (VIIRS)"
            vmin, vmax = -100, 50
            plot_data, is_rgb = ir_c, False
        else:
            r, g, b, _ = process_jpss_data(local_files, target_area, obs_dt, composite)
            if g is None:
                plot_data = np.nan_to_num(r, nan=0.0)
                if np.nanmax(plot_data) > 200:
                    plot_data = plot_data - 273.15
                    cmap = mcolors.LinearSegmentedColormap.from_list("OTT", OTT_nodes)
                    vmin, vmax = -100, 50
                else:
                    cmap = "gray"
                    vmin, vmax = 0, 1
                display, is_rgb = composite.upper(), False
            else:
                plot_data = _stack_rgb(r, g, b)
                cmap = vmin = vmax = None
                display, is_rgb = composite, True

        sat_tag = f"VIIRS-{meta['sat']}"
        ts = obs_dt.strftime("%Y%m%d_%H%M")
        out_base = os.path.join(output_dir, f"{storm_id}_{ts}_{product}_{sat_tag}")
        metadata = {
            "satellite_name": f"JPSS/{meta['family']}/{meta['sat']}",
            "target_dt": obs_dt, "center_lat": lat, "center_lon": lon,
            "crop_deg": max(lon_deg, lat_deg), "crop_lon": lon_deg, "crop_lat": lat_deg,
            "product": display, "storm_id": storm_id, "storm_name": storm.get("storm_name", ""),
            "winds": storm.get("winds"), "pressure": storm.get("pressure"),
            "grid": grid, "grid_thick": grid_thick, "grid_color": grid_color, "grid_style": grid_style,
            "no_coastlines": no_coastlines, "label": label,
            "par": False, "tcad": False, "tcid": False, "ico": False, "invest": False,
            "active_storms": None, "crop_km": crop_km, "polygon": storm.get("polygon"),
            "floater": floater, "info": info, "target_extent": extent,
        }
        _jpss_simple_plot(plot_data, out_base, metadata, cmap=cmap, vmin=vmin, vmax=vmax,
                          logo_path=logo_path, export_formats=export_formats or ["avif"], is_rgb=is_rgb)
        logging.info(f"JPSS: wrote {out_base}.*")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _jpss_simple_plot(data, out_base, metadata, cmap=None, vmin=None, vmax=None,
                      logo_path=None, export_formats=None, is_rgb=False):
    extent = metadata.get("target_extent")
    fig = plt.figure(figsize=(10, 10), dpi=200, facecolor="black")
    proj = ccrs.PlateCarree()
    ax = fig.add_axes([0, 0, 1, 1], projection=proj, facecolor="black")
    if is_rgb:
        ax.imshow(data, origin="upper", extent=extent, transform=proj, interpolation="nearest")
    else:
        ax.imshow(data, origin="upper", extent=extent, transform=proj,
                  cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if not metadata.get("no_coastlines"):
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="#00FF00")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#00FF00", alpha=0.5)
    if extent:
        ax.set_extent([extent[0], extent[2], extent[1], extent[3]], crs=proj)
    ax.axis("off")
    if metadata.get("grid") or metadata.get("label"):
        gl = ax.gridlines(draw_labels=metadata.get("label"), linewidth=metadata.get("grid_thick", 0.4),
                          color=metadata.get("grid_color", "#00BFFF"), alpha=0.6,
                          linestyle=metadata.get("grid_style", "--"))
        gl.top_labels = False
        gl.right_labels = False
    try:
        add_modern_info(ax, metadata, logo_path)
    except Exception:
        pass
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches=None, pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    with Image.open(buf) as img:
        img = img.convert("RGB")
        for fmt in (export_formats or ["avif"]):
            fmt = fmt.lower().strip()
            out = f"{out_base}.{fmt}"
            try:
                if fmt == "avif":
                    img.save(out, format="AVIF", quality=95, subsampling="4:4:4")
                elif fmt == "png":
                    img.save(out, format="PNG", compress_level=1)
                elif fmt in ("jpg", "jpeg"):
                    img.save(out, format="JPEG", quality=95)
                elif fmt == "webp":
                    img.save(out, format="WEBP", quality=95)
                else:
                    continue
                logging.info(f"Saved: {out}")
            except OSError as e:
                alt = f"{out_base}.png"
                logging.warning(f"Save {fmt} failed ({e}); fallback PNG")
                img.save(alt, format="PNG", compress_level=1)



def main():
    parser = argparse.ArgumentParser(description="MonWatch-CLI — Automated satellite storm imagery")
    parser.add_argument("--storm", help="Specific storm ID or name")
    parser.add_argument("--auto-satellite", dest="auto_satellite", action="store_true",
                        help="Auto-pick the best observing geostationary satellite for the "
                             "target location (e.g. Himawari for the Philippines/WPac, "
                             "GOES-East for the Atlantic, GK-2A for East Asia, MTG for "
                             "Africa/Europe when EUMETSAT credentials are present). "
                             "Candidates are ranked by viewing geometry, then probed in "
                             "order until one has data at the requested or latest "
                             "timestamp. Overrides --him/--gk2a/--goes/--mtg and --multi.")
    parser.add_argument("--filter", dest="basin_filter", default=None,
                    help="Comma-separated basin filter for current ATCF storms. "
                         "Examples: WP, EPAC, CPAC, AL/ATL, IO, SH, SI, AU, SP. "
                         "Omit to process every active storm.")
    parser.add_argument("--crop-km", type=float, default=1000,
                        help="Crop size in km (shortcuts: --1000, --3000)")
    parser.add_argument("--product", default="sandwich",
                        type=lambda s: s.lower(),
                        choices=["sandwich", "true", "dvorak", "ir", "infrared", "z1-ir", "althea-ott2", "z1-true", "z1-dvorak", "b03", "irv", "bt0", "falsecolor", "falsecoloradv", "firetemp", "dayconv", "fire"],
                        help="Composite product: sandwich, true, dvorak, ir (BT PWARDS), infrared (Him), z1-ir (MonWatch IR), althea-ott2 (OTT2 IR, Althea Kate), z1-true (MonWatch True), z1-dvorak (MonWatch Dvorak), b03 (B03 visible), irv (B03+B13 Rayleigh IR blend), bt0 (BT0 PWARDS), falsecolor (VIS/IR false color), falsecoloradv (false color advanced), firetemp (fire temperature RGB), fire (3.9um shortwave IR hotspot), dayconv (day convection RGB)")
    parser.add_argument("--products", type=str, default=None,
                        help="Comma-separated list of products to generate in one batch (ir,infrared,z1-ir,althea-ott2,z1-dvorak,bt0,dvorak,falsecolor,falsecoloradv,firetemp,dayconv,fire). Overrides --product.")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--width", type=int, default=2000, help="Output width in pixels (square)")
    parser.add_argument("--logo", nargs="?", const="logo/splash.png", default=None,
                        help="Show the PWARDS logo (default path logo/splash.png, or pass a custom PNG path). Omit for no logo.")
    parser.add_argument("--info", action="store_true",
                        help="Show the info box overlay (storm name, time, coords, winds)")
    parser.add_argument("--radar", action="store_true",
                        help="Radar mode (placeholder; use --garbinradar for the GarbinWx composite)")
    parser.add_argument("--radar-type", default="DBZ", type=lambda s: s.upper(),
                        choices=["DBZ", "RR", "RAIN"], help="Radar base product: DBZ (reflectivity) or RR/RAIN (rain rate)")
    parser.add_argument("--garbinradar", action="store_true",
                        help="Fetch the GarbinWx radar composite image (requires garbinwxid via .env)")
    parser.add_argument("--phradar", action="store_true",
                        help="Fetch PAGASA Panahon national radar mosaic (panahon.gov.ph; no key required)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--decompress-workers", type=int, default=8)
    parser.add_argument("--latest", action="store_true",
                        help="Force latest available time (default behavior)")
    parser.add_argument("--date", type=str,
                        help="Date in YYYYMMDD format (use with --time)")
    parser.add_argument("--time", type=str,
                        help="Time in HHMM format (use with --date)")
    parser.add_argument("--datefrom", type=str,
                        help="Start date in YYYYMMDD format (for range)")
    parser.add_argument("--dateto", type=str,
                        help="End date in YYYYMMDD format (for range)")
    parser.add_argument("--timefrom", type=str,
                        help="Start time in HHMM format (for range)")
    parser.add_argument("--timeto", type=str,
                        help="End time in HHMM format (for range)")
    parser.add_argument("--target", action="store_true",
                        help="Download from AHI-L1b-Target (Region 3) instead of FLDK")
    parser.add_argument("--fulldisk", action="store_true",
                        help="Native full disk (satellite GEOS). IR (AHI B13 R20) is already 2 km "
                             "so it is NOT resampled onto a Plate Carree 2 km grid. "
                             "Renders once — not once per active storm.")
    sat_group = parser.add_mutually_exclusive_group()
    sat_group.add_argument("--him", action="store_true",
                            help="Use Himawari (AHI) satellite data (default)")
    sat_group.add_argument("--him9", action="store_true",
                           help="Use Himawari-9 specifically")
    sat_group.add_argument("--him8", action="store_true",
                           help="Use Himawari-8 specifically")
    sat_group.add_argument("--gk2a", action="store_true",
                            help="Use GEO-KOMPSAT-2A (AMI) satellite data from NOAA PDS")
    sat_group.add_argument("--goes", action="store_true",
                            help="Auto-select GOES-East or GOES-West from scene longitude "
                                 "(East if lon > -106, else West; uses GOES-19/18 with 16/17 fallback)")
    sat_group.add_argument("--goes16", action="store_true",
                            help="Use GOES-16 (GOES-East legacy) data")
    sat_group.add_argument("--goes17", action="store_true",
                            help="Use GOES-17 (GOES-West legacy) data")
    sat_group.add_argument("--goes18", action="store_true",
                            help="Use GOES-18 (GOES-West) data; falls back to GOES-17 if down")
    sat_group.add_argument("--goes19", action="store_true",
                            help="Use GOES-19 (GOES-East) data; falls back to GOES-16 if down")
    sat_group.add_argument("--mtg", action="store_true",
                            help="Use MTG satellite data from EUMETSAT")
    sat_group.add_argument("--mtsat", action="store_true",
                            help="Use MTSAT-2 (Himawari-7) historical data from CEReS Chiba University raw HRIT archive (2007-2015)")
    sat_group.add_argument("--mtsat1", action="store_true",
                            help="Use MTSAT-1R (Himawari-6) historical data from CEReS Chiba University raw HRIT archive (2005-2014)")
    sat_group.add_argument("--mtsat2", action="store_true",
                            help="Use MTSAT-2 (Himawari-7) historical data from CEReS Chiba University raw HRIT archive (2007-2015); alias for --mtsat")
    sat_group.add_argument("--VIIRS-SDR", dest="viirs_sdr", action="store_true",
                            help="Use JPSS VIIRS Sensor Data Records from NOAA CLASS (polar-orbiting)")
    sat_group.add_argument("--VIIRS-EDR", dest="viirs_edr", action="store_true",
                            help="Use JPSS VIIRS Environmental Data Records from NOAA CLASS")
    sat_group.add_argument("--JPSS-GRAN", dest="jpss_gran", action="store_true",
                            help="Use JPSS-GRAN family (VIIRS granule EDRs) from NOAA CLASS")
    sat_group.add_argument("--VIIRSI-EDR", dest="viirsi_edr", action="store_true",
                            help="Use VIIRSI-EDR family from NOAA CLASS")
    parser.add_argument("--jpss", type=str, default=None,
                        help="JPSS CLASS family name (e.g. VIIRS-SDR, VIIRS-EDR, JPSS-GRAN)")
    parser.add_argument("--jpss-product", type=str, default=None,
                        help="Optional CLASS product subfolder under the JPSS family")
    parser.add_argument("--jpss-sat", type=str, default=None,
                        help="Optional JPSS satellite id (J01=NOAA-20, J02=NOAA-21, NPP)")
    parser.add_argument("--NOAA-20", dest="noaa20", action="store_true",
                        help="Use NOAA-20 (J01) for JPSS/VIIRS")
    parser.add_argument("--NOAA-21", dest="noaa21", action="store_true",
                        help="Use NOAA-21 (J02) for JPSS/VIIRS")
    parser.add_argument("--NOAA", dest="noaa_auto", action="store_true",
                        help="Auto-select JPSS satellite (prefer NOAA-21/J02, then NOAA-20/J01, then NPP)")
    parser.add_argument("--NPP", dest="npp_sat", action="store_true",
                        help="Use S-NPP for JPSS/VIIRS")
    parser.add_argument("--multi", action="store_true",
                        help="Process with BOTH Himawari and GK-2A; equivalent to running --him then --gk2a with the same parameters. Output filenames get a _HIM9 or _GK2A suffix")
    parser.add_argument("--global", dest="global_mode", action="store_true",
                        help="Global mode: fetch GOES + Himawari-9 (+Himawari-8 fallback) + GK-2A + MTG at the SAME timestamp and render them as stacked full-disk panels on one world map")
    parser.add_argument("--floater", action="store_true",
                        help="Floater mode: image with colorscale on right, coordinates on edges, metadata header on top")
    parser.add_argument("--project", type=str, default="flat",
                        help="Map projection (default: flat). "
                             "flat=PlateCarree/eqc resample+display; "
                             "geos/native=resample+display in geostationary (faster, no eqc); "
                             "disk=orthographic display; equal_earth, robinson, mollweide, "
                             "mercator, sinusoidal=display-only reprojection. "
                             "Aliases: ortho, equalearth, platecarree, eqc, nat.")
    parser.add_argument("--export", type=str, default="avif",
                        help="Output format(s): avif, png, mp4, jpg, webp (comma-separated for multiple)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for MP4 export (default 30)")
    parser.add_argument("--nopng", action="store_true",
                        help="MP4 export: pipe frames directly to ffmpeg without writing PNG files")
    parser.add_argument("--grid", action="store_true",
                        help="Show lat/lon grid lines (cyan)")
    parser.add_argument("--thick", type=float, default=0.4,
                        help="Grid line thickness (default 0.4)")
    parser.add_argument("--color", type=str, default="#00BFFF",
                        help="Grid line color (hex or name, default cyan)")
    parser.add_argument("--style", type=str, default="--",
                        help="Grid line style: solid, dashed, dotted, dashdot (default dashed)")
    parser.add_argument("--label", action="store_true",
                        help="Show lat/lon labels on image edges")
    parser.add_argument("--no-coastlines", action="store_true",
                        help="Hide coastlines and borders")
    parser.add_argument("--par", action="store_true",
                        help="Draw PAR alert polygon")
    parser.add_argument("--tcad", action="store_true",
                        help="Draw TCAD polygon")
    parser.add_argument("--tcid", action="store_true",
                        help="Draw TCID polygon")
    parser.add_argument("--ico", action="store_true",
                        help="Draw storm icons from ico/ folder")
    parser.add_argument("--invest", action="store_true",
                        help="Draw white circles for invests")
    parser.add_argument("--peak", action="store_true",
                        help="Fetch archive track and use peak intensity time")
    parser.add_argument("--year", type=int, default=None,
                        help="Season year for --storm historical IBTrACS track lookup (NOAA NCEI, full archive back to the 1970s)")
    parser.add_argument("--philippines", action="store_true",
                        help="Center on Philippines (125E, 12N) instead of a storm")
    parser.add_argument("--westpac", action="store_true",
                        help="Center on Western Pacific (140E, 13.5N) instead of a storm")
    parser.add_argument("--nl", action="store_true",
                        help="Center on North Luzon (118-123E, 13-15N) instead of a storm")
    parser.add_argument("--sl", action="store_true",
                        help="Center on South Luzon (polygon crop) instead of a storm")
    parser.add_argument("--ipar", action="store_true",
                        help="Center on PAR polygon (polygon crop) instead of a storm")
    parser.add_argument("--pmd", action="store_true",
                        help="Center on PAGASA Monitoring Domain (125E, 12N) with wide bounds")
    parser.add_argument("--camsur", action="store_true",
                        help="Camarines Sur (polygon crop)")
    parser.add_argument("--pwardsc", action="store_true",
                        help="Pasacao Sector (segment crop)")
    parser.add_argument("--manila", action="store_true",
                        help="Manila Sector (polygon crop)")
    parser.add_argument("--region", type=str, default=None,
                        help="Named region(s) to center on, comma-separated (e.g. conus,europe). "
                             "Auto-selects a default satellite when none is given. "
                             "Available: " + ", ".join(sorted(NAMED_REGIONS)))
    parser.add_argument("--conus", action="store_true",
                        help="Center on CONUS (defaults to GOES-19); alias for --region conus")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory of pre-downloaded raw .DAT files (from scan loop)")
    parser.add_argument("--lat", type=float, default=None,
                        help="Center latitude for a custom storm location")
    parser.add_argument("--lon", type=float, default=None,
                        help="Center longitude for a custom storm location")
    args, unknown = parser.parse_known_args()


    basin_filter = None
    if getattr(args, "basin_filter", None):
        basin_filter = {
            _normalize_basin(x)
            for x in re.split(r"[,\s;]+", args.basin_filter)
            if x.strip()
        }
        
    if args.goes16:
        sat_source = "goes16"
    elif args.goes17:
        sat_source = "goes17"
    elif args.goes18:
        sat_source = "goes18"
    elif args.goes19:
        sat_source = "goes19"
    elif args.goes:
        sat_source = "goes"
    elif args.mtg:
        sat_source = "mtg"
    elif args.mtsat1:
        sat_source = "mtsat1"
    elif args.mtsat2 or args.mtsat:
        sat_source = "mtsat"
    elif args.gk2a:
        sat_source = "gk2a"
    elif args.him9:
        sat_source = "him9"
    elif args.him8:
        sat_source = "him8"
    elif args.him:
        sat_source = "him"
    elif getattr(args, "viirs_sdr", False):
        sat_source = "VIIRS-SDR"
    elif getattr(args, "viirs_edr", False):
        sat_source = "VIIRS-EDR"
    elif getattr(args, "jpss_gran", False):
        sat_source = "JPSS-GRAN"
    elif getattr(args, "viirsi_edr", False):
        sat_source = "VIIRSI-EDR"
    elif getattr(args, "jpss", None):
        sat_source = args.jpss.strip()
    else:
        sat_source = None

    if getattr(args, "noaa21", False):
        args.jpss_sat = "J02"
    elif getattr(args, "noaa20", False):
        args.jpss_sat = "J01"
    elif getattr(args, "npp_sat", False):
        args.jpss_sat = "NPP"
    elif getattr(args, "noaa_auto", False):
        if not getattr(args, "jpss_sat", None):
            args.jpss_sat = None
        logging.info("JPSS: --NOAA auto (prefer NOAA-21/J02, then NOAA-20/J01, then NPP)")

    _jpss_platform = any([
        getattr(args, "noaa20", False), getattr(args, "noaa21", False),
        getattr(args, "noaa_auto", False), getattr(args, "npp_sat", False),
        bool(getattr(args, "jpss_sat", None)),
    ])
    if _jpss_platform and sat_source is None:
        sat_source = "VIIRS-SDR"
        logging.info("JPSS: no family flag given; defaulting to VIIRS-SDR")

    multi = bool(getattr(args, "multi", False))
    _any_goes = any([args.goes, getattr(args, "goes16", False), getattr(args, "goes17", False),
                     args.goes18, args.goes19])
    if multi:
        sat_sources = []
        if args.him or not any([args.gk2a, _any_goes, args.mtg, args.mtsat, args.mtsat1, args.mtsat2]):
            sat_sources.append("him")
        if args.gk2a:
            sat_sources.append("gk2a")
        if getattr(args, "goes16", False):
            sat_sources.append("goes16")
        if getattr(args, "goes17", False):
            sat_sources.append("goes17")
        if args.goes18:
            sat_sources.append("goes18")
        if args.goes19:
            sat_sources.append("goes19")
        if args.goes and not any([getattr(args, "goes16", False), getattr(args, "goes17", False),
                                  args.goes18, args.goes19]):
            sat_sources.append("goes")
        if args.mtg:
            sat_sources.append("mtg")
        if args.mtsat1:
            sat_sources.append("mtsat1")
        if args.mtsat2:
            sat_sources.append("mtsat2")
        if args.mtsat:
            sat_sources.append("mtsat")
        if len(sat_sources) > 1:
            sat_names = {"him": "Himawari-9", "gk2a": "GK-2A", "goes": "GOES",
                         "goes16": "GOES-16", "goes17": "GOES-17",
                         "goes18": "GOES-18", "goes19": "GOES-19", "mtg": "MTG",
                         "mtsat": "MTSAT-2", "mtsat2": "MTSAT-2", "mtsat1": "MTSAT-1R"}
            sat_list = [sat_names.get(s, s) for s in sat_sources]
            logging.warning(f"--multi: processing with {' and '.join(sat_list)}")
    else:
        sat_sources = [sat_source]
    if not multi and sat_source in ("gk2a", "goes", "goes16", "goes17", "goes18", "goes19", "mtg", "mtsat", "mtsat1"):
        if args.target:
            logging.warning("--target is only supported with Himawari; disabling for this satellite source.")
            args.target = False
        if sat_source == "gk2a" and "--color" not in sys.argv:
            args.color = GK2A_DEFAULT_GRID_COLOR

    def _sat_output(sat):
        return args.output

    def _sat_use_target(sat):
        return args.target if sat == "him" else False

    def _sat_color(sat):
        if sat == "him":
            return args.color if ("--color" in sys.argv or sat == "him") else "#00BFFF"
        elif sat == "gk2a":
            return args.color if ("--color" in sys.argv) else GK2A_DEFAULT_GRID_COLOR
        elif sat in ("goes", "goes16", "goes17", "goes18", "goes19"):
            return args.color if ("--color" in sys.argv) else "#00FF00"
        elif sat == "mtg":
            return args.color if ("--color" in sys.argv) else "#FFFF00"
        elif sat in ("mtsat", "mtsat2"):
            return args.color if ("--color" in sys.argv) else "#FF00FF"
        elif sat == "mtsat1":
            return args.color if ("--color" in sys.argv) else "#FF8000"
        else:
            return args.color if ("--color" in sys.argv) else "#FFFFFF"
    if "--1000" in unknown:
        args.crop_km = 1000
    if "--3000" in unknown:
        args.crop_km = 3000

    export_formats = [fmt.strip().lower() for fmt in args.export.split(",")]
    valid_formats = {'avif', 'png', 'jpg', 'jpeg', 'webp', 'mp4'}
    export_formats = [f for f in export_formats if f in valid_formats]
    if not export_formats:
        export_formats = ['avif']

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    _load_dotenv()

    if args.radar:
        logging.warning("--radar is a placeholder; use --garbinradar for the GarbinWx radar composite viewer.")

    _sat_flag_given = any([args.him, args.him9, args.him8, args.gk2a, args.goes,
                       getattr(args, "goes16", False), getattr(args, "goes17", False),
                       args.goes18, args.goes19, args.mtg, args.mtsat,
                       args.mtsat1, args.mtsat2,
                       getattr(args, "viirs_sdr", False), getattr(args, "viirs_edr", False),
                       getattr(args, "jpss_gran", False), getattr(args, "viirsi_edr", False),
                       bool(getattr(args, "jpss", None))])
    radar_overlay = None
    if _sat_flag_given:
        if args.garbinradar:
            radar_overlay = _garbin_radar_overlay(args.radar_type, args.date, args.time)
            if radar_overlay is None:
                logging.warning("GarbinWx radar overlay requested but no composite available; rendering satellite only.")
        elif getattr(args, "phradar", False):
            radar_overlay = _phradar_radar_overlay(args.radar_type, args.date, args.time)
            if radar_overlay is None:
                logging.warning("Panahon radar overlay requested but no composite available; rendering satellite only.")

    logging.info("Starting MonWatch-CLI")
    logging.info(f"Crop: {args.crop_km} km, Product: {args.product}, Width: {args.width} px")

    if getattr(args, "global_mode", False):
        try:
            process_global(args.output, args.width, args.product, args.logo,
                           latest=args.latest, date_str=args.date, time_str=args.time,
                           grid=args.grid, grid_thick=args.thick, grid_color=args.color,
                           grid_style=args.style, no_coastlines=args.no_coastlines,
                           label=args.label, export_formats=export_formats,
                           download_workers=args.download_workers, quiet=not args.verbose)
        except Exception as e:
            logging.error(f"Global mode failed: {e}")
            if args.verbose:
                logging.debug(traceback.format_exc())
            sys.exit(1)
        logging.info("Done.")
        return

    if args.timefrom or args.timeto or args.datefrom or args.dateto:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        log_date_from = args.datefrom or now_utc.strftime("%Y%m%d")
        log_date_to = args.dateto or log_date_from
        log_time_from = args.timefrom or "0000"
        log_time_to = args.timeto or now_utc.strftime("%H%M")
        logging.info(f"Creating a From-To Data: From: {log_date_from} {log_time_from} | To: {log_date_to} {log_time_to}")

    region_specs = []
    if args.conus:
        region_specs.append(("conus", NAMED_REGIONS["conus"]))
    if args.region:
        for part in args.region.replace(";", ",").split(","):
            name = part.strip().lower().replace(" ", "-")
            if not name:
                continue
            spec = NAMED_REGIONS.get(name)
            if spec is None:
                logging.error(f"Unknown region '{name}'. Available: {', '.join(sorted(NAMED_REGIONS))}")
                sys.exit(1)
            region_specs.append((name, spec))
    if region_specs:
        default_sat = next((s.get("satellite") for _, s in region_specs if s.get("satellite")), None)
        explicit_sat = any([args.him, args.him9, args.him8, args.gk2a, args.goes,
                            getattr(args, "goes16", False), getattr(args, "goes17", False),
                            args.goes18, args.goes19, args.mtg, args.mtsat, args.mtsat1, args.mtsat2])
        if default_sat and not explicit_sat and (sat_source is None or sat_sources == [None]):
            sat_source = default_sat
            sat_sources = [default_sat]
            logging.info(f"Region default satellite: {default_sat}")

    if args.products:
        batch_products = [p.strip().lower() for p in args.products.split(",")]
        valid_batch = [p for p in batch_products if p in BATCH_PRODUCTS]
        if not valid_batch:
            logging.error(f"No valid products in --products. Valid: {BATCH_PRODUCTS}")
            sys.exit(1)
        logging.info(f"Batch mode: processing {valid_batch}")
        process_batch = True
    else:
        valid_batch = None
        process_batch = False

    if args.lat is not None and args.lon is not None:
        if (args.garbinradar or getattr(args, "phradar", False)) and not _sat_flag_given:
            storm = {
                "atcf_id": "CUSTOM",
                "storm_name": "Custom Radar",
                "latitude": args.lat,
                "longitude": args.lon,
                "winds": None,
                "pressure": None,
            }
            os.makedirs(args.output, exist_ok=True)
            logging.info(f"Radar‑only mode: custom center ({args.lat}, {args.lon})")
            try:
                _radar_viewer = process_phradar_viewer if getattr(args, "phradar", False) else process_garbin_radar_viewer
                _radar_viewer(
                    storm, args.output, args.width,
                    radar_type=args.radar_type,
                    date_str=args.date, time_str=args.time,
                    crop_km=args.crop_km,
                    info=args.info, logo_path=args.logo,
                    grid=args.grid, grid_thick=args.thick,
                    grid_color=args.color, grid_style=args.style,
                    no_coastlines=args.no_coastlines,
                    label=args.label, export_formats=export_formats,
                    floater=args.floater
                )
            except Exception as e:
                logging.error(f"Radar viewer failed: {e}")
                if args.verbose:
                    logging.debug(traceback.format_exc())
                sys.exit(1)
            logging.info("Done.")
            return
            
        ibtracs_track = None
        custom_storm = {
            "atcf_id": "CUSTOM",
            "storm_name": "Custom Point",
            "latitude": args.lat,
            "longitude": args.lon,
            "winds": None,
            "pressure": None,
        }
        os.makedirs(args.output, exist_ok=True)
        logging.info(f"Custom center: ({args.lat}, {args.lon}) - processing single point.")
        for sat in sat_sources:
            out_dir = _sat_output(sat)
            os.makedirs(out_dir, exist_ok=True)
            logging.info(f"  Satellite: {sat} -> {out_dir}")
            try:
                if process_batch:
                    process_storm_batch(custom_storm, args.crop_km, valid_batch, out_dir, args.width,
                                        args.download_workers, args.decompress_workers, args.logo,
                                        args.latest, args.date, args.time,
                                        args.grid, args.thick, _sat_color(sat), args.style, args.no_coastlines, args.label,
                                        args.par, args.tcad, args.tcid,
                                        ico=args.ico, invest=args.invest, peak=False, active_storms=[],
                                        data_dir=args.data_dir, export_formats=export_formats,
                                        date_from=args.datefrom, date_to=args.dateto, time_from=args.timefrom, time_to=args.timeto,
                                        use_target=_sat_use_target(sat), floater=args.floater, fps=args.fps, nopng=args.nopng,
                                        track=ibtracs_track, sat_source=sat, fulldisk=args.fulldisk, info=args.info, radar_overlay=radar_overlay,
                                        project=args.project)
                else:
                    process_storm(custom_storm, args.crop_km, args.product, out_dir, args.width,
                                  args.download_workers, args.decompress_workers, args.logo,
                                  args.latest, args.date, args.time,
                                  args.grid, args.thick, _sat_color(sat), args.style, args.no_coastlines, args.label,
                                  args.par, args.tcad, args.tcid,
                                  ico=args.ico, invest=args.invest, peak=False, active_storms=[],
                                  data_dir=args.data_dir, export_formats=export_formats,
                                  date_from=args.datefrom, date_to=args.dateto, time_from=args.timefrom, time_to=args.timeto,
                                  use_target=_sat_use_target(sat), floater=args.floater, fps=args.fps, nopng=args.nopng,
                                  track=ibtracs_track, sat_source=sat, fulldisk=args.fulldisk, info=args.info, radar_overlay=radar_overlay,
                                        project=args.project, jpss_product=getattr(args, 'jpss_product', None), jpss_sat=getattr(args, 'jpss_sat', None))
            except Exception as e:
                logging.error(f"Failed to process custom point ({sat}): {e}")
                if args.verbose:
                    import traceback
                    logging.debug(traceback.format_exc())
        logging.info("Done.")
        return

    all_storms = fetch_knackwx_atcf()
    if not all_storms:
        if args.year:
            logging.warning("KnackWx ATCF unavailable; relying on --year IBTrACS lookup.")
        elif args.garbinradar or getattr(args, "phradar", False):
            logging.warning("No ATCF data received; continuing for radar viewer (fixed region).")
        else:
            logging.error("No ATCF data received.")
            sys.exit(1)

    logging.info(f"Received {len(all_storms)} total ATCF storms.")

    active_storms = _filter_storms_by_basin(all_storms, basin_filter)
    if basin_filter:
        logging.info(f"Basin filter {sorted(basin_filter)} -> {len(active_storms)} active storm(s).")
    else:
        logging.info(f"No basin filter -> {len(active_storms)} active storm(s).")

    ibtracs_track = None
    if args.storm:
        target = args.storm.upper()
        storms = [s for s in all_storms if
                  s.get("atcf_id", "").upper() == target or
                  s.get("long_atcf_id", "").upper() == target or
                  s.get("storm_name", "").upper() == target]
        if args.year:
            ibtracs_track = fetch_ibtracs_track(target, args.year)
            if ibtracs_track is not None:
                storms = [build_ibtracs_storm(ibtracs_track, target, args.date, args.time, args.peak)]
                logging.info(f"Found historical (IBTrACS): {target} -> {storms[0]['atcf_id']} "
                             f"({len(ibtracs_track)} fixes)")
        elif args.date and len(args.date) >= 4 and args.date[:4].isdigit():
            hist_year = int(args.date[:4])
            ibtracs_track = fetch_ibtracs_track(target, hist_year)
            if ibtracs_track is not None:
                storms = [build_ibtracs_storm(ibtracs_track, target, args.date, args.time, args.peak)]
                logging.info(f"Found historical (IBTrACS): {target} -> {storms[0]['atcf_id']} "
                             f"({len(ibtracs_track)} fixes)")
        if not storms:
            logging.info(f"Storm '{target}' not in current data; fetching historical archive...")
            atcf_id = None
            if len(target) == 3 and target[-1].isalpha() and target[:-1].isdigit():
                atcf_id = target
            elif target.startswith('WP') or target.startswith('EP'):
                atcf_id = target[2:5].upper()
            elif target.isdigit() and len(target) <= 2:
                atcf_id = f"{target.zfill(2)}W"
            else:
                for s in fetch_knackwx_atcf():
                    if s.get("storm_name", "").upper() == target:
                        atcf_id = s.get("atcf_id", "")
                        if atcf_id and len(atcf_id) == 3 and atcf_id[:-1].isdigit():
                            break
            if atcf_id:
                url = f"https://api.knackwx.com/atcf/v2/track/archive?stormID={atcf_id}"
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200 and r.text.strip():
                        lines = r.text.strip().splitlines()
                        latest = lines[-1]
                        parts = [p.strip() for p in latest.split(",")]
                        if len(parts) >= 8:
                            lat_str, lon_str = parts[6], parts[7]
                            lat = float(lat_str[:-1]) / 10.0 * (1 if lat_str[-1] == 'N' else -1)
                            lon = float(lon_str[:-1]) / 10.0 * (1 if lon_str[-1] == 'E' else -1)
                            storms = [{
                                "atcf_id": atcf_id,
                                "long_atcf_id": atcf_id,
                                "storm_name": target,
                                "latitude": lat,
                                "longitude": lon,
                                "basin": "WP",
                                "winds": int(parts[8]),
                                "pressure": int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else None
                            }]
                            logging.info(f"Found historical: {target} ({atcf_id}) at ({lat:.1f}, {lon:.1f})")
                except Exception as e:
                    logging.warning(f"Historical fetch failed: {e}")
        if not storms:
            logging.error(f"Storm '{target}' not found.")
            sys.exit(1)
        logging.info(f"Found {len(storms)} storm(s) matching '{target}'.")
    else:
        storms = active_storms
        if basin_filter:
            logging.info(f"Filtered to {len(storms)} storm(s) in basins: {sorted(basin_filter)}")
        else:
            logging.info(f"Processing all {len(storms)} active storm(s).")

    if args.target:
        args.crop_km = 1000
        storms = [{"atcf_id": "TARGET", "storm_name": "Target Area",
                    "latitude": 20.0, "longitude": 135.0,
                    "lat_min": 15.0, "lat_max": 25.0,
                    "lon_min": 130.0, "lon_max": 140.0,
                    "winds": None, "pressure": None}]
        logging.info("Target Area (AHI-L1b-Target) - dynamic region from R3xx observations")
    elif args.philippines:
        args.crop_km = 0
        storms = [{"atcf_id": "PHL", "storm_name": "Philippines",
                    "latitude": 12.0, "longitude": 125.0,
                    "lat_min": -3, "lat_max": 27,
                    "lon_min": 100, "lon_max": 150,
                    "winds": None, "pressure": None}]
        logging.info("Philippines region (100-150E, 3S-27N)")
    elif args.westpac:
        args.crop_km = 0
        storms = [{"atcf_id": "WPAC", "storm_name": "WestPac",
                    "latitude": 13.5, "longitude": 140.0,
                    "lat_min": -3, "lat_max": 30,
                    "lon_min": 100, "lon_max": 180,
                    "winds": None, "pressure": None}]
        logging.info("WestPac region (100-180E, 3S-30N)")
    elif args.nl:
        args.crop_km = 0
        nl_polygon = [(119.5358742096911, 13.02009544514386),
                      (124.0340104026942, 14.18402026467277),
                      (122.7516577088883, 19.35769875199276),
                      (118.0952363998624, 18.26122615445196),
                      (119.5358742096911, 13.02009544514386)]
        storms = [{"atcf_id": "NL", "storm_name": "North Luzon",
                    "latitude": 16.25, "longitude": 121.25,
                    "lat_min": 13.0, "lat_max": 19.5,
                    "lon_min": 118.0, "lon_max": 124.5,
                    "polygon": nl_polygon,
                    "winds": None, "pressure": None}]
        logging.info("North Luzon region (polygon crop)")
    elif args.sl:
        args.crop_km = 0
        sl_polygon = [(123.435562492173, 11.35680688409037),
                      (125.9211075633607, 12.68106906170556),
                      (124.7942287667553, 14.73186587827745),
                      (121.9301316542767, 13.63005565300873),
                      (123.435562492173, 11.35680688409037)]
        storms = [{"atcf_id": "SL", "storm_name": "South Luzon",
                    "latitude": 13.0, "longitude": 123.75,
                    "lat_min": 11.0, "lat_max": 15.0,
                    "lon_min": 121.5, "lon_max": 126.0,
                    "polygon": sl_polygon,
                    "winds": None, "pressure": None}]
        logging.info("South Luzon region (polygon crop)")
    elif args.ipar:
        args.crop_km = 0
        ipar_polygon = [(115, 5), (115, 15), (120, 21), (120, 25), (135, 25), (135, 5), (115, 5)]
        storms = [{"atcf_id": "IPAR", "storm_name": "PAR (Philippine Area of Responsibility)",
                    "latitude": 15.0, "longitude": 125.0,
                    "lat_min": 3, "lat_max": 27,
                    "lon_min": 113, "lon_max": 137,
                    "polygon": ipar_polygon,
                    "winds": None, "pressure": None}]
        logging.info("PAR region (polygon crop, 2deg margin)")
    elif args.pmd:
        args.crop_km = 0
        storms = [{"atcf_id": "PMD", "storm_name": "PAGASA Monitoring Domain",
                    "latitude": 12.5, "longitude": 135.0,
                    "lat_min": -10, "lat_max": 35,
                    "lon_min": 100, "lon_max": 170,
                    "winds": None, "pressure": None}]
        logging.info("PAGASA Monitoring Domain (100-170E, 10S-35N)")
    elif args.camsur:
        args.crop_km = 0
        storms = [{"atcf_id": "PWARDS", "storm_name": "Camarines Sur",
                    "latitude": 13.5, "longitude": 123.0,
                    "lat_min": 12.5, "lat_max": 14.5,
                    "lon_min": 122.0, "lon_max": 124.0,
                    "winds": None, "pressure": None}]
        logging.info("Camarines Sur (segment crop)")
    elif args.pwardsc:
        args.crop_km = 0
        pwards_polygon = [(122.842079, 13.5978129), (122.7995069, 13.5657756), (123.1133039, 13.3514154),
                          (123.2169874, 13.4816571), (122.9835279, 13.6478625), (122.842079, 13.5978129)]
        storms = [{"atcf_id": "PWARDSC", "storm_name": "Pasacao Sector",
                    "latitude": 13.5, "longitude": 123.0,
                    "lat_min": 12.5, "lat_max": 14.5,
                    "lon_min": 122.0, "lon_max": 124.0,
                    "polygon": pwards_polygon,
                    "winds": None, "pressure": None}]
        logging.info("Pasacao Sector (polygon crop)")
    elif args.manila:
        args.crop_km = 0
        manila_polygon = [(119.9266126249696, 14.96836194974614),
                          (120.3698817906371, 13.64168093050986),
                          (122.0166302239072, 14.02675364957583),
                          (121.5460988652676, 15.38266289322173),
                          (119.9266126249696, 14.96836194974614)]
        storms = [{"atcf_id": "MNLA", "storm_name": "Manila Sector",
                    "latitude": 14.5, "longitude": 121.0,
                    "lat_min": 13.5, "lat_max": 15.5,
                    "lon_min": 119.5, "lon_max": 122.5,
                    "polygon": manila_polygon,
                    "winds": None, "pressure": None}]
        logging.info("Manila Sector (polygon crop)")
    elif region_specs:
        args.crop_km = 0
        args.peak = False
        storms = []
        for name, spec in region_specs:
            storms.append({
                "atcf_id": name.upper().replace("-", ""),
                "storm_name": spec["storm_name"],
                "latitude": spec["latitude"],
                "longitude": spec["longitude"],
                "lat_min": spec["lat_min"],
                "lat_max": spec["lat_max"],
                "lon_min": spec["lon_min"],
                "lon_max": spec["lon_max"],
                "winds": None,
                "pressure": None,
            })
            if "polygon" in spec:
                storms[-1]["polygon"] = spec["polygon"]
            logging.info(f"Region '{name}' -> {spec['storm_name']} "
                         f"({spec['lon_min']:.0f}-{spec['lon_max']:.0f}E, "
                         f"{spec['lat_min']:.0f}-{spec['lat_max']:.0f}N)")
    elif not storms:
        if args.fulldisk:
            logging.info("No active storms; continuing with a single full-disk scene.")
        else:
            logging.info("No active Western Pacific storms to process.")
            return

        
    if (args.garbinradar or getattr(args, "phradar", False)) and not _sat_flag_given:
        if args.lat is not None and args.lon is not None:
            storm = {
                "atcf_id": "CUSTOM",
                "storm_name": "Custom Radar",
                "latitude": args.lat,
                "longitude": args.lon,
                "winds": None,
                "pressure": None,
            }
            logging.info(f"Radar‑only mode: custom center ({args.lat}, {args.lon})")
        elif storms:
            storm = storms[0]
            logging.info(f"Radar‑only mode: using region '{storm.get('atcf_id')}' centered at ({storm.get('latitude')}, {storm.get('longitude')})")
        else:
            storm = {
                "atcf_id": "PHL",
                "storm_name": "Philippines",
                "latitude": 12.0,
                "longitude": 125.0,
                "lat_min": -3, "lat_max": 27,
                "lon_min": 100, "lon_max": 150,
                "winds": None,
                "pressure": None,
            }
            logging.info("Radar‑only mode: using Philippines region (no custom lat/lon)")
            
        os.makedirs(args.output, exist_ok=True)
        try:
            _radar_viewer = process_phradar_viewer if getattr(args, "phradar", False) else process_garbin_radar_viewer
            _radar_viewer(
                storm, args.output, args.width,
                radar_type=args.radar_type,
                date_str=args.date, time_str=args.time,
                crop_km=args.crop_km,
                info=args.info, logo_path=args.logo,
                grid=args.grid, grid_thick=args.thick,
                grid_color=args.color, grid_style=args.style,
                no_coastlines=args.no_coastlines,
                label=args.label, export_formats=export_formats,
                floater=args.floater
            )
        except Exception as e:
            logging.error(f"Radar viewer failed: {e}")
            if args.verbose:
                logging.debug(traceback.format_exc())
            sys.exit(1)
        logging.info("Done.")
        return

    os.makedirs(args.output, exist_ok=True)

    if args.fulldisk:
        keep_label = bool(args.storm) and storms
        label_src = storms[0] if keep_label else {}
        storms = [{
            "atcf_id": (label_src.get("atcf_id") if keep_label else None) or "FLDK",
            "storm_name": (label_src.get("storm_name") if keep_label else None) or "Full Disk",
            "latitude": 0.0,
            "longitude": 140.7,
            "winds": label_src.get("winds") if keep_label else None,
            "pressure": label_src.get("pressure") if keep_label else None,
        }]
        logging.info(f"Fulldisk: single native scene ({storms[0]['atcf_id']}), not per-storm")

    auto_sat_mode = (getattr(args, "auto_satellite", False)
                     and not _sat_flag_given and not args.fulldisk)
    auto_probe_bands = None
    if auto_sat_mode:
        if process_batch:
            auto_probe_bands = sorted({b for p in valid_batch
                                       for b in PRODUCT_BANDS.get(p, [13])})
        else:
            auto_probe_bands = PRODUCT_BANDS.get(args.product, [13])
        logging.info("--auto-satellite: per-storm satellite selection enabled.")

    for storm in storms:
        if auto_sat_mode:
            s_lat = storm.get("latitude")
            s_lon = storm.get("longitude")
            if (s_lat is None or s_lon is None) and all(
                    storm.get(k) is not None for k in
                    ("lat_min", "lat_max", "lon_min", "lon_max")):
                s_lat = (storm["lat_min"] + storm["lat_max"]) / 2.0
                s_lon = (storm["lon_min"] + storm["lon_max"]) / 2.0
            if s_lat is None or s_lon is None:
                logging.warning(f"  [auto-satellite] no target coords for "
                                f"{storm.get('atcf_id', 'UNKNOWN')}; skipping.")
                continue

            cands = _auto_satellite_candidates(s_lat, s_lon)
            if not cands:
                logging.warning(f"  [auto-satellite] no candidate satellite has "
                                f"a usable view of {storm.get('atcf_id')} "
                                f"({s_lat:.2f}, {s_lon:.2f}); skipping.")
                continue
            logging.info(f"  [auto-satellite] {storm.get('atcf_id')} "
                         f"({s_lat:.2f}, {s_lon:.2f}) -> "
                         + ", ".join(n for _, n in cands))

            picked_src, picked_name, picked_dt = None, None, None
            for cand_src, cand_name in cands:
                logging.info(f"    Probing {cand_name} ({cand_src})...")
                pdt, _bucket = _auto_probe_satellite(
                    cand_src, args.date, args.time, auto_probe_bands,
                    use_target=False)
                if pdt is None:
                    logging.info(f"    [--] {cand_name} unavailable")
                    continue
                logging.info(f"    [OK] {cand_name} has data at "
                             f"{pdt.strftime('%Y-%m-%d %H:%M')}Z")
                if picked_dt is None or pdt > picked_dt:
                    picked_src, picked_name, picked_dt = cand_src, cand_name, pdt
            if picked_dt is not None:
                logging.info(f"    Picked newest: {picked_name} "
                             f"({picked_dt.strftime('%Y-%m-%d %H:%M')}Z)")
            if picked_src is None:
                logging.warning(f"  [auto-satellite] no candidate had data for "
                                f"{storm.get('atcf_id')}; skipping.")
                continue
            logging.info(f"  [auto-satellite] selected {picked_name} "
                         f"({picked_src}) for {storm.get('atcf_id')}")
            sats_for_storm = [picked_src]
        else:
            sats_for_storm = list(sat_sources)

        for sat in sats_for_storm:
            out_dir = _sat_output(sat)
            os.makedirs(out_dir, exist_ok=True)
            logging.info(f"  Satellite: {sat} -> {out_dir}")
            try:
                if process_batch:
                    process_storm_batch(storm, args.crop_km, valid_batch, out_dir, args.width,
                                        args.download_workers, args.decompress_workers, args.logo,
                                        args.latest, args.date, args.time,
                                        args.grid, args.thick, _sat_color(sat), args.style, args.no_coastlines, args.label,
                                        args.par, args.tcad, args.tcid,
                                        ico=args.ico, invest=args.invest, peak=args.peak, active_storms=active_storms,
                                        data_dir=args.data_dir, export_formats=export_formats,
                                        date_from=args.datefrom, date_to=args.dateto, time_from=args.timefrom, time_to=args.timeto,
                                        use_target=_sat_use_target(sat), floater=args.floater, fps=args.fps, nopng=args.nopng,
                                        track=ibtracs_track, sat_source=sat, fulldisk=args.fulldisk, info=args.info, radar_overlay=radar_overlay,
                                        project=args.project)
                else:
                    process_storm(storm, args.crop_km, args.product, out_dir, args.width,
                                  args.download_workers, args.decompress_workers, args.logo,
                                  args.latest, args.date, args.time,
                                  args.grid, args.thick, _sat_color(sat), args.style, args.no_coastlines, args.label,
                                  args.par, args.tcad, args.tcid,
                                  ico=args.ico, invest=args.invest, peak=args.peak, active_storms=active_storms,
                                  data_dir=args.data_dir, export_formats=export_formats,
                                  date_from=args.datefrom, date_to=args.dateto, time_from=args.timefrom, time_to=args.timeto,
                                  use_target=_sat_use_target(sat), floater=args.floater, fps=args.fps, nopng=args.nopng,
                                  track=ibtracs_track, sat_source=sat, fulldisk=args.fulldisk, info=args.info, radar_overlay=radar_overlay,
                                        project=args.project, jpss_product=getattr(args, 'jpss_product', None), jpss_sat=getattr(args, 'jpss_sat', None))
            except Exception as e:
                logging.error(f"Failed to process storm {storm.get('atcf_id')} ({sat}): {e}")
                if args.verbose:
                    import traceback
                    logging.debug(traceback.format_exc())

    logging.info("Done.")

if __name__ == "__main__":
    main()
