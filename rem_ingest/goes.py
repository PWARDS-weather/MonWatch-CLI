# =============================================================================
# rem_ingest/goes.py — GOES-16/17/18/19 ABI ingest for MonWatch-CLI
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
import re
import logging
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr
from pyresample import AreaDefinition, kd_tree

from . import common
from .common import (
    fs, _download_session, resize_like, linear_normalize, false_color_rgb,
    AHI_TO_ABI,
)
from ._rgb_corrections import apply_rgb_corrections

GOES_SATELLITE_MAP = {
    "goes16": "noaa-goes16",
    "goes17": "noaa-goes17",
    "goes18": "noaa-goes18",
    "goes19": "noaa-goes19",
}
GOES_DEFAULT = "goes18"

_GOES_PLANK = {"h": 6.62606957e-34, "c": 2.99792458e8, "k": 1.3806488e-23}

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
        candidates = [f for f in all_files if f"M6{band_str}" in f]
        if not candidates:
            logging.warning(f"No GOES files found for band {band_str} in {path}")
            continue
        closest = min(candidates,
                      key=lambda f: abs((_goes_extract_time(f) - target_naive).total_seconds()))
        discovered[b] = [closest]
    return discovered


def get_latest_available_dt_goes(satellite, max_hours=4):
    bucket = GOES_SATELLITE_MAP.get(satellite, satellite)
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    best = None
    for hour_offset in range(0, max_hours + 1):
        test_dt = now - datetime.timedelta(hours=hour_offset)
        path = (f"{bucket}/ABI-L1b-RadF/{test_dt.year}/"
                f"{test_dt.strftime('%j')}/{test_dt.hour:02d}/")
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
            logging.info(f"{thread_name}: downloaded B{band:02d} "
                         f"{os.path.basename(local_path)} ({file_size:.1f} MB)")
            return True
        except Exception as e:
            logging.warning(f"{thread_name}: attempt {attempt}/{retries} failed "
                            f"for {os.path.basename(remote_path)}: {e}")
            if os.path.exists(local_path):
                try: os.remove(local_path)
                except Exception: pass
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
        with ThreadPoolExecutor(max_workers=download_workers,
                                thread_name_prefix='Downloader') as ex:
            futures = {ex.submit(download_goes_file, r, l, b): (r, l, b)
                       for r, l, b in tasks}
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


def process_goes_data(local_files_map, target_area, target_dt,
                      composite_type, resample_type="nearest"):
    def _read(ahi_band, area=None):
        paths = local_files_map.get(ahi_band)
        if not paths:
            raise ValueError(f"Missing GOES band {ahi_band} "
                             f"(ABI C{AHI_TO_ABI[ahi_band]:02d})")
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
        r, g, b = false_color_rgb(vis, ir, sza,
                                  advanced=(composite_type == "falsecoloradv"))
        return r, g, b, None

    if composite_type == "firetemp":
        r = linear_normalize(_read(7), 273.0, 350.0)
        g = linear_normalize(_read(6), 0.0, 50.0)
        b = linear_normalize(_read(9), 0.0, 50.0)
        return r, g, b, None

    if composite_type == "fire":
        return _read(7), None, None, None

    if composite_type == "dayconv":
        b05, b03, b07, b08, b10, b13 = (_read(5), _read(3), _read(7),
                                         _read(8), _read(10), _read(13))
        r = linear_normalize(b08 - b10, -35.0, 5.0)
        g = linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
        b = linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
        return r, g, b, None

    if composite_type == "true":
        r = _read(3)
        b = resize_like(_read(1), r.shape)
        v = resize_like(_read(4), r.shape)
        ir = resize_like(_read(13), r.shape)
        g = 0.45 * r + 0.10 * v + 0.45 * b
        r, g, b = apply_rgb_corrections(r, g, b, ir, target_area, target_dt, mode=1)
        return r, g, b, None

    raise ValueError(f"Unsupported GOES composite: {composite_type}")