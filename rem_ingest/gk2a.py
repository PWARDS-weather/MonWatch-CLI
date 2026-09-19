# =============================================================================
# rem_ingest/gk2a.py — GK-2A AMI ingest for MonWatch-CLI
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
import shutil
import logging
import datetime
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr
import pyproj
from pyresample import AreaDefinition, kd_tree

from . import common
from .common import fs, _download_session, linear_normalize, false_color_rgb
from ._rgb_corrections import apply_rgb_corrections

GK2A_BUCKET                  = "noaa-gk2a-pds"
GK2A_HTTPS_BASE              = "https://noaa-gk2a-pds.s3.amazonaws.com"
GK2A_DEFAULT_GRID_COLOR      = "#FFFF00"
GK2A_DEFAULT_COASTLINE_COLOR = "#e433ff"

GK2A_CHANNEL_BAND = {
    "vi004": (1, 0.470), "vi005": (2, 0.509), "vi006": (3, 0.639), "vi008": (4, 0.863),
    "nr016": (5, 1.610),
    "sw038": (7, 3.830), "wv063": (8, 6.210), "wv069": (9, 6.940), "wv073": (10, 7.330),
    "ir087": (11, 8.590), "ir096": (12, 9.620), "ir105": (13, 10.350), "ir112": (14, 11.230),
    "ir123": (15, 12.360), "ir133": (16, 13.290),
}
GK2A_BAND_CHANNEL = {b: c for c, (b, _) in GK2A_CHANNEL_BAND.items()}
GK2A_CHANNEL_CWL  = {c: wl for c, (_, wl) in GK2A_CHANNEL_BAND.items()}
GK2A_IR_CHANNELS  = {"sw038", "wv063", "wv069", "wv073",
                     "ir087", "ir096", "ir105", "ir112", "ir123", "ir133"}

_GK2A_PLANK = {"h": 6.62606957e-34, "c": 2.99792458e8, "k": 1.3806488e-23}

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
    proj = {"proj": "geos", "h": h, "lon_0": sub_lon_deg,
            "a": a_rad, "b": b_rad, "sweep": "x"}
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


def _gk2a_radiance(attrs, dn):
    gain = float(attrs["DN_to_Radiance_Gain"])
    offset = float(attrs["DN_to_Radiance_Offset"])
    rad = dn * gain + offset
    return np.where(rad < 0, np.nan, rad)


def _gk2a_brightness_temperature(attrs, rad, wl_um):
    h = _GK2A_PLANK["h"]; c = _GK2A_PLANK["c"]; k = _GK2A_PLANK["k"]
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
    cfac = float(attrs["cfac"]); lfac = float(attrs["lfac"])
    coff = float(attrs["coff"]); loff = float(attrs["loff"])
    ncols = int(attrs["number_of_columns"]); nlines = int(attrs["number_of_lines"])
    cols = coff + np.rad2deg(gx / h) * (cfac / 2 ** 16)
    rows = loff + np.rad2deg(gy / h) * (lfac / 2 ** 16)
    finite = np.isfinite(cols) & np.isfinite(rows)
    if not finite.any():
        return None
    cols = cols[finite]; rows = rows[finite]
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

        def _calibrate(dn):
            rad = _gk2a_radiance(attrs, dn)
            if code in GK2A_IR_CHANNELS:
                return _gk2a_brightness_temperature(attrs, rad, GK2A_CHANNEL_CWL[code])
            rad_to_alb = float(attrs.get("Radiance_to_Albedo_c", 0.0))
            return rad * rad_to_alb * 100.0

        if target_area is None:
            data = ds["image_pixel_values"].values.astype(np.float32)
            data = _calibrate(data)
            return np.where(np.isfinite(data), data, np.nan).astype(np.float32), full_area

        bbox = _gk2a_source_bbox(full_area, attrs, target_area)
        if bbox is None:
            logging.info(f"GK2A: global/wide target area; full-disk resample "
                         f"for {os.path.basename(local_path)}")
            data = ds["image_pixel_values"].values.astype(np.float32)
            data = _calibrate(data)
            data = np.where(np.isfinite(data), data, np.nan).astype(np.float32)
            out = kd_tree.resample_nearest(full_area, data, target_area,
                                           radius_of_influence=50000,
                                           fill_value=np.nan, reduce_data=True)
            return out.astype(np.float32), full_area

        row0, row1, col0, col1 = bbox
        dn = ds["image_pixel_values"].isel(
            dim_image_y=slice(row0, row1),
            dim_image_x=slice(col0, col1),
        ).values.astype(np.float32)
        data = _calibrate(dn)
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
    stamp = (f"_{dt_obj.year:04d}{dt_obj.month:02d}{dt_obj.day:02d}"
             f"{dt_obj.hour:02d}{dt_obj.minute:02d}.nc")
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
        stamp = (f"_{dt.year:04d}{dt.month:02d}{dt.day:02d}"
                 f"{dt.hour:02d}{dt.minute:02d}.nc")
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
        with ThreadPoolExecutor(max_workers=download_workers,
                                thread_name_prefix='Downloader') as ex:
            futures = {ex.submit(download_gk2a_file, r, l, b): (r, l, b)
                       for r, l, b in tasks}
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
            logging.warning(f"  Prefetch: missing GK2A bands {missing} for "
                            f"{dt.strftime('%Y-%m-%d %H:%M')}Z")
            continue
        for b, paths in remote_map.items():
            all_remote.setdefault(b, []).extend(paths)
        fetched += 1
    if not all_remote:
        shutil.rmtree(cache_dir, ignore_errors=True)
        return None, None
    logging.info(f"Prefetching {len(all_remote)} GK2A band set(s) for "
                 f"{fetched} slot(s) into {cache_dir}...")
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


def process_gk2a_data(local_files_map, target_area, target_dt, composite_type,
                     resample_type="nearest"):
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
        r = linear_normalize(_band(7), 273.0, 350.0)
        g = linear_normalize(_band(5), 0.0, 50.0)
        b = linear_normalize(_band(9), 0.0, 50.0)
        return r, g, b, None

    if composite_type == "fire":
        return _band(7), None, None, None

    if composite_type in ("infrared", "dvorak"):
        return _band(13), None, None, None

    if composite_type in ("sandwich", "irv"):
        return _band(3), _band(13), None, None

    if composite_type == "b03":
        return _band(3), None, None, None

    if composite_type in ("falsecolor", "falsecoloradv"):
        vis = _band(3)
        ir  = _band(13)
        from pyorbital.astronomy import sun_zenith_angle
        if target_area is None:
            sza = np.zeros(vis.shape, dtype=np.float32)
        else:
            lons, lats = target_area.get_lonlats()
            sza = sun_zenith_angle(target_dt, lons, lats)
        r, g, b = false_color_rgb(vis, ir, sza,
                                  advanced=(composite_type == "falsecoloradv"))
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
        b03 = _band(3); b05 = _band(5); b07 = _band(7)
        b08 = _band(8); b10 = _band(10); b13 = _band(13)
        r = linear_normalize(b08 - b10, -35.0, 5.0)
        g = linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
        b = linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
        return r, g, b, None

    raise ValueError(f"Unsupported GK2A composite: {composite_type}")