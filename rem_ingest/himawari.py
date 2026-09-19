# =============================================================================
# rem_ingest/himawari.py — Himawari-8/9 AHI ingest for MonWatch-CLI
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
import gc
import bz2
import glob
import shutil
import logging
import datetime
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import dask
import xarray as xr
from satpy import Scene
from PIL import Image

from . import common
from .common import (
    fs, _download_session, resize_like, reduced_area,
    area_with_shape, align_result_to_area, upscale_rgb,
    normalize_reflectance, linear_normalize, stack_rgb, false_color_rgb,
)

S3_HTTPS_BASE = "https://noaa-himawari9.s3.amazonaws.com"
HIMAWARI_BUCKETS = ("noaa-himawari9", "noaa-himawari8")
HIMAWARI_DEFAULT_SEGMENTS = [f"S{i:02d}" for i in range(1, 11)]

def discover_ahi_files(satellite, dt_obj, bands_list, segments,
                       use_target=False, target_segment=None):
    if use_target:
        path = (f"{satellite}/AHI-L1b-Target/"
                f"{dt_obj.year:04d}/{dt_obj.month:02d}/{dt_obj.day:02d}/"
                f"{dt_obj.hour:02d}{dt_obj.minute:02d}/")
    else:
        path = (f"{satellite}/AHI-L1b-FLDK/"
                f"{dt_obj.year:04d}/{dt_obj.month:02d}/{dt_obj.day:02d}/"
                f"{dt_obj.hour:02d}{dt_obj.minute:02d}/")
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
                    all_present = False; break
            else:
                for seg in segments:
                    if not any(band_str in f and f"_{seg}" in f for f in all_files):
                        all_present = False; break
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
            logging.info(f"{thread_name}: downloading B{band} segment {segment} "
                         f"from {os.path.basename(remote_path)} (attempt {attempt}/{retries})")
            key = remote_path
            bucket = "noaa-himawari9"
            for prefix in HIMAWARI_BUCKETS:
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
            logging.warning(f"{thread_name}: attempt {attempt}/{retries} failed for "
                            f"{remote_path}: {e}\n{tb}")
            if os.path.exists(local_path):
                try: os.remove(local_path)
                except Exception: pass
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
        logging.info(f"{thread_name}: decompressed B{band} segment {segment} -> "
                     f"{os.path.basename(dat_path)}")
    except Exception as e:
        logging.error(f"Failed to decompress {bz2_path}: {e}")


def download_and_decompress_all(remote_map, local_dir,
                                download_workers=16, decompress_workers=8):
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
        with ThreadPoolExecutor(max_workers=download_workers,
                                thread_name_prefix='Downloader') as ex:
            futures = {ex.submit(download_one, r, l, b, s, cancel_event): (r, l, b, s)
                       for r, l, b, s in download_tasks}
            for f in as_completed(futures):
                if cancel_event.is_set():
                    for other in futures: other.cancel()
                    all_ok = False; break
                if not f.result():
                    all_ok = False
                    cancel_event.set()
                    for other in futures: other.cancel()
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
        with ThreadPoolExecutor(max_workers=decompress_workers,
                                thread_name_prefix='Decompressor') as ex:
            futures = [ex.submit(decompress_one, bz, dat, b, s)
                       for bz, dat, b, s in decompress_tasks]
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


def process_ahi_data(local_files_map, target_area, target_dt, composite_type,
                     resample_type="nearest"):
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
            logging.info(f"DEBUG: B13 native range: {np.nanmin(ir):.1f} - "
                         f"{np.nanmax(ir):.1f} K, shape: {ir.shape}, "
                         f"nans: {np.isnan(ir).sum()}")
            return ir, None, None, None

        if composite_type == "sandwich":
            scn.load(["B03", "B13"])
            vis = scn["B03"].compute().astype(np.float32)
            ir = scn["B13"].compute().astype(np.float32)
            if vis.shape != ir.shape:
                vis = resize_like(vis, ir.shape)
            return vis, ir, None, None

        if composite_type == "b03":
            scn.load(["B03"])
            return scn["B03"].compute().astype(np.float32), None, None, None

        if composite_type == "b07":
            scn.load(["B07"])
            return scn["B07"].compute().astype(np.float32), None, None, None

        if composite_type == "b09":
            scn.load(["B09"])
            return scn["B09"].compute().astype(np.float32), None, None, None

        if composite_type == "irv":
            scn.load(["B03", "B13"])
            vis = scn["B03"].compute().astype(np.float32)
            ir = scn["B13"].compute().astype(np.float32)
            if vis.shape != ir.shape:
                vis = resize_like(vis, ir.shape)
            return vis, ir, None, None

        if composite_type in ("falsecolor", "falsecoloradv"):
            from pyorbital.astronomy import sun_zenith_angle
            scn.load(["B03", "B13"])
            vis = scn["B03"].compute().astype(np.float32)
            ir = scn["B13"].compute().astype(np.float32)
            target_shape = min([vis.shape, ir.shape], key=lambda s: s[0] * s[1])
            vis = resize_like(vis, target_shape)
            ir = resize_like(ir, target_shape)
            area = scn["B13"].attrs.get("area") or scn["B03"].attrs.get("area")
            if area is None:
                area = _native_target_area(local_files_map)
            if area is not None:
                area = area_with_shape(area, target_shape)
                sza = sun_zenith_angle(target_dt, *area.get_lonlats())
                sza = resize_like(np.asarray(sza, dtype=np.float32), target_shape)
            else:
                sza = np.zeros(target_shape, dtype=np.float32)
            r, g, b = false_color_rgb(vis, ir, sza,
                                      advanced=(composite_type == "falsecoloradv"))
            return r, g, b, None

        if composite_type == "firetemp":
            scn.load(["B07", "B06", "B09"])
            b07 = scn["B07"].compute().astype(np.float32)
            b06 = scn["B06"].compute().astype(np.float32)
            b09 = scn["B09"].compute().astype(np.float32)
            ts = min([b07.shape, b06.shape, b09.shape], key=lambda s: s[0] * s[1])
            b07 = resize_like(b07, ts); b06 = resize_like(b06, ts); b09 = resize_like(b09, ts)
            r = linear_normalize(b07, 273.0, 350.0)
            g = linear_normalize(b06, 0.0, 50.0)
            b = linear_normalize(b09, 0.0, 50.0)
            return r, g, b, None

        if composite_type == "dayconv":
            scn.load(["B05", "B03", "B07", "B08", "B10", "B13"])
            arrays = [scn[k].compute().astype(np.float32)
                      for k in ("B05", "B03", "B07", "B08", "B10", "B13")]
            ts = min([a.shape for a in arrays], key=lambda s: s[0] * s[1])
            arrays = [resize_like(a, ts) for a in arrays]
            b05, b03, b07, b08, b10, b13 = arrays
            r = linear_normalize(b08 - b10, -35.0, 5.0)
            g = linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
            b = linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
            return r, g, b, None

        if composite_type == "true":
            scn.load(["B01", "B03", "B04", "B13"])
            r = scn["B03"].compute().astype(np.float32)
            b = resize_like(scn["B01"].compute().astype(np.float32), r.shape)
            v = resize_like(scn["B04"].compute().astype(np.float32), r.shape)
            ir = resize_like(scn["B13"].compute().astype(np.float32), r.shape)
            g = 0.45 * r + 0.10 * v + 0.45 * b
            area = scn["B03"].attrs.get("area") or _native_target_area(local_files_map)
            if area is not None:
                from ._rgb_corrections import apply_rgb_corrections
                r, g, b = apply_rgb_corrections(r, g, b, ir, area, target_dt, mode=1)
                return r, g, b, None
            return r, g, b, ir

        raise ValueError(f"Unsupported composite_type for native AHI data: {composite_type}")

    if composite_type == "true":
        scn.load(["B01", "B03", "B04", "B13"])
        work_area = reduced_area(target_area)
        res = scn.resample(work_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        r, b, veggie, ir = dask.compute(
            res["B03"].data, res["B01"].data, res["B04"].data, res["B13"].data)
        r = r.astype(np.float32); b = b.astype(np.float32)
        veggie = veggie.astype(np.float32); ir = ir.astype(np.float32)
        g = 0.45 * r + 0.10 * veggie + 0.45 * b
        from ._rgb_corrections import apply_rgb_corrections
        r_c, g_c, b_c = apply_rgb_corrections(r, g, b, ir, work_area, target_dt, mode=1)
        r_c, g_c, b_c = upscale_rgb(r_c, g_c, b_c, target_area)
        return r_c, g_c, b_c, None

    if composite_type in ("infrared", "dvorak"):
        scn.load(["B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        ir = res["B13"].data.compute().astype(np.float32)
        logging.info(f"DEBUG: B13 raw range: {np.nanmin(ir):.1f} - "
                     f"{np.nanmax(ir):.1f} K, shape: {ir.shape}, nans: {np.isnan(ir).sum()}")
        return ir, None, None, None

    if composite_type == "sandwich":
        scn.load(["B03", "B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        vis, ir = dask.compute(res["B03"].data, res["B13"].data)
        return vis.astype(np.float32), ir.astype(np.float32), None, None

    if composite_type == "b03":
        scn.load(["B03"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        return res["B03"].data.compute().astype(np.float32), None, None, None

    if composite_type == "b07":
        scn.load(["B07"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        return res["B07"].data.compute().astype(np.float32), None, None, None

    if composite_type == "b09":
        scn.load(["B09"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        return res["B09"].data.compute().astype(np.float32), None, None, None

    if composite_type == "irv":
        scn.load(["B03", "B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        vis, ir = dask.compute(res["B03"].data, res["B13"].data)
        vis = vis.astype(np.float32); ir = ir.astype(np.float32)
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
        scn.load(["B03", "B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        vis, ir = dask.compute(res["B03"].data, res["B13"].data)
        vis = vis.astype(np.float32); ir = ir.astype(np.float32)
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
        scn.load(["B07", "B06", "B09"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        b07, b06, b09 = dask.compute(res["B07"].data, res["B06"].data, res["B09"].data)
        b07 = b07.astype(np.float32); b06 = b06.astype(np.float32); b09 = b09.astype(np.float32)
        r = linear_normalize(b07, 273.0, 350.0)
        g = linear_normalize(b06, 0.0, 50.0)
        b = linear_normalize(b09, 0.0, 50.0)
        return r, g, b, None

    if composite_type == "dayconv":
        scn.load(["B05", "B03", "B07", "B08", "B10", "B13"])
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=50000)
        b05, b03, b07, b08, b10, b13 = dask.compute(
            res["B05"].data, res["B03"].data, res["B07"].data,
            res["B08"].data, res["B10"].data, res["B13"].data)
        b05 = b05.astype(np.float32); b03 = b03.astype(np.float32)
        b07 = b07.astype(np.float32); b08 = b08.astype(np.float32)
        b10 = b10.astype(np.float32); b13 = b13.astype(np.float32)
        r = linear_normalize(b08 - b10, -35.0, 5.0)
        g = linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
        b = linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
        return r, g, b, None

    raise ValueError(f"Unsupported AHI composite: {composite_type}")