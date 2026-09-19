# =============================================================================
# rem_ingest/jpss_common.py — shared JPSS/VIIRS helpers for MonWatch-CLI
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
import time
import gzip
import shutil
import tarfile
import logging
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

from .common import fs, _download_session

VIIRS_NORAD_IDS = {
    "n20": 43013, "n21": 54234, "snpp": 37849,
    "j01": 43013, "j02": 54234, "npp": 37849,
}
VIIRS_HALF_SWATH_KM = 1600.0
VIIRS_TLE_CACHE = "viirs_tle_cache.txt"
VIIRS_TLE_MAX_AGE = 86400

def _spacetrack_credentials():
    user = (os.environ.get("SPACETRACK_USERNAME") or "").strip()
    pw = (os.environ.get("SPACETRACK_PASSWORD") or "").strip()
    if user and pw:
        return user, pw
    return None, None

def _jpss_extract_orbit(name):
    m = re.search(r"_b(\d+)_", os.path.basename(name))
    return m.group(1) if m else None


def _jpss_parse_granule_time_from_name(name):
    base = os.path.basename(name)
    match = re.search(r'_d(\d{8})_t(\d{6})\d*_', base)
    if match:
        try:
            return datetime.datetime.strptime(
                f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
        except ValueError:
            pass
    match = re.search(r'_s(\d{14})', base)
    if match:
        try:
            return datetime.datetime.strptime(match.group(1)[:12], "%Y%m%d%H%M")
        except ValueError:
            pass
    return None

def _jpss_haversine_km(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _jpss_dist_to_box_km(sub_lats, sub_lons, lat, lon, crop_deg, asr=1.0):
    sub_lats = np.atleast_1d(sub_lats)
    sub_lons = np.atleast_1d(sub_lons)
    dlon = ((sub_lons - lon + 180.0) % 360.0) - 180.0
    lon_half = crop_deg * asr
    c_lats = np.clip(sub_lats, lat - crop_deg, lat + crop_deg)
    c_dlons = np.clip(dlon, -lon_half, lon_half)
    c_lons = lon + c_dlons
    return _jpss_haversine_km(sub_lats, sub_lons, c_lats, c_lons)

def _jpss_parse_tle_text(tle_text, target_catnr):
    lines = [ln.strip() for ln in tle_text.strip().splitlines() if ln.strip()]
    catnr_str = f"{int(target_catnr):05d}"
    for i in range(len(lines) - 2):
        l1, l2 = lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            if l1[2:7].strip() == catnr_str or l2[2:7].strip() == catnr_str:
                return lines[i].strip(), l1, l2
    return None


def _jpss_fetch_tle(norad_id):
    mirrors = [
        "https://bin.ssec.wisc.edu/pub/tle/weather.txt",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle",
        f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle",
    ]
    for url in mirrors:
        try:
            resp = requests.get(url, timeout=12)
            if resp.status_code == 200:
                parsed = _jpss_parse_tle_text(resp.text, norad_id)
                if parsed:
                    return parsed
        except Exception:
            continue
    raise ConnectionError(f"Could not retrieve TLE for NORAD {norad_id}")


def _jpss_fetch_tle_spacetrack(norad_id, dt_obj):
    user, pw = _spacetrack_credentials()
    if not user or not pw:
        raise RuntimeError("Space-Track credentials not configured")

    session = requests.Session()
    resp = session.post(
        "https://www.space-track.org/ajaxauth/login",
        data={"identity": user, "password": pw},
        timeout=15,
    )
    if resp.status_code != 200 or "Login Failed" in resp.text:
        raise RuntimeError(f"Space-Track login failed: HTTP {resp.status_code}")

    start_dt = (dt_obj - datetime.timedelta(days=4)).strftime("%Y-%m-%d")
    end_dt = (dt_obj + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"https://www.space-track.org/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{int(norad_id)}/EPOCH/{start_dt}--{end_dt}/"
        f"orderby/EPOCH desc/limit/1/format/3le"
    )
    resp = session.get(url, timeout=20)
    if resp.status_code == 200 and resp.text.strip():
        parsed = _jpss_parse_tle_text(resp.text, norad_id)
        if parsed:
            return parsed
        lines = [ln.strip() for ln in resp.text.strip().splitlines() if ln.strip()]
        if len(lines) >= 2 and lines[0].startswith("1 ") and lines[1].startswith("2 "):
            return f"NORAD_{int(norad_id):05d}", lines[0], lines[1]

    start_wide = (dt_obj - datetime.timedelta(days=14)).strftime("%Y-%m-%d")
    end_wide = (dt_obj + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
    url_wide = (
        f"https://www.space-track.org/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{int(norad_id)}/EPOCH/{start_wide}--{end_wide}/"
        f"orderby/EPOCH desc/limit/1/format/3le"
    )
    resp = session.get(url_wide, timeout=20)
    if resp.status_code == 200 and resp.text.strip():
        parsed = _jpss_parse_tle_text(resp.text, norad_id)
        if parsed:
            return parsed
    raise ConnectionError(
        f"Space-Track returned no TLE for NORAD {norad_id} near {dt_obj}")


def _jpss_get_orbital(sat_key, dt_obj=None):
    from pyorbital.orbital import Orbital
    norad_id = VIIRS_NORAD_IDS.get(str(sat_key).lower())
    if not norad_id:
        raise ValueError(f"No NORAD id for sat '{sat_key}'")

    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    is_historical = bool(dt_obj is not None
                         and abs((now - dt_obj).total_seconds()) > 86400 * 2)

    if is_historical:
        user, pw = _spacetrack_credentials()
        if user and pw:
            try:
                logging.info(f"JPSS TLE: Space-Track historical fetch for "
                             f"{sat_key} near {dt_obj.date()}")
                name, l1, l2 = _jpss_fetch_tle_spacetrack(norad_id, dt_obj)
                return Orbital(name, line1=l1, line2=l2)
            except Exception as e:
                logging.warning(f"JPSS TLE: Space-Track failed ({e}); "
                                f"falling back to recent TLE")
        else:
            logging.warning(
                "JPSS TLE: historical date requested but no Space-Track "
                "credentials (set star-trackuser= and star-trackpass= in "
                ".env or env.txt)"
            )

    cache = VIIRS_TLE_CACHE
    if os.path.exists(cache):
        age = time.time() - os.path.getmtime(cache)
        if age < VIIRS_TLE_MAX_AGE:
            try:
                with open(cache, "r") as f:
                    parsed = _jpss_parse_tle_text(f.read(), norad_id)
                if parsed:
                    name, l1, l2 = parsed
                    return Orbital(name, line1=l1, line2=l2)
            except Exception:
                pass

    name, l1, l2 = _jpss_fetch_tle(norad_id)
    try:
        existing = {}
        if os.path.exists(cache):
            with open(cache, "r") as f:
                content = f.read().strip().splitlines()
            for i in range(0, len(content) - 2, 3):
                if content[i + 1].startswith("1 ") and content[i + 2].startswith("2 "):
                    existing[content[i + 1][2:7].strip()] = (
                        content[i], content[i + 1], content[i + 2])
        existing[f"{int(norad_id):05d}"] = (name, l1, l2)
        with open(cache, "w") as f:
            for s_name, s_l1, s_l2 in existing.values():
                f.write(f"{s_name}\n{s_l1}\n{s_l2}\n")
    except OSError:
        pass
    return Orbital(name, line1=l1, line2=l2)

def _jpss_subsat_track(orb, times):
    try:
        lons, lats, _ = orb.get_lonlatalt(times)
        lons = np.atleast_1d(np.asarray(lons, dtype=np.float64))
        lats = np.atleast_1d(np.asarray(lats, dtype=np.float64))
        if len(lons) == len(times) and len(lats) == len(times):
            return lons, lats
    except Exception:
        pass
    lons = np.empty(len(times))
    lats = np.empty(len(times))
    for i, t in enumerate(times):
        lo, la, _ = orb.get_lonlatalt(t)
        lons[i], lats[i] = lo, la
    return lons, lats


def _jpss_find_passes(orb, dt_obj, lat, lon, crop_deg, asr=1.0,
                      search_window_hours=14, step_seconds=5):
    window = datetime.timedelta(hours=search_window_hours)
    start_t, end_t = dt_obj - window, dt_obj + window
    n = int((end_t - start_t).total_seconds() / step_seconds) + 1
    times = [start_t + datetime.timedelta(seconds=i * step_seconds)
             for i in range(n)]
    sub_lons, sub_lats = _jpss_subsat_track(orb, times)
    dists = _jpss_dist_to_box_km(sub_lats, sub_lons, lat, lon, crop_deg, asr=asr)
    hits = dists <= (VIIRS_HALF_SWATH_KM + 50.0)
    passes, in_pass, p0 = [], False, None
    for i, h in enumerate(hits):
        if h and not in_pass:
            in_pass, p0 = True, i
        elif not h and in_pass:
            in_pass = False
            passes.append((times[p0], times[i - 1]))
    if in_pass:
        passes.append((times[p0], times[-1]))
    pad = datetime.timedelta(seconds=90)
    return [(max(start_t, s - pad), min(end_t, e + pad)) for s, e in passes]


def _jpss_rank_passes(orb, passes, dt_obj, lat, lon, crop_deg, asr=1.0,
                      max_passes=6, min_coverage=0.97):
    if not passes:
        return []
    lat_min, lat_max = lat - crop_deg, lat + crop_deg
    lon_half = crop_deg * asr
    grid_lats = np.linspace(lat_min, lat_max, 15)
    grid_lons = np.linspace(lon - lon_half, lon + lon_half, 15)
    glat, glon = np.meshgrid(grid_lats, grid_lons)
    glat_f, glon_f = glat.flatten(), glon.flatten()
    total = len(glat_f)

    metrics = []
    for p_start, p_end in passes:
        dur = max(30.0, (p_end - p_start).total_seconds())
        n = max(3, int(dur / 15.0) + 1)
        samples = [p_start + datetime.timedelta(seconds=15.0 * i)
                   for i in range(n)]
        sub_lons, sub_lats = _jpss_subsat_track(orb, samples)
        center_dist = float(np.min(_jpss_haversine_km(sub_lats, sub_lons,
                                                      lat, lon)))
        covers_center = center_dist <= VIIRS_HALF_SWATH_KM
        p_mid = p_start + (p_end - p_start) / 2
        time_diff = abs((p_mid - dt_obj).total_seconds())
        covered = np.zeros(total, dtype=bool)
        for s_lat, s_lon in zip(sub_lats, sub_lons):
            covered |= (_jpss_haversine_km(s_lat, s_lon, glat_f, glon_f)
                        <= VIIRS_HALF_SWATH_KM)
        metrics.append({
            "pass": (p_start, p_end),
            "covers_center": covers_center,
            "time_diff": time_diff,
            "covered": covered,
            "pct": float(np.sum(covered) / total),
        })

    metrics.sort(key=lambda m: (0 if m["covers_center"] else 1, m["time_diff"]))
    selected, cum = [], np.zeros(total, dtype=bool)
    for m in metrics:
        new = int(np.sum(m["covered"] & ~cum))
        if len(selected) == 0 or new > 0:
            selected.append(m["pass"])
            cum |= m["covered"]
            if (float(np.sum(cum) / total) >= min_coverage
                    or len(selected) >= max_passes):
                break
    return selected

def _jpss_find_h5_dataset(h5_group, name_suffix):
    found = {}

    def _visitor(name, obj):
        if "ds" not in found and hasattr(obj, "shape") \
                and name.split("/")[-1] == name_suffix:
            found["ds"] = obj

    try:
        h5_group.visititems(_visitor)
    except Exception:
        pass
    return found.get("ds")


def _jpss_check_geo_coverage(bucket, geo_key, lat, lon, crop_deg,
                             asr=1.0, subsample=8):
    try:
        import h5py
    except ImportError:
        return True
    path = f"{bucket}/{geo_key}"
    try:
        if fs is None:
            logging.debug("JPSS GEO check: no s3fs; accepting candidate")
            return True
        with fs.open(path, "rb") as remote_f:
            with h5py.File(remote_f, "r") as hf:
                lat_ds = _jpss_find_h5_dataset(hf, "Latitude")
                lon_ds = _jpss_find_h5_dataset(hf, "Longitude")
                if lat_ds is None or lon_ds is None:
                    return False
                lats = lat_ds[::subsample, ::subsample].astype(np.float64)
                lons = lon_ds[::subsample, ::subsample].astype(np.float64)
        valid = (lats >= -90) & (lats <= 90) & (lons >= -180) & (lons <= 360)
        if not np.any(valid):
            return False
        lat_min, lat_max = lat - crop_deg, lat + crop_deg
        lon_norm = ((lon + 180) % 360) - 180
        lons_norm = ((lons + 180) % 360) - 180
        lon_diff = np.abs(((lons_norm - lon_norm + 180) % 360) - 180)
        lon_crop = crop_deg * asr
        hit = valid & (lats >= lat_min) & (lats <= lat_max) & (lon_diff <= lon_crop)
        return bool(np.any(hit))
    except Exception as e:
        logging.debug(f"JPSS GEO coverage check failed {geo_key}: {e}")
        return False

def _jpss_pds_list_keys(bucket, prefix, max_keys=2000):
    import urllib.parse
    import xml.etree.ElementTree as ET
    keys, token = [], None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params = {"list-type": "2", "prefix": prefix,
                  "max-keys": str(min(max_keys, 1000))}
        if token:
            params["continuation-token"] = token
        url = f"https://{bucket}.s3.amazonaws.com/?{urllib.parse.urlencode(params)}"
        try:
            resp = _download_session.get(url, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            logging.warning(f"JPSS PDS list failed s3://{bucket}/{prefix}: {e}")
            break
        for c in root.findall("s3:Contents", ns):
            k = c.find("s3:Key", ns)
            if k is not None and k.text:
                keys.append(k.text)
        truncated = root.find("s3:IsTruncated", ns)
        if truncated is not None and truncated.text == "true":
            nt = root.find("s3:NextContinuationToken", ns)
            token = nt.text if nt is not None else None
            if not token or len(keys) >= max_keys:
                break
        else:
            break
    return keys

def jpss_download_tar(url, local_path, retries=3):
    import threading
    thread_name = threading.current_thread().name
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"{thread_name}: downloading JPSS TAR "
                         f"{os.path.basename(local_path)} "
                         f"(attempt {attempt}/{retries})")
            resp = _download_session.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            sz = os.path.getsize(local_path) / (1024 * 1024)
            logging.info(f"{thread_name}: downloaded "
                         f"{os.path.basename(local_path)} ({sz:.1f} MB)")
            return True
        except Exception as e:
            logging.warning(f"{thread_name}: JPSS download attempt "
                            f"{attempt} failed: {e}")
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
                            with gzip.open(p, "rb") as fi, \
                                 open(plain, "wb") as fo:
                                shutil.copyfileobj(fi, fo, length=1024 * 1024)
                            os.remove(p)
                            extracted[-1] = plain
                        except Exception as e:
                            logging.warning(f"JPSS gzip decompress failed for "
                                            f"{p}: {e}")
    except Exception as e:
        logging.error(f"JPSS extract failed for {tar_path}: {e}")
        return []
    logging.info(f"JPSS extracted {len(extracted)} file(s) from "
                 f"{os.path.basename(tar_path)}")
    return extracted


def jpss_select_closest_files(file_paths, target_dt, max_files=12):
    scored = []
    for p in file_paths:
        gt = _jpss_parse_granule_time_from_name(p)
        if gt is None:
            score = 1e12
        else:
            score = (-gt.timestamp() if target_dt is None
                     else abs((gt - target_dt).total_seconds()))
        scored.append((score, gt, p))
    scored.sort(key=lambda x: x[0])
    selected = [p for _, _, p in scored[:max_files]]
    if scored and scored[0][1] is not None:
        logging.info(f"JPSS closest granule time: "
                     f"{scored[0][1].strftime('%Y-%m-%d %H:%M')}Z "
                     f"(requested "
                     f"{target_dt.strftime('%Y-%m-%d %H:%M') + 'Z' if target_dt else 'latest'})")
    return selected