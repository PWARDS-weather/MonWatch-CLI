# =============================================================================
# rem_ingest/mtsat.py — MTSAT-1R / MTSAT-2 HRIT ingest for MonWatch-CLI
#
# (C) 2025-2026 PWARDS-weather
# SPDX-License-Identifier: Apache-2.0 OR GPL-3.0-or-later
#
# Dual-licensed. You may use, modify, and distribute this file under the terms
# of EITHER the Apache License, Version 2.0, or the GNU General Public License,
# Version 3.0 or later — not both. See LICENSE.txt in the project root for the
# full license texts and the copyright notice.
# =============================================================================
import os
import re
import glob
import gzip
import shutil
import ftplib
import tarfile
import logging
import datetime
import tempfile

import numpy as np
from satpy import Scene

from . import common
from .common import (
    resize_like, linear_normalize, false_color_rgb,
)

MTSAT_FTP_HOSTS = ["mtsat.cr.chiba-u.ac.jp", "gms.cr.chiba-u.ac.jp"]
MTSAT_FTP_ROOT = "/pub"

MTSAT_SAT_CONFIG = {
    "mtsat":  {"code": "MTSAT2", "dir": "MTSAT-2",
               "reader": "mtsat2-imager_hrit",
               "name": "MTSAT-2",  "lon": 145.0},
    "mtsat2": {"code": "MTSAT2", "dir": "MTSAT-2",
               "reader": "mtsat2-imager_hrit",
               "name": "MTSAT-2",  "lon": 145.0},
    "mtsat1": {"code": "MTSAT1", "dir": "MTSAT-1R",
               "reader": "jami_hrit",
               "name": "MTSAT-1R", "lon": 140.0},
}

AHI_TO_MTSAT = {
    1: "VIS", 2: "VIS", 3: "VIS", 4: "VIS",
    7: "IR4", 9: "IR3", 13: "IR1", 14: "IR2",
}

def _mtsat_cfg(sat_source):
    return MTSAT_SAT_CONFIG.get(sat_source) or MTSAT_SAT_CONFIG["mtsat"]


def _mtsat_remote_tar(cfg, dt_obj):
    return (f"{MTSAT_FTP_ROOT}/{cfg['dir']}/HRIT/"
            f"{dt_obj.year:04d}{dt_obj.month:02d}/{dt_obj.day:02d}/"
            f"HRIT_{cfg['code']}_{dt_obj:%Y%m%d%H%M}.tar")

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
            logging.info(f"MTSAT ({cfg['code']}): archive slot "
                         f"{probe:%Y-%m-%d %H:%M}Z -> {os.path.basename(tar)}")
            return {b: [tar] for b in bands_list}
        probe -= datetime.timedelta(hours=1)
    logging.warning(f"MTSAT archive tar not found for "
                    f"{dt_obj.strftime('%Y-%m-%d %H:%M')}Z")
    return {}


def download_mtsat_tar(tar_remote, local_path):
    for host in MTSAT_FTP_HOSTS:
        for attempt in range(3):
            try:
                logging.info(f"  Downloading {os.path.basename(tar_remote)} "
                             f"from {host} (attempt {attempt + 1}/3)")
                ftp = _mtsat_ftp(host)
                try:
                    with open(local_path, "wb") as fh:
                        ftp.retrbinary(f"RETR {tar_remote}",
                                       fh.write, blocksize=1024 * 1024)
                    sz = os.path.getsize(local_path) / (1024 * 1024)
                    logging.info(f"  Downloaded {os.path.basename(tar_remote)} "
                                 f"({sz:.1f} MB)")
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

    for tar_remote in tars:
        tar_local = os.path.join(tmpdir, os.path.basename(tar_remote))
        if not os.path.exists(tar_local):
            if not download_mtsat_tar(tar_remote, tar_local):
                logging.warning(f"MTSAT download failed for {tar_remote}")
                return None

        try:
            with tarfile.open(tar_local, "r:*") as tf:
                members = [m for m in tf.getmembers()
                           if m.isfile() and m.name.endswith(".gz")]
                for m in members:
                    m.name = os.path.basename(m.name)
                tf.extractall(tmpdir, members=members, filter="data")
            for gz_path in glob.glob(os.path.join(tmpdir, "*.gz")):
                plain = gz_path[:-3]
                try:
                    with gzip.open(gz_path, "rb") as f_in, \
                         open(plain, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
                    os.remove(gz_path)
                except Exception as e:
                    logging.warning(f"MTSAT gzip decompress failed for "
                                    f"{os.path.basename(gz_path)}: {e}")
        except Exception as e:
            logging.error(f"MTSAT extract failed for {tar_local}: {e}")
            return None

    hrit_files = [os.path.join(tmpdir, n) for n in os.listdir(tmpdir)
                  if n.startswith("HRIT_") and not n.endswith(".gz")]

    local = {}
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

def process_mtsat_data(local_files_map, target_area, target_dt,
                       composite_type, resample_type="nearest",
                       sat_source="mtsat"):
    all_files = []
    for paths in local_files_map.values():
        all_files.extend(paths)
    if not all_files:
        raise ValueError("No MTSAT local files")

    bands_needed = {
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
    if bands_needed is None:
        raise ValueError(f"Unsupported MTSAT composite: {composite_type}")
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
        if composite_type in ("sandwich", "irv"):
            return _native(3), _native(13), None, None
        if composite_type == "fire":
            return _native(7), None, None, None
        return _native(3), _native(13), None, None

    res = scn.resample(target_area, resampler=resample_type,
                       reduce_data=True, radius_of_influence=60000)

    def _read(ahi_band):
        arr = res[AHI_TO_MTSAT[ahi_band]].data.compute(scheduler="sync")
        return np.asarray(arr).astype(np.float32)

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
        vis_norm = np.clip(vis / 100.0 if np.nanmax(vis) > 1.0 else vis,
                           0.0, 1.0)
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
        result = np.clip(vis_corr * day_weight + ir_layer * night_weight,
                         0.0, 1.0)
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

    if composite_type == "fire":
        return _read(7), None, None, None

    raise ValueError(f"Unsupported MTSAT composite: {composite_type}")