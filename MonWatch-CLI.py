#!/usr/bin/env python3
# =============================================================================
#
# MonWatch-CLI -- Automated Himawari-9/8, GK-2A, GOES, MTG, MTSAT Storm Imagery
# A CLI companion to MonWatch-UI for servers and automated systems.
# Part of the PWARDS ecosystem -- "Free Science For Everyone".
#
# (C) 2025-2026 PWARDS-weather
#
# SPDX-License-Identifier: Apache-2.0 OR GPL-3.0-or-later
#
# This project is dual-licensed. You may use, modify, and distribute this
# software under the terms of EITHER:
#
#   * the Apache License, Version 2.0
#     <http://www.apache.org/licenses/LICENSE-2.0>, OR
#   * the GNU General Public License, Version 3.0 or later
#     <https://www.gnu.org/licenses/gpl-3.0.txt>.
#
# You are not required to comply with both -- choose whichever license suits
# your use. See LICENSE.txt in the project root for the full text of both
# licenses and the copyright notice.
#
# ---------------------------------------------------------------------------
# A request (not a legal condition) from the maintainers:
#   If you intend to modify, redistribute, or build on MonWatch-CLI, we would
#   appreciate a heads-up so we can coordinate upstream changes. Please open
#   an issue or PR at <https://github.com/PWARDS-weather/MonWatch-UI> or mail
#   pwards.sci@gmail.com. This is a courtesy, not a license term -- you are
#   free to use, modify, and redistribute under Apache-2.0 or GPL-3.0 without
#   notifying anyone.
# ---------------------------------------------------------------------------
#
# =============================================================================

# =============================================================================
# PLANS / TODO
#
# * 2-D height map
#     Replace _cold_centroid with a sliding-window FFT phase-correlation between
#     Himawari and GK-2A tiles (np.fft.fft2 on 64x64 blocks). Each block's peak
#     yields a local (dx, dy) that feeds _beyev_height_from_disparity per pixel.
#
# * Better physics
#     Swap the spherical Earth model for an oblate one via
#     pyproj.Proj(proj="geos", ...). The forward/inverse model becomes a couple
#     of pyproj.transform calls instead of ray-sphere math; the algorithm is
#     otherwise unchanged.
#
# * Tie the disparity to cloud-top temperature
# =============================================================================

import os
import sys
import json
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
import time
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

from rem_ingest import common as _ri_common

from rem_ingest import himawari as _ri_himawari

discover_ahi_files           = _ri_himawari.discover_ahi_files
get_latest_available_dt      = _ri_himawari.get_latest_available_dt
download_and_decompress_all  = _ri_himawari.download_and_decompress_all
prefetch_all_slots           = _ri_himawari.prefetch_all_slots
process_monwatch_ahi_data      = _ri_himawari.process_ahi_data
_native_target_area          = _ri_himawari._native_target_area

_SANDWICH_IR_LUT             = _ri_common._SANDWICH_IR_LUT
_DVORAK_IR_LUT               = _ri_common._DVORAK_IR_LUT
_sandwich_ir_lookup          = _ri_common.sandwich_ir_lookup
_dvorak_ir_lookup            = _ri_common.dvorak_ir_lookup
_dvorak_cmap                 = _ri_common.dvorak_cmap
_linear_normalize            = _ri_common.linear_normalize
_normalize_reflectance       = _ri_common.normalize_reflectance
_stack_rgb                   = _ri_common.stack_rgb
_false_color_rgb             = _ri_common.false_color_rgb
_resize_like                 = _ri_common.resize_like
_reduced_area                = _ri_common.reduced_area
_upscale_rgb                 = _ri_common.upscale_rgb
get_required_segments        = _ri_common.get_required_segments
generate_time_slots          = _ri_common.generate_time_slots
_project_is_native_geos      = _ri_common.project_is_native_geos
_geos_area_for_crop          = _ri_common.geos_area_for_crop
_standard_fulldisk_geos_area = _ri_common.standard_fulldisk_geos_area
_area_with_shape             = _ri_common.area_with_shape
_align_result_to_area        = _ri_common.align_result_to_area
_sat_subpoint_lon            = _ri_common.sat_subpoint_lon
_normalize_basin             = _ri_common.normalize_basin
_storm_in_wpac               = _ri_common.storm_in_wpac
AHI_TO_ABI                   = _ri_common.AHI_TO_ABI
AHI_TO_FCI                   = _ri_common.AHI_TO_FCI
hotspot_SIR_nodes            = _ri_common.hotspot_SIR_nodes
INFRARED_HIM_nodes           = _ri_common.INFRARED_HIM_nodes
OTT_nodes                    = _ri_common.OTT_nodes
OTT2_nodes                   = _ri_common.OTT2_nodes
DVORAK_nodes                 = _ri_common.DVORAK_nodes

from rem_ingest import gk2a as _ri_gk2a

discover_gk2a_files         = _ri_gk2a.discover_gk2a_files
get_latest_available_dt_gk2a = _ri_gk2a.get_latest_available_dt_gk2a
download_gk2a_file          = _ri_gk2a.download_gk2a_file
download_gk2a_files         = _ri_gk2a.download_gk2a_files
prefetch_all_slots_gk2a     = _ri_gk2a.prefetch_all_slots_gk2a
process_gk2a_data           = _ri_gk2a.process_gk2a_data

GK2A_BUCKET                  = _ri_gk2a.GK2A_BUCKET
GK2A_HTTPS_BASE              = _ri_gk2a.GK2A_HTTPS_BASE
GK2A_DEFAULT_GRID_COLOR      = _ri_gk2a.GK2A_DEFAULT_GRID_COLOR
GK2A_DEFAULT_COASTLINE_COLOR = _ri_gk2a.GK2A_DEFAULT_COASTLINE_COLOR
GK2A_CHANNEL_BAND            = _ri_gk2a.GK2A_CHANNEL_BAND
GK2A_BAND_CHANNEL            = _ri_gk2a.GK2A_BAND_CHANNEL
GK2A_CHANNEL_CWL             = _ri_gk2a.GK2A_CHANNEL_CWL
GK2A_IR_CHANNELS             = _ri_gk2a.GK2A_IR_CHANNELS

from rem_ingest import goes as _ri_goes

discover_goes_files          = _ri_goes.discover_goes_files
get_latest_available_dt_goes = _ri_goes.get_latest_available_dt_goes
download_goes_file           = _ri_goes.download_goes_file
download_goes_files          = _ri_goes.download_goes_files
_goes_read_band              = _ri_goes._goes_read_band
_global_read_goes            = _ri_goes._global_read_goes
_process_goes_storm_data     = _ri_goes.process_goes_data
GOES_SATELLITE_MAP           = _ri_goes.GOES_SATELLITE_MAP
GOES_DEFAULT                 = _ri_goes.GOES_DEFAULT
_resolve_goes_source         = _ri_goes._resolve_goes_source
_goes_candidate_buckets      = _ri_goes._goes_candidate_buckets

from rem_ingest import mtg as _ri_mtg

MTG_COLLECTION               = _ri_mtg.MTG_COLLECTION
_load_eumetsat_creds         = _ri_mtg._load_eumetsat_creds
_ensure_mtg_setup            = _ri_mtg._ensure_mtg_setup
_eumetsat_creds_available    = _ri_mtg.eumetsat_creds_available
_download_product_with_retry = _ri_mtg._download_product_with_retry
_read_mtg_tailored_channel   = _ri_mtg._read_mtg_tailored_channel
discover_mtg_files           = _ri_mtg.discover_mtg_files
_global_read_mtg             = _ri_mtg._global_read_mtg
_process_mtg_storm_data      = _ri_mtg.process_mtg_data

FCI_CHANNEL_WAVELENGTH_UM    = _ri_mtg.FCI_CHANNEL_WAVELENGTH_UM
FCI_IR_CHANNELS              = _ri_mtg.FCI_IR_CHANNELS
FCI_CHANNEL_SSD              = _ri_mtg.FCI_CHANNEL_SSD

from rem_ingest import mtsat as _ri_mtsat

MTSAT_FTP_HOSTS              = _ri_mtsat.MTSAT_FTP_HOSTS
MTSAT_FTP_ROOT               = _ri_mtsat.MTSAT_FTP_ROOT
MTSAT_SAT_CONFIG             = _ri_mtsat.MTSAT_SAT_CONFIG
AHI_TO_MTSAT                 = _ri_mtsat.AHI_TO_MTSAT

_mtsat_cfg                   = _ri_mtsat._mtsat_cfg
_mtsat_ftp                   = _ri_mtsat._mtsat_ftp
_mtsat_remote_tar            = _ri_mtsat._mtsat_remote_tar
mtsat_tar_exists             = _ri_mtsat.mtsat_tar_exists
discover_mtsat_files         = _ri_mtsat.discover_mtsat_files
download_mtsat_tar           = _ri_mtsat.download_mtsat_tar
download_mtsat_tars          = _ri_mtsat.download_mtsat_tars
_process_mtsat_storm_data    = _ri_mtsat.process_mtsat_data

from rem_ingest import jpss as _ri_jpss

JPSS_CLASS_BASE              = _ri_jpss.JPSS_CLASS_BASE
JPSS_PDS_BUCKETS             = _ri_jpss.JPSS_PDS_BUCKETS
JPSS_PDS_SAT_TOKEN           = _ri_jpss.JPSS_PDS_SAT_TOKEN
JPSS_PDS_SAT_LABEL           = _ri_jpss.JPSS_PDS_SAT_LABEL
JPSS_PDS_BUCKET_TO_KEY       = _ri_jpss.JPSS_PDS_BUCKET_TO_KEY
JPSS_PDS_COMPOSITE_BANDS     = _ri_jpss.JPSS_PDS_COMPOSITE_BANDS
JPSS_FAMILY_DEFAULT_PRODUCTS = _ri_jpss.JPSS_FAMILY_DEFAULT_PRODUCTS
VIIRS_NORAD_IDS              = _ri_jpss.VIIRS_NORAD_IDS
VIIRS_HALF_SWATH_KM          = _ri_jpss.VIIRS_HALF_SWATH_KM
VIIRS_TLE_CACHE              = _ri_jpss.VIIRS_TLE_CACHE
VIIRS_TLE_MAX_AGE            = _ri_jpss.VIIRS_TLE_MAX_AGE
VIIRS_PRODUCT_INFO           = _ri_jpss.VIIRS_PRODUCT_INFO

_jpss_pds_resolve_bucket     = _ri_jpss._jpss_pds_resolve_bucket
_jpss_pds_bucket_candidates  = _ri_jpss._jpss_pds_bucket_candidates
_jpss_extract_orbit          = _ri_jpss._jpss_extract_orbit
_jpss_pds_list_keys          = _ri_jpss._jpss_pds_list_keys
_jpss_haversine_km           = _ri_jpss._jpss_haversine_km
_jpss_dist_to_box_km         = _ri_jpss._jpss_dist_to_box_km
_jpss_parse_tle_text         = _ri_jpss._jpss_parse_tle_text
_jpss_fetch_tle              = _ri_jpss._jpss_fetch_tle
_jpss_fetch_tle_spacetrack   = _ri_jpss._jpss_fetch_tle_spacetrack
_jpss_get_orbital            = _ri_jpss._jpss_get_orbital
_jpss_subsat_track           = _ri_jpss._jpss_subsat_track
_jpss_find_passes            = _ri_jpss._jpss_find_passes
_jpss_rank_passes            = _ri_jpss._jpss_rank_passes
_jpss_find_h5_dataset        = _ri_jpss._jpss_find_h5_dataset
_jpss_check_geo_coverage     = _ri_jpss._jpss_check_geo_coverage
_jpss_list_dir               = _ri_jpss._jpss_list_dir
_jpss_parse_granule_time_from_name = _ri_jpss._jpss_parse_granule_time_from_name
_jpss_satpy_reader_for_files = _ri_jpss._jpss_satpy_reader_for_files

jpss_list_available_dates    = _ri_jpss.jpss_list_available_dates
jpss_closest_date            = _ri_jpss.jpss_closest_date
jpss_list_families           = _ri_jpss.jpss_list_families
jpss_list_products           = _ri_jpss.jpss_list_products
jpss_list_sats               = _ri_jpss.jpss_list_sats
jpss_list_tars               = _ri_jpss.jpss_list_tars
jpss_download_tar            = _ri_jpss.jpss_download_tar
jpss_extract_tar             = _ri_jpss.jpss_extract_tar
jpss_select_closest_files    = _ri_jpss.jpss_select_closest_files
jpss_resolve_product         = _ri_jpss.jpss_resolve_product
jpss_resolve_sat             = _ri_jpss.jpss_resolve_sat
discover_jpss_files          = _ri_jpss.discover_jpss_files
download_jpss_and_extract    = _ri_jpss.download_jpss_and_extract
discover_jpss_pds_files      = _ri_jpss.discover_jpss_pds_files
download_jpss_pds_files      = _ri_jpss.download_jpss_pds_files
process_jpss_data            = _ri_jpss.process_jpss_data
process_jpss_storm           = _ri_jpss.process_jpss_storm

from gro_ingest import garbinradar as _gi_garbin

_garbin_dbz_cmap             = _gi_garbin.dbz_cmap
_garbin_identity             = _gi_garbin.identity
_garbin_time_reference       = _gi_garbin.time_reference
_garbin_candidate_timestamps = _gi_garbin.candidate_timestamps
_fetch_garbin_radar          = _gi_garbin.fetch_radar
_garbin_radar_bytes          = _gi_garbin.radar_bytes
_garbin_radar_overlay        = _gi_garbin.radar_overlay
process_garbin_radar         = _gi_garbin.process_radar

GARBIN_RADAR_BASE            = _gi_garbin.GARBIN_RADAR_BASE
GARBIN_RADAR_BOUNDS          = _gi_garbin.GARBIN_RADAR_BOUNDS
GARBIN_USER_AGENT            = _gi_garbin.GARBIN_USER_AGENT
GARBIN_BROWSER_HEADERS       = _gi_garbin.GARBIN_BROWSER_HEADERS
HEX_COLORS_DBZ               = _gi_garbin.HEX_COLORS_DBZ

from gro_ingest import phradar as _gi_phradar

_phradar_colorize_la         = _gi_phradar.colorize_la
_phradar_fetch_image_bytes   = _gi_phradar.fetch_image_bytes
_phradar_radar_bytes         = _gi_phradar.radar_bytes
_phradar_radar_overlay       = _gi_phradar.radar_overlay

PHRADAR_ORIGIN               = _gi_phradar.PHRADAR_ORIGIN
PHRADAR_BOUNDS               = _gi_phradar.PHRADAR_BOUNDS
PHRADAR_USER_AGENT           = _gi_phradar.PHRADAR_USER_AGENT

def _load_args_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_parser_from_config(config):
    parser = argparse.ArgumentParser(
        description=config.get("description", "MonWatch-CLI")
    )

    group_objs = {}
    for gdef in config.get("groups", []):
        name = gdef["name"]
        if gdef.get("exclusive"):
            group_objs[name] = parser.add_mutually_exclusive_group(
                required=gdef.get("required", False)
            )
        else:
            group_objs[name] = parser.add_argument_group(gdef.get("title", name))

    def _mk_type(tname):
        return {
            "str": str,
            "int": int,
            "float": float,
            "upper": lambda s: s.upper(),
            "lower": lambda s: s.lower(),
        }.get(tname, str)

    for adef in config["arguments"]:
        kwargs = {}
        if "help" in adef:
            kwargs["help"] = adef["help"]

        atype = adef.get("type", "str")
        if atype == "bool":
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = _mk_type(atype)
            if "default" in adef:
                kwargs["default"] = adef["default"]
            if "choices" in adef:
                kwargs["choices"] = adef["choices"]
            if "nargs" in adef:
                kwargs["nargs"] = adef["nargs"]
            if "const" in adef:
                kwargs["const"] = adef["const"]

        if "dest" in adef:
            kwargs["dest"] = adef["dest"]

        flags = adef["flags"]
        if isinstance(flags, str):
            flags = [flags]

        target = group_objs[adef["group"]] if "group" in adef else parser
        target.add_argument(*flags, **kwargs)

    return parser

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

def _beyev_find_common_slot(him_segs, date_str=None, time_str=None,
                            max_back=6):
    def _probe(dt):
        him = discover_ahi_files("noaa-himawari9", dt, [13], him_segs)
        if not him:
            him = discover_ahi_files("noaa-himawari8", dt, [13], him_segs)
        gk2a = discover_gk2a_files(dt, [13])
        return (him or None), (gk2a or None)

    if time_str:
        d = date_str if date_str else datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y%m%d")
        try:
            dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
        except ValueError:
            logging.error(f"BEYEV: invalid --date {date_str} / --time {time_str}")
            return None, None, None
        him, gk2a = _probe(dt)
        if him and gk2a:
            return dt, him, gk2a
        logging.error(f"BEYEV: no common slot at requested "
                      f"{dt:%Y-%m-%d %H:%M}Z")
        return None, None, None

    candidates = []
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    base = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
    for i in range(max_back + 1):
        candidates.append(base - datetime.timedelta(minutes=10 * i))

    him_latest = (get_latest_available_dt("noaa-himawari9", him_segs, [13])
                  or get_latest_available_dt("noaa-himawari8", him_segs, [13]))
    gk2a_latest = get_latest_available_dt_gk2a([13])
    for extra in (him_latest, gk2a_latest):
        if extra is not None and extra not in candidates:
            candidates.append(extra)

    candidates.sort(reverse=True)
    logging.info(f"BEYEV: probing {len(candidates)} candidate slot(s) for a "
                 f"common Himawari+GK-2A timestamp...")
    for dt in candidates:
        him, gk2a = _probe(dt)
        if him and gk2a:
            logging.info(f"BEYEV: common slot found at {dt:%Y-%m-%d %H:%M}Z")
            return dt, him, gk2a
        logging.debug(f"  {dt:%Y-%m-%d %H:%M}Z: "
                      f"HIM={'ok' if him else '--'}  "
                      f"GK2A={'ok' if gk2a else '--'}")

    logging.error(f"BEYEV: no common Himawari/GK-2A slot within "
                  f"~{max_back * 10 + 10} minutes of now")
    return None, None, None


def _paired_percentile_stretch(a, b, lo_pct=2.0, hi_pct=98.0):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_fin = a[np.isfinite(a)]
    b_fin = b[np.isfinite(b)]
    joined = np.concatenate([a_fin.ravel(), b_fin.ravel()])
    if joined.size == 0:
        return a, b
    lo = float(np.percentile(joined, lo_pct))
    hi = float(np.percentile(joined, hi_pct))
    if hi <= lo:
        return a, b
    a2 = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    b2 = np.clip((b - lo) / (hi - lo), 0.0, 1.0)
    logging.debug(f"BEYEV stretch: lo={lo:.1f}K hi={hi:.1f}K "
                  f"(p{lo_pct:.0f}/p{hi_pct:.0f} of joint)")
    return a2, b2











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










_WPAC_LAT_MIN, _WPAC_LAT_MAX = -3.0, 30.0
_WPAC_LON_MIN, _WPAC_LON_MAX = 100.0, 180.0

_GEOS_R_KM = 6371.0
_GEOS_H_KM = 35786.0

def _beyev_geos_apparent(lat, lon, h_km, sat_lon):
    R, H = _GEOS_R_KM, _GEOS_H_KM
    lon_rel = np.radians(lon - sat_lon)
    lat_r   = np.radians(lat)
    r_c     = R + h_km

    xc = r_c * np.cos(lat_r) * np.cos(lon_rel)
    yc = r_c * np.cos(lat_r) * np.sin(lon_rel)
    zc = r_c * np.sin(lat_r)
    xs, ys, zs = R + H, 0.0, 0.0
    dx, dy, dz = xc - xs, yc - ys, zc - zs

    a = dx*dx + dy*dy + dz*dz
    b = 2.0 * (xs*dx + ys*dy + zs*dz)
    c = xs*xs + ys*ys + zs*zs - R*R
    disc = b*b - 4.0*a*c
    if disc < 0.0:
        return None

    t = (-b - np.sqrt(disc)) / (2.0 * a)
    xi, yi, zi = xs + t*dx, ys + t*dy, zs + t*dz
    lat_a = np.degrees(np.arcsin(np.clip(zi / R, -1.0, 1.0)))
    lon_a = np.degrees(np.arctan2(yi, xi)) + sat_lon
    return lat_a, lon_a


def _beyev_disparity_km(lat, lon, h_km,
                        him_sat_lon=140.7, gk2a_sat_lon=128.2):
    p_him  = _beyev_geos_apparent(lat, lon, h_km, him_sat_lon)
    p_gk2a = _beyev_geos_apparent(lat, lon, h_km, gk2a_sat_lon)
    if p_him is None or p_gk2a is None:
        return None
    lat_h, lon_h = p_him
    lat_g, lon_g = p_gk2a
    cos_lat = np.cos(np.radians(0.5 * (lat_h + lat_g)))
    dlon = ((lon_g - lon_h + 180.0) % 360.0) - 180.0
    return dlon * 111.32 * cos_lat


def _beyev_height_from_disparity(disparity_km, lat, lon,
                                 him_sat_lon=140.7, gk2a_sat_lon=128.2,
                                 h_max_km=20.0, n=81):
    hs = np.linspace(0.0, h_max_km, n)
    disps = np.array([
        _beyev_disparity_km(lat, lon, h, him_sat_lon, gk2a_sat_lon) or 0.0
        for h in hs
    ])
    order = np.argsort(disps)
    return float(np.interp(disparity_km, disps[order], hs[order]))

def _beyev_stereo_chart(him_ir, gk2a_ir, area_extent, dt, storm_id,
                        out_dir, center_lat, center_lon,
                        him_sat_lon=140.7, gk2a_sat_lon=128.2,
                        export_formats=None):
    import matplotlib.pyplot as plt

    h, w = him_ir.shape
    if gk2a_ir.shape != (h, w):
        gk2a_ir = _resize_like(gk2a_ir, (h, w))

    x0, y0, x1, y1 = area_extent
    lons = np.linspace(x0, x1, w)
    lats = np.linspace(y1, y0, h)

    row_fracs = [0.35, 0.50, 0.65]
    row_idx   = [int(rf * (h - 1)) for rf in row_fracs]

    fig, axes = plt.subplots(
        nrows=4, ncols=1, figsize=(12, 4 + 3 * len(row_idx)),
        facecolor="#0a0a0a",
        gridspec_kw={"height_ratios": [1, 1, 1, 1.25]},
    )

    HIM_C  = "#ff3344"
    GK2A_C = "#3388ff"

    def _cold_centroid(profile):
        tb = np.where(np.isfinite(profile), profile, 300.0)
        wgt = np.clip(300.0 - tb, 0.0, 200.0)
        if wgt.sum() < 1e-6:
            return None
        return float(np.average(np.arange(len(tb)), weights=wgt))

    derived = []

    for ax, ri in zip(axes[:len(row_idx)], row_idx):
        him_line  = him_ir[ri, :]
        gk2a_line = gk2a_ir[ri, :]

        him_v  = np.isfinite(him_line)
        gk2a_v = np.isfinite(gk2a_line)

        ax.plot(lons[him_v],  him_line[him_v]  - 273.15,
                color=HIM_C,  lw=1.2, label="Himawari-9  B13")
        ax.plot(lons[gk2a_v], gk2a_line[gk2a_v] - 273.15,
                color=GK2A_C, lw=1.2, label="GK-2A  ir105")

        ax.set_facecolor("#0a0a0a")
        ax.tick_params(colors="white", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#333333")
        ax.grid(True, color="#222", lw=0.5)
        ax.set_ylabel("BT (°C)", color="white", fontsize=9)

        lat_row = lats[ri]
        ax.text(0.01, 0.95, f"row {ri}  (lat {lat_row:+.1f}°)",
                transform=ax.transAxes, color="white", fontsize=9,
                va="top", bbox=dict(facecolor="black", alpha=0.55, pad=3))

        c_him  = _cold_centroid(him_line)
        c_gk2a = _cold_centroid(gk2a_line)
        if c_him is not None and c_gk2a is not None:
            km_per_px = ((x1 - x0) * 111.32
                         * np.cos(np.radians(lat_row)) / max(w - 1, 1))
            disparity_km = (c_gk2a - c_him) * km_per_px
            h_est = _beyev_height_from_disparity(
                disparity_km, center_lat, center_lon,
                him_sat_lon=him_sat_lon, gk2a_sat_lon=gk2a_sat_lon)
            derived.append(h_est)
            ax.text(0.99, 0.95,
                    f"Δx={disparity_km:+.1f} km   →   h ≈ {h_est:.1f} km",
                    transform=ax.transAxes, color="#ffcc00", fontsize=9,
                    va="top", ha="right",
                    bbox=dict(facecolor="black", alpha=0.55, pad=3))
        else:
            ax.text(0.99, 0.95, "no valid IR on this row",
                    transform=ax.transAxes, color="#888", fontsize=9,
                    va="top", ha="right")
        ax.legend(loc="lower right", facecolor="#111", edgecolor="#333",
                  labelcolor="white", fontsize=8, framealpha=0.85)

    ax_par = axes[-1]
    ax_par.set_facecolor("#0a0a0a")
    hs = np.linspace(0.0, 20.0, 121)
    curve = np.array([
        _beyev_disparity_km(center_lat, center_lon, hh,
                            him_sat_lon, gk2a_sat_lon) or 0.0
        for hh in hs
    ])
    ax_par.plot(hs, curve, color="#ffcc00", lw=2.0, label="Him − GK-2A parallax")
    ax_par.axhline(0, color="#555", lw=0.5)
    ax_par.set_xlabel("Cloud-top height (km)", color="white", fontsize=10)
    ax_par.set_ylabel("E–W disparity (km)",  color="white", fontsize=10)
    ax_par.tick_params(colors="white", labelsize=9)
    for s in ax_par.spines.values():
        s.set_color("#333333")
    ax_par.grid(True, color="#222", lw=0.5)
    for k, h_est in enumerate(derived):
        ax_par.axvline(h_est, color="#22ff88", lw=0.9, alpha=0.6, ls="--")
        ax_par.text(h_est, ax_par.get_ylim()[1] * 0.9,
                    f"  row {k}", color="#22ff88", fontsize=8,
                    rotation=90, va="top")
    ax_par.legend(loc="lower right", facecolor="#111",
                  edgecolor="#333", labelcolor="white", fontsize=8)

    fig.suptitle(
        f"BEYEV stereoscopic profiles — {storm_id} — "
        f"{dt.strftime('%Y-%m-%d %H:%M UTC')}",
        color="white", fontsize=13, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    base = os.path.join(
        out_dir, f"{storm_id}_{dt.strftime('%Y%m%d_%H%M')}_beyev_stereo_chart")
    png_path = base + ".png"
    fig.savefig(png_path, facecolor=fig.get_facecolor(),
                bbox_inches="tight", dpi=140)
    plt.close(fig)
    logging.info(f"BEYEV: stereo chart -> {png_path}")

    if export_formats:
        try:
            with Image.open(png_path) as img:
                for fmt in export_formats:
                    fmt = fmt.lower().strip()
                    out_fmt = f"{base}.{fmt}"
                    try:
                        if fmt == "avif":
                            img.save(out_fmt, format="AVIF",
                                     quality=95, subsampling="4:4:4")
                        elif fmt == "png":
                            continue
                        elif fmt in ("jpg", "jpeg"):
                            img.save(out_fmt, format="JPEG",
                                     quality=95, subsampling=0)
                        elif fmt == "webp":
                            img.save(out_fmt, format="WEBP",
                                     quality=95, method=6)
                        else:
                            continue
                        logging.info(f"BEYEV: stereo chart -> {out_fmt}")
                    except OSError as e:
                        logging.warning(f"BEYEV stereo chart {fmt} failed: {e}")
        except Exception as e:
            logging.warning(f"BEYEV stereo chart export failed: {e}")

    return base

def _filter_storms_by_basin(storms, basin_filter):
    if not basin_filter:
        return list(storms)
    return [s for s in storms if _normalize_basin(s.get("basin", "")) in basin_filter]

def process_monwatch_ahi_data(local_files_map, target_area, target_dt,
                            composite_type, resample_type="nearest",
                            sat_source="him"):
    if sat_source == "gk2a":
        return process_gk2a_data(local_files_map, target_area, target_dt,
                                 composite_type, resample_type)
    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        return _process_goes_storm_data(local_files_map, target_area, target_dt,
                                        composite_type, resample_type)
    if sat_source == "mtg":
        return _process_mtg_storm_data(local_files_map, target_area, target_dt,
                                       composite_type, resample_type)
    if sat_source in MTSAT_SAT_CONFIG:
        return _process_mtsat_storm_data(local_files_map, target_area, target_dt,
                                         composite_type, resample_type,
                                         sat_source=sat_source)
    return _ri_himawari.process_ahi_data(local_files_map, target_area, target_dt,
                                         composite_type, resample_type)

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
                    vis, ir, _, _ = process_monwatch_ahi_data(
                        local, global_area, dt, "sandwich", resample_type="nearest", sat_source="him")
                    rgb = _global_sandwich_rgb(vis, ir, dt, global_area)
                else:
                    ir, _, _, _ = process_monwatch_ahi_data(
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
            logging.debug(traceback.format_exc())
        
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
            b07, _, _, _ = process_monwatch_ahi_data(local_dat_map, seg_area, target_dt, "b07", sat_source=sat_source)
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
                res = process_monwatch_ahi_data(local_dat_map, None, target_dt, composite, sat_source=sat_source)
                res = tuple(_upscale_to_width(a, output_width) if a is not None else None
                            for a in res)
            else:
                res = process_monwatch_ahi_data(local_dat_map, seg_area, target_dt, composite,
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


def process_beyev(storm, crop_km, output_dir, output_width,
                  download_workers, decompress_workers, logo_path,
                  latest=False, date_str=None, time_str=None,
                  grid=False, grid_thick=0.4, grid_color="#00BFFF", grid_style="--",
                  no_coastlines=False, label=False, info=False,
                  export_formats=None, project="flat"):
                      
    lat = storm.get("latitude")
    lon = storm.get("longitude")
    if lat is None or lon is None:
        logging.warning("BEYEV: storm has no lat/lon; skipping.")
        return
    storm_id = storm.get("atcf_id") or storm.get("storm_name", "UNKNOWN")
    storm_name = storm.get("storm_name", "")
    logging.info(f"BEYEV: target {storm_id} @ ({lat:.2f}, {lon:.2f}), "
                 f"crop {crop_km} km")
                 
    def _view_angle(sub_lon):
        lon_n = ((float(lon) + 180.0) % 360.0) - 180.0
        lat_r = np.radians(float(lat))
        c = np.cos(lat_r) * np.cos(np.radians(lon_n - float(sub_lon)))
        return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))

    him_ang  = _view_angle(140.7)
    gk2a_ang = _view_angle(128.2)

    if him_ang >= 81.0 or gk2a_ang >= 81.0:
        off = []
        if him_ang  >= 81.0: off.append(f"Himawari-9 ({him_ang:.1f}°)")
        if gk2a_ang >= 81.0: off.append(f"GK-2A ({gk2a_ang:.1f}°)")
        logging.warning(f"BEYEV: {storm_id} at ({lat:.2f}, {lon:.2f}) is off "
                        f"the usable disk of {' and '.join(off)}; skipping.")
        return

    if him_ang >= 75.0 or gk2a_ang >= 75.0:
        logging.warning(f"BEYEV: {storm_id} near the limb — HIM {him_ang:.1f}°, "
                        f"GK2A {gk2a_ang:.1f}°; parallax will be distorted.")

    logging.info(f"BEYEV: view angles HIM {him_ang:.1f}°, GK2A {gk2a_ang:.1f}°")

    half_deg = max(crop_km, 100.0) / 111.32 / 2.0
    him_segs = get_required_segments((lat, lon, half_deg), buffer=False)

    dt, him_remote, gk2a_remote = _beyev_find_common_slot(
        him_segs, date_str=date_str, time_str=time_str, max_back=6)
    if dt is None:
        return

    him_tmpdir = tempfile.mkdtemp(prefix="beyev_him_")
    gk2a_tmpdir = tempfile.mkdtemp(prefix="beyev_gk2a_")
    try:
        him_local = download_and_decompress_all(
            him_remote, him_tmpdir, download_workers, decompress_workers)
        if not him_local:
            logging.error("BEYEV: Himawari download failed")
            return
        gk2a_local = download_gk2a_files(gk2a_remote, gk2a_tmpdir,
                                         download_workers)
        if not gk2a_local:
            logging.error("BEYEV: GK-2A download failed")
            return

        R = 6378137.0
        deg2rad = np.pi / 180.0
        x_half = half_deg * R * deg2rad
        y_min = (lat - half_deg) * R * deg2rad
        y_max = (lat + half_deg) * R * deg2rad
        area_def = AreaDefinition(
            "beyev", "BEYEV crop", "eqc",
            {"proj": "eqc", "lon_0": lon, "lat_ts": 0},
            output_width, output_width,
            (-x_half, y_min, x_half, y_max))

        logging.info("BEYEV: resampling Himawari-9 B13 ...")
        him_ir, _, _, _ = process_monwatch_ahi_data(
            him_local, area_def, dt, "infrared", sat_source="him")
        logging.info("BEYEV: resampling GK-2A ir105 ...")
        gk2a_ir, _, _, _ = process_gk2a_data(
            gk2a_local, area_def, dt, "infrared")

        if him_ir is None or gk2a_ir is None:
            logging.error("BEYEV: one of the IR arrays is empty")
            return
        if him_ir.shape != gk2a_ir.shape:
            gk2a_ir = _resize_like(gk2a_ir, him_ir.shape)
            
        try:
            _beyev_stereo_chart(
                him_ir, gk2a_ir,
                area_extent=tuple(area_def.area_extent),
                dt=dt, storm_id=storm_id,
                out_dir=output_dir,
                center_lat=lat, center_lon=lon,
                him_sat_lon=140.7, gk2a_sat_lon=128.2,
                export_formats=export_formats,
            )
        except Exception as _chart_err:
            logging.warning(f"BEYEV: stereo chart failed: {_chart_err}")

        def _nan_frac(a):
            return float(np.mean(~np.isfinite(np.asarray(a, dtype=np.float32))))

        him_bad  = _nan_frac(him_ir)
        gk2a_bad = _nan_frac(gk2a_ir)
        if him_bad > 0.95 or gk2a_bad > 0.95:
            logging.warning(f"BEYEV: resampled IR mostly empty for {storm_id} "
                            f"(HIM {him_bad*100:.0f}% NaN, GK2A {gk2a_bad*100:.0f}% NaN); "
                            f"skipping.")
            return

        def _norm_ir(a):
            a = np.asarray(a, dtype=np.float32)
            a = np.where(np.isfinite(a), a, 300.0)
            return np.clip((313.15 - a) / (313.15 - 173.15), 0.0, 1.0)

        him_n = _norm_ir(him_ir)
        gk2a_n = _norm_ir(gk2a_ir)

        him_c, gk2a_c = _paired_percentile_stretch(him_n, gk2a_n,
                                                   lo_pct=2.0, hi_pct=98.0)

        him_c = np.power(him_c, 0.85)
        gk2a_c = np.power(gk2a_c, 0.85)

        rgb = np.zeros((*him_c.shape, 3), dtype=np.float32)
        rgb[..., 0] = gk2a_c
        rgb[..., 1] = him_c
        rgb[..., 2] = him_c
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

        metadata = {
            'satellite_name': 'HIMAWARI-9 + GK-2A',
            'target_dt': dt,
            'center_lat': lat,
            'center_lon': lon,
            'crop_deg': half_deg,
            'crop_lon': half_deg,
            'crop_lat': half_deg,
            'product': 'BEYEV (anaglyph)',
            'storm_id': storm_id,
            'storm_name': storm_name,
            'winds': storm.get('winds'),
            'pressure': storm.get('pressure'),
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
            'polygon': None,
            'floater': False,
            'info': info,
            'project': project,
            'sat_lon': 140.7,
            'data_is_geos': False,
        }

        ts = dt.strftime('%Y%m%d_%H%M')
        out_name = f"{storm_id}_{ts}_beyev-rc_{crop_km:.0f}km.avif"
        out_path = os.path.join(output_dir, out_name)
        plot_image(rgb_u8, out_path, metadata,
                   logo_path=logo_path, export_formats=export_formats)
        logging.info(f"BEYEV: wrote {out_path}")
    finally:
        shutil.rmtree(him_tmpdir, ignore_errors=True)
        shutil.rmtree(gk2a_tmpdir, ignore_errors=True)

def _beyevs_estimate_cloud_height(ir_kelvin):
    bt = np.asarray(ir_kelvin, dtype=np.float32)
    bt = np.where(np.isfinite(bt), bt, 300.0)
    h_km = (288.0 - bt) / 6.5
    return np.clip(h_km, 0.0, 18.0)


def _beyevs_fill_holes(rgb, max_iter=12):
    out = rgb.copy()
    mask = (out.sum(axis=-1) == 0)
    if not mask.any():
        return out
    h, w = mask.shape
    for _ in range(max_iter):
        any_filled = False
        new_mask = mask.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted_rgb = np.roll(out, (dy, dx), axis=(0, 1))
            shifted_valid = np.roll(~mask, (dy, dx), axis=(0, 1))
            if dy > 0:
                shifted_valid[:dy] = False
            elif dy < 0:
                shifted_valid[dy:] = False
            if dx > 0:
                shifted_valid[:, :dx] = False
            elif dx < 0:
                shifted_valid[:, dx:] = False
            fill = new_mask & shifted_valid
            if fill.any():
                out[fill] = shifted_rgb[fill]
                new_mask[fill] = False
                any_filled = True
        mask = new_mask
        if not mask.any() or not any_filled:
            break
    return out


def _beyevs_perspective_warp(ir_kelvin, center_lat, center_lon,
                             crop_lon_deg, crop_lat_deg,
                             dip_deg=45.0, azimuth_deg=0.0, range_km=None,
                             height_scale=1.0, fov_deg=55.0,
                             out_size=None, bg_color=(0, 0, 0)):
    ir = np.asarray(ir_kelvin, dtype=np.float32)
    h_src, w_src = ir.shape
    if out_size is None:
        out_size = (h_src, w_src)
    h_out, w_out = out_size

    cloud_h = _beyevs_estimate_cloud_height(ir) * float(height_scale)

    lat_top   = center_lat + crop_lat_deg / 2.0
    lat_bot   = center_lat - crop_lat_deg / 2.0
    lon_left  = center_lon - crop_lon_deg / 2.0
    lon_right = center_lon + crop_lon_deg / 2.0
    lats = np.linspace(lat_top, lat_bot, h_src, dtype=np.float64)
    lons = np.linspace(lon_left, lon_right, w_src, dtype=np.float64)
    lon_g, lat_g = np.meshgrid(lons, lats)
    cos_lat = np.cos(np.radians(lat_g))
    E = (lon_g - center_lon) * 111.320 * cos_lat
    N = (lat_g - center_lat) * 110.574
    U = cloud_h.astype(np.float64)

    if range_km is None or range_km <= 0:
        scene_w_km = crop_lon_deg * 111.32 * max(np.cos(np.radians(center_lat)), 0.05)
        scene_h_km = crop_lat_deg * 111.32
        scene_size = max(scene_w_km, scene_h_km, 1.0)
        range_km = scene_size * 1.8

    dip = np.radians(float(dip_deg))
    az  = np.radians(float(azimuth_deg))
    horiz = range_km * np.cos(dip)
    vert  = range_km * np.sin(dip)
    cam_E = -horiz * np.sin(az)
    cam_N = -horiz * np.cos(az)
    cam_U = vert

    inv_rng = 1.0 / max(range_km, 1e-6)
    fwd = np.array([-cam_E * inv_rng, -cam_N * inv_rng, -cam_U * inv_rng])
    right = np.array([fwd[1], -fwd[0], 0.0])
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rn
    up = np.cross(right, fwd)

    vx = E - cam_E
    vy = N - cam_N
    vz = U - cam_U
    csx = vx * right[0] + vy * right[1] + vz * right[2]
    csy = vx * up[0]    + vy * up[1]    + vz * up[2]
    csz = vx * fwd[0]   + vy * fwd[1]   + vz * fwd[2]

    focal = (h_out / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    valid = csz > 1e-3
    csz_safe = np.where(valid, csz, 1.0)
    px = w_out / 2.0 + (csx / csz_safe) * focal
    py = h_out / 2.0 - (csy / csz_safe) * focal

    rgb = _sandwich_ir_lookup(ir)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    px_f = px.ravel()
    py_f = py.ravel()
    csz_f = csz.ravel()
    valid_f = valid.ravel()
    order = np.argsort(-csz_f, kind="stable")
    px_s = px_f[order]
    py_s = py_f[order]
    valid_s = valid_f[order]
    rgb_s = rgb_u8.reshape(-1, 3)[order]

    xi = np.rint(px_s).astype(np.int64)
    yi = np.rint(py_s).astype(np.int64)
    in_bounds = valid_s & (xi >= 0) & (xi < w_out) & (yi >= 0) & (yi < h_out)

    out = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    if bg_color != (0, 0, 0):
        out[:] = bg_color
    out[yi[in_bounds], xi[in_bounds]] = rgb_s[in_bounds]

    out = _beyevs_fill_holes(out)
    return out


def _beyevs_save_rgb(rgb_u8, out_path, export_formats):
    base_path = _export_base_path(out_path)
    img = Image.fromarray(rgb_u8)
    for fmt in (export_formats or ["avif"]):
        fmt = fmt.lower().strip()
        out_fmt = f"{base_path}.{fmt}"
        try:
            if os.path.exists(out_fmt):
                try:
                    os.remove(out_fmt)
                except OSError:
                    pass
            if fmt == "avif":
                img.save(out_fmt, format="AVIF", quality=95, subsampling="4:4:4")
            elif fmt == "png":
                img.save(out_fmt, format="PNG", compress_level=1)
            elif fmt in ("jpg", "jpeg"):
                img.save(out_fmt, format="JPEG", quality=95, subsampling=0)
            elif fmt == "webp":
                img.save(out_fmt, format="WEBP", quality=95, method=6)
            else:
                logging.warning(f"BEYEVS: unknown export format '{fmt}', skipping")
                continue
            logging.info(f"BEYEVS: saved {out_fmt}")
        except OSError as e:
            logging.warning(f"BEYEVS: failed to save {out_fmt}: {e}")


def _sat_display_name(sat_source):
    if sat_source == "gk2a":
        return "GK-2A"
    if sat_source == "mtg":
        return "MTG-I1"
    if sat_source in ("mtsat", "mtsat2"):
        return "MTSAT-2"
    if sat_source == "mtsat1":
        return "MTSAT-1R"
    if sat_source in ("goes16", "goes17", "goes18", "goes19"):
        return f"GOES-{sat_source[-2:]}"
    if sat_source == "goes":
        return "GOES"
    if sat_source == "him8":
        return "HIMAWARI-8"
    return "HIMAWARI-9"


def _beyevs_resolve_dt(sat_source, sat, segments, bands):
    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        return get_latest_available_dt_goes(sat)
    if sat_source == "mtg":
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        return (now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
                - datetime.timedelta(minutes=20))
    if sat_source in ("mtsat", "mtsat2", "mtsat1"):
        cfg = _mtsat_cfg(sat_source)
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        probe = now.replace(minute=0, second=0, microsecond=0)
        for _ in range(24):
            if mtsat_tar_exists(cfg, probe):
                return probe
            probe -= datetime.timedelta(hours=1)
        return None
    if sat_source == "gk2a":
        return get_latest_available_dt_gk2a(bands)
    return get_latest_available_dt(sat, segments, bands, use_target=False)


def process_beyevs(storm, crop_km, output_dir, output_width,
                   download_workers, decompress_workers, logo_path,
                   latest=False, date_str=None, time_str=None,
                   grid=False, grid_thick=0.4, grid_color="#00BFFF",
                   grid_style="--", no_coastlines=False, label=False,
                   info=False, export_formats=None, project="flat",
                   sat_source="him",
                   dip=45.0, azimuth=0.0, range_km=None, height_scale=1.0,
                   fov=55.0, fulldisk=False):
    lat = storm.get("latitude")
    lon = storm.get("longitude")
    if lat is None or lon is None:
        logging.warning("BEYEVS: storm has no lat/lon; skipping.")
        return
    storm_id   = storm.get("atcf_id") or storm.get("storm_name", "UNKNOWN")
    storm_name = storm.get("storm_name", "")

    logging.info(f"BEYEVS: {storm_id} @ ({lat:.2f}, {lon:.2f})  "
                 f"crop={crop_km:.0f} km  dip={dip:.1f}°  az={azimuth:.1f}°  "
                 f"range={range_km if range_km else 'auto'}  "
                 f"h_scale={height_scale:.2f}  fov={fov:.1f}°")
                 
    if fulldisk:
        lat = 0.0
        lon = _sat_subpoint_lon(sat_source)
        half_deg = 80.0
        logging.info(f"BEYEVS: full-disk mode — centre lon={lon}, half_span=±{half_deg:.0f}°")
    else:
        half_deg = max(crop_km, 100.0) / 111.32 / 2.0
    bands = [13]

    if sat_source in ("goes", "goes16", "goes17", "goes18", "goes19"):
        sat_source = _resolve_goes_source(sat_source, lon)
        satellites = _goes_candidate_buckets(sat_source)
        segments = []
    elif sat_source == "mtg":
        satellites = (MTG_COLLECTION,)
        segments = []
    elif sat_source in ("mtsat", "mtsat2", "mtsat1"):
        satellites = (sat_source,)
        segments = []
    elif sat_source == "gk2a":
        satellites = (GK2A_BUCKET,)
        segments = []
    else:
        satellites = ("noaa-himawari9", "noaa-himawari8")
        segments = get_required_segments((lat, lon, half_deg), buffer=False)

    dt = None
    if time_str:
        d = date_str if date_str else datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y%m%d")
        try:
            dt = datetime.datetime.strptime(f"{d}{time_str}", "%Y%m%d%H%M")
        except ValueError:
            logging.error(f"BEYEVS: invalid --date {date_str} / --time {time_str}")
            return

    local_dat_map = None
    sat_used = None
    tmpdir = None

    for sat in satellites:
        probe_dt = dt or _beyevs_resolve_dt(sat_source, sat, segments, bands)
        if probe_dt is None:
            continue
        for _attempt in range(2):
            if sat_source == "gk2a":
                remote_map = discover_gk2a_files(probe_dt, bands)
            else:
                remote_map = _discover_files(
                    sat_source, sat, probe_dt, bands, segments,
                    use_target=False, center_lat=lat, center_lon=lon,
                    roi_deg=half_deg)
            if not remote_map:
                break
            tmpdir = tempfile.mkdtemp(prefix="beyevs_")
            if sat_source == "gk2a":
                local = download_gk2a_files(remote_map, tmpdir, download_workers)
            else:
                local = _download_files(sat_source, remote_map, tmpdir,
                                        download_workers, decompress_workers)
            if local is not None:
                local_dat_map = local
                sat_used = sat
                dt = probe_dt
                break
            shutil.rmtree(tmpdir, ignore_errors=True)
            tmpdir = None
            probe_dt = probe_dt - datetime.timedelta(minutes=10)
            logging.info(f"BEYEVS: retrying prior slot {probe_dt:%Y-%m-%d %H:%M}Z")
        if sat_used is not None:
            break

    if sat_used is None or local_dat_map is None:
        logging.error("BEYEVS: no usable satellite data found.")
        return

    try:
        R = 6378137.0
        deg2rad = np.pi / 180.0
        half_m = half_deg * R * deg2rad
        area_def = AreaDefinition(
            "beyevs_crop", "BEYEVS crop", "eqc",
            {"proj": "eqc", "lon_0": lon, "lat_ts": 0},
            output_width, output_width,
            (-half_m, -half_m, half_m, half_m))

        logging.info("BEYEVS: resampling IR band ...")
        ir, _, _, _ = process_monwatch_ahi_data(
            local_dat_map, area_def, dt, "infrared", sat_source=sat_source)
        if ir is None:
            logging.error("BEYEVS: IR array empty")
            return
        ir = np.nan_to_num(ir, nan=300.0)

        logging.info("BEYEVS: applying bird-setup perspective warp ...")
        rgb = _beyevs_perspective_warp(
            ir,
            center_lat=lat, center_lon=lon,
            crop_lon_deg=half_deg * 2.0, crop_lat_deg=half_deg * 2.0,
            dip_deg=dip, azimuth_deg=azimuth,
            range_km=range_km, height_scale=height_scale,
            fov_deg=fov, out_size=(output_width, output_width))

        sat_name = _sat_display_name(sat_source)
        ts = dt.strftime("%Y%m%d_%H%M")
        out_stem = (f"{storm_id}_{ts}_beyevs-{sat_name.replace(' ', '')}"
                    f"_dip{int(dip):d}_az{int(azimuth):+d}_{crop_km:.0f}km")
        out_path = os.path.join(output_dir, out_stem + ".avif")

        if export_formats is None:
            export_formats = ["avif"]
        _beyevs_save_rgb(rgb, out_path, export_formats)
        logging.info(f"BEYEVS: done -> {out_path}")
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

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
               else "HIM8" if sat_source == "him8"
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
        bounds = (lat, lon, half_lat)
        out_h = round(output_width * (half_lat / half_lon))
    else:
        R_earth = 6371.0
        lat_deg_per_km = 1.0 / 111.32
        lon_deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat)))
        half_km = crop_km / 2.0
        half_deg = max(half_km * lat_deg_per_km, half_km * lon_deg_per_km)
        half_lon = half_lat = half_deg
        bounds = (lat, lon, half_km * lat_deg_per_km)
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
                if sat_source == "him8":
                    satellites = ("noaa-himawari8", "noaa-himawari9")
                elif sat_source == "him9":
                    satellites = ("noaa-himawari9", "noaa-himawari8")
                else:
                    satellites = ("noaa-himawari9", "noaa-himawari8")
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
                seg_ir, _, _, _ = process_monwatch_ahi_data(seg_map, None, seg_dt, "infrared", sat_source=sat_source)
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
            target_area_for_monwatch = None
        else:
            logging.info("  Loading and resampling B13 data (shared across all IR products)...")
            target_area_for_monwatch = None if use_target else area_def
        ir, _, _, _ = process_monwatch_ahi_data(local_dat_map, target_area_for_monwatch, dt, "infrared", sat_source=sat_source)
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
                      "JPSS-OZONE", "JPSS-NGRN", "JPSS-OCL2", "JPSS-SND",
                      "N20", "N21", "SNPP", "NOAA-20", "NOAA-21", "NPP",
                      "NOAA20", "NOAA21", "S-NPP"}
    _jpss_pds_aliases = {
        "n20": ("VIIRS-SDR", "J01"), "noaa-20": ("VIIRS-SDR", "J01"),
        "noaa20": ("VIIRS-SDR", "J01"), "j01": ("VIIRS-SDR", "J01"),
        "n21": ("VIIRS-SDR", "J02"), "noaa-21": ("VIIRS-SDR", "J02"),
        "noaa21": ("VIIRS-SDR", "J02"), "j02": ("VIIRS-SDR", "J02"),
        "snpp": ("VIIRS-SDR", "NPP"), "npp": ("VIIRS-SDR", "NPP"),
        "s-npp": ("VIIRS-SDR", "NPP"),
    }
    if sat_source:
        sat_key = str(sat_source).strip().lower().replace("_", "-")
        if sat_key in _jpss_pds_aliases:
            fam, sat = _jpss_pds_aliases[sat_key]
            return process_jpss_storm(
                storm, crop_km, product, output_dir, output_width,
                family=fam, jpss_product=jpss_product, jpss_sat=jpss_sat or sat,
                date_str=date_str, time_str=time_str,
                download_workers=download_workers, logo_path=logo_path,
                grid=grid, grid_thick=grid_thick, grid_color=grid_color, grid_style=grid_style,
                no_coastlines=no_coastlines, label=label,
                export_formats=export_formats, floater=floater, info=info, project=project)
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
               else "HIM8" if sat_source == "him8"
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
            bounds = (lat, lon, half_lat)
            out_h = round(output_width * (half_lat / half_lon))
        else:
            R_earth = 6371.0
            lat_deg_per_km = 1.0 / 111.32
            lon_deg_per_km = 1.0 / (111.32 * np.cos(np.radians(lat)))
            half_km = crop_km / 2.0
            half_deg = max(half_km * lat_deg_per_km, half_km * lon_deg_per_km)
            half_lon = half_lat = half_deg
            bounds = (lat, lon, half_km * lat_deg_per_km)
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
                if sat_source == "him8":
                    satellites = ("noaa-himawari8", "noaa-himawari9")
                elif sat_source == "him9":
                    satellites = ("noaa-himawari9", "noaa-himawari8")
                else:
                    satellites = ("noaa-himawari9",) if use_target \
                                 else ("noaa-himawari9", "noaa-himawari8")
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

        target_area_for_monwatch = area_def
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
            seg_area = target_area_for_monwatch
            meta_lat_bak = metadata.get('center_lat')
            meta_lon_bak = metadata.get('center_lon')
            meta_ext_bak = metadata.get('target_extent')
            if use_target:
                native = _native_target_area(seg_map)
                if native is None:
                    seg_area = target_area_for_monwatch
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
                    seg_area = target_area_for_monwatch
                elif seg_lat is not None and seg_lon is not None:
                    metadata['center_lat'] = seg_lat
                    metadata['center_lon'] = seg_lon
                if seg_extent:
                    metadata['target_extent'] = seg_extent

            def _read(composite):
                if use_fulldisk_resolution:
                    res = process_monwatch_ahi_data(seg_map, None, use_dt, composite,
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
                    return process_monwatch_ahi_data(seg_map, seg_area, use_dt, composite,
                                                    resample_type=resample_type, sat_source=sat_source)
                res = process_monwatch_ahi_data(seg_map, None, use_dt, composite, sat_source=sat_source)
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

def _load_dotenv():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else "."
    candidates = [
        ".env",
        "env.txt",
        os.path.join(script_dir, ".env"),
        os.path.join(script_dir, "env.txt"),
    ]
    loaded_any = False
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
            loaded_any = True
            logging.debug(f"Loaded env from {env_path}")
        except Exception as e:
            logging.warning(f"Failed to load env from {env_path}: {e}")

    def _first_env(*names):
        for n in names:
            v = os.environ.get(n)
            if v:
                return v.strip()
        return None

    user = _first_env(
        "SPACETRACK_USERNAME", "SPACETRACK_USER",
        "star-trackuser", "star_trackuser", "STAR_TRACK_USER",
        "spacetrack_username", "spacetrack_user",
    )
    pw = _first_env(
        "SPACETRACK_PASSWORD", "SPACETRACK_PASS",
        "star-trackpass", "star_trackpass", "STAR_TRACK_PASS",
        "spacetrack_password", "spacetrack_pass",
    )
    if user:
        os.environ["SPACETRACK_USERNAME"] = user
    if pw:
        os.environ["SPACETRACK_PASSWORD"] = pw
    return loaded_any


def _spacetrack_credentials():
    user = (os.environ.get("SPACETRACK_USERNAME") or "").strip()
    pw = (os.environ.get("SPACETRACK_PASSWORD") or "").strip()
    if user and pw:
        return user, pw
    return None, None

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

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _args_json = os.path.join(_script_dir, "args.json")
    if not os.path.exists(_args_json):
        print(f"ERROR: args.json not found at {_args_json}", file=sys.stderr)
        sys.exit(1)

    _args_config = _load_args_config(_args_json)
    parser = _build_parser_from_config(_args_config)
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
    elif getattr(args, "pds_n21", False):
        sat_source = "n21"
    elif getattr(args, "pds_n20", False):
        sat_source = "n20"
    elif getattr(args, "pds_snpp", False):
        sat_source = "snpp"
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

    if getattr(args, "pds_n21", False) or getattr(args, "noaa21", False):
        args.jpss_sat = "J02"
    elif getattr(args, "pds_n20", False) or getattr(args, "noaa20", False):
        args.jpss_sat = "J01"
    elif getattr(args, "pds_snpp", False) or getattr(args, "npp_sat", False):
        args.jpss_sat = "NPP"
    elif getattr(args, "noaa_auto", False):
        if not getattr(args, "jpss_sat", None):
            args.jpss_sat = None
        logging.info("JPSS: --NOAA auto (prefer NOAA-21/J02, then NOAA-20/J01, then NPP)")

    _jpss_platform = any([
        getattr(args, "noaa20", False), getattr(args, "noaa21", False),
        getattr(args, "noaa_auto", False), getattr(args, "npp_sat", False),
        getattr(args, "pds_n20", False), getattr(args, "pds_n21", False),
        getattr(args, "pds_snpp", False),
        bool(getattr(args, "jpss_sat", None)),
    ])
    if _jpss_platform and sat_source is None:
        sat_source = "VIIRS-SDR"
        logging.info("JPSS: no family flag given; defaulting to VIIRS-SDR")

    multi = bool(getattr(args, "multi", False))
    _any_goes = any([args.goes, getattr(args, "goes16", False), getattr(args, "goes17", False),
                     args.goes18, args.goes19])
    if multi:
        sat_sources = ["him", "gk2a"]
        if getattr(args, "goes16", False):
            sat_sources.append("goes16")
        if getattr(args, "goes17", False):
            sat_sources.append("goes17")
        if args.goes18:
            sat_sources.append("goes18")
        if args.goes19:
            sat_sources.append("goes19")
        if args.goes and not any([getattr(args, "goes16", False),
                                  getattr(args, "goes17", False),
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
        _seen = set()
        sat_sources = [s for s in sat_sources if not (s in _seen or _seen.add(s))]
        if len(sat_sources) > 1:
            sat_names = {"him": "Himawari-9", "gk2a": "GK-2A", "goes": "GOES",
                         "goes16": "GOES-16", "goes17": "GOES-17",
                         "goes18": "GOES-18", "goes19": "GOES-19", "mtg": "MTG",
                         "mtsat": "MTSAT-2", "mtsat2": "MTSAT-2", "mtsat1": "MTSAT-1R"}
            sat_list = [sat_names.get(s, s) for s in sat_sources]
            logging.warning(f"--multi: processing with {' and '.join(sat_list)}")
    else:
        sat_sources = [sat_source]
    if not multi and sat_source in ("gk2a", "goes", "goes16", "goes17", "goes18",
                                    "goes19", "mtg", "mtsat", "mtsat1", "mtsat2"):
        if args.target:
            logging.warning("--target is only supported with Himawari; disabling for this satellite source.")
            args.target = False
        if sat_source == "gk2a" and "--color" not in sys.argv:
            args.color = GK2A_DEFAULT_GRID_COLOR

    def _sat_output(sat):
        return args.output

    def _sat_use_target(sat):
        return args.target if sat in ("him", "him8", "him9") else False

    def _sat_color(sat):
        if sat in ("him", "him8", "him9"):
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
                       getattr(args, "pds_n20", False), getattr(args, "pds_n21", False),
                       getattr(args, "pds_snpp", False),
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
        
        if getattr(args, "beyev", False):
            try:
                process_beyev(
                    custom_storm, args.crop_km, args.output, args.width,
                    args.download_workers, args.decompress_workers, args.logo,
                    latest=args.latest, date_str=args.date, time_str=args.time,
                    grid=args.grid, grid_thick=args.thick, grid_color=args.color,
                    grid_style=args.style, no_coastlines=args.no_coastlines,
                    label=args.label, info=args.info,
                    export_formats=export_formats, project=args.project)
            except Exception as e:
                logging.error(f"BEYEV failed for CUSTOM: {e}")
                if args.verbose:
                    logging.debug(traceback.format_exc())
            logging.info("Done.")
            return

        if getattr(args, "beyevs", False):
            _beyevs_sat = sat_sources[0] or "him"
            _fldk_lon = _sat_subpoint_lon(
                _resolve_goes_source(_beyevs_sat, args.lon)
                if _beyevs_sat == "goes" else _beyevs_sat
            )
            _beyevs_storm = {
                "atcf_id": "FLDK" if args.fulldisk else "CUSTOM",
                "storm_name": "Full Disk" if args.fulldisk else "Custom Point",
                "latitude": 0.0 if args.fulldisk else args.lat,
                "longitude": _fldk_lon if args.fulldisk else args.lon,
                "winds": None, "pressure": None,
            }
            try:
                process_beyevs(
                    _beyevs_storm, args.crop_km, args.output, args.width,
                    args.download_workers, args.decompress_workers, args.logo,
                    latest=args.latest, date_str=args.date, time_str=args.time,
                    grid=args.grid, grid_thick=args.thick, grid_color=args.color,
                    grid_style=args.style, no_coastlines=args.no_coastlines,
                    label=args.label, info=args.info,
                    export_formats=export_formats, project=args.project,
                    sat_source=_beyevs_sat,
                    dip=args.beyevs_dip, azimuth=args.beyevs_azimuth,
                    range_km=args.beyevs_range,
                    height_scale=args.beyevs_height, fov=args.beyevs_fov,
                    fulldisk=args.fulldisk)
            except Exception as e:
                logging.error(f"BEYEVS failed for CUSTOM: {e}")
                if args.verbose:
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
            
    if getattr(args, "beyev", False):
        os.makedirs(args.output, exist_ok=True)

        _explicit = bool(
            args.storm or region_specs
            or args.target or args.philippines or args.westpac
            or args.nl or args.sl or args.ipar or args.pmd
            or args.camsur or args.pwardsc or args.manila
        )

        if not _explicit:
            storms = [s for s in storms if _storm_in_wpac(s)]
            logging.info(f"BEYEV: no explicit target — WestPac sweep, "
                         f"{len(storms)} storm(s).")

        if not storms:
            logging.warning("BEYEV: no target to process.")
            logging.info("Done.")
            return

        logging.info(f"BEYEV: {len(storms)} target(s) at "
                     f"{args.crop_km:.0f} km crop -> {args.output}")
        for storm in storms:
            if all(storm.get(k) is not None for k in
                   ("lat_min", "lat_max", "lon_min", "lon_max")):
                lat_span = storm["lat_max"] - storm["lat_min"]
                lon_span = storm["lon_max"] - storm["lon_min"]
                eff_crop_km = max(lat_span, lon_span) * 111.32
                storm = dict(storm)
                storm["latitude"]  = (storm["lat_min"] + storm["lat_max"]) / 2.0
                storm["longitude"] = (storm["lon_min"] + storm["lon_max"]) / 2.0
            else:
                eff_crop_km = args.crop_km

            try:
                process_beyev(
                    storm, eff_crop_km, args.output, args.width,
                    args.download_workers, args.decompress_workers, args.logo,
                    latest=args.latest, date_str=args.date, time_str=args.time,
                    grid=args.grid, grid_thick=args.thick, grid_color=args.color,
                    grid_style=args.style, no_coastlines=args.no_coastlines,
                    label=args.label, info=args.info,
                    export_formats=export_formats, project=args.project)
            except Exception as e:
                logging.error(f"BEYEV failed for {storm.get('atcf_id')}: {e}")
                if args.verbose:
                    logging.debug(traceback.format_exc())
        logging.info("Done.")
        return
        
    if getattr(args, "beyevs", False):
        os.makedirs(args.output, exist_ok=True)

        _beyevs_sat = sat_sources[0] or "him"

        if args.fulldisk:
            _fldk_lon = _sat_subpoint_lon(
                _resolve_goes_source(_beyevs_sat) if _beyevs_sat == "goes" else _beyevs_sat
            )
            storms = [{
                "atcf_id": "FLDK",
                "storm_name": "Full Disk",
                "latitude": 0.0,
                "longitude": _fldk_lon,
                "winds": None, "pressure": None,
            }]
            logging.info(f"BEYEVS: full-disk render — sub-point lon={_fldk_lon}")
        elif not storms:
            logging.warning("BEYEVS: no target to process.")
            logging.info("Done.")
            return

        logging.info(f"BEYEVS: {len(storms)} target(s) "
                     f"dip={args.beyevs_dip:.1f}° "
                     f"az={args.beyevs_azimuth:.1f}° "
                     f"h_scale={args.beyevs_height:.2f} "
                     f"fov={args.beyevs_fov:.1f}°"
                     + (" [FULL DISK]" if args.fulldisk else ""))

        for storm in storms:
            if (not args.fulldisk) and all(
                    storm.get(k) is not None for k in
                    ("lat_min", "lat_max", "lon_min", "lon_max")):
                lat_span = storm["lat_max"] - storm["lat_min"]
                lon_span = storm["lon_max"] - storm["lon_min"]
                eff_crop_km = max(lat_span, lon_span) * 111.32
                storm = dict(storm)
                storm["latitude"]  = (storm["lat_min"] + storm["lat_max"]) / 2.0
                storm["longitude"] = (storm["lon_min"] + storm["lon_max"]) / 2.0
            else:
                eff_crop_km = args.crop_km

            try:
                process_beyevs(
                    storm, eff_crop_km, args.output, args.width,
                    args.download_workers, args.decompress_workers, args.logo,
                    latest=args.latest, date_str=args.date, time_str=args.time,
                    grid=args.grid, grid_thick=args.thick, grid_color=args.color,
                    grid_style=args.style, no_coastlines=args.no_coastlines,
                    label=args.label, info=args.info,
                    export_formats=export_formats, project=args.project,
                    sat_source=_beyevs_sat,
                    dip=args.beyevs_dip, azimuth=args.beyevs_azimuth,
                    range_km=args.beyevs_range,
                    height_scale=args.beyevs_height, fov=args.beyevs_fov,
                    fulldisk=args.fulldisk)
            except Exception as e:
                logging.error(f"BEYEVS failed for {storm.get('atcf_id')}: {e}")
                if args.verbose:
                    logging.debug(traceback.format_exc())
        logging.info("Done.")
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