# =============================================================================
# rem_ingest/mtg.py — MTG-I1 FCI ingest for MonWatch-CLI
#
# (C) 2025-2026 PWARDS-weather
# SPDX-License-Identifier: Apache-2.0 OR GPL-3.0-or-later
#
# Dual-licensed. You may use, modify, and distribute this file under the terms
# of EITHER the Apache License, Version 2.0, or the GNU General Public License,
# Version 3.0 or later — not both. See LICENSE.txt in the project root for the
# full license texts and the copyright notice.
# =============================================================================
#
# Requires the optional 'eumdac' package and EUMETSAT consumer credentials,
# supplied via (checked in this order):
#
#   1. Environment variables:
#          EUMETSAT_CONSUMER_KEY
#          EUMETSAT_CONSUMER_SECRET
#
#   2. An env.txt file in the project root containing those two keys,
#      one per line, in "KEY=value" form:
#          EUMETSAT_CONSUMER_KEY=...
#          EUMETSAT_CONSUMER_SECRET=...
# =============================================================================
import os
import re
import io
import time
import socket
import zipfile
import logging
import datetime
import threading

import numpy as np
import xarray as xr
from pyresample import AreaDefinition, kd_tree
from satpy import Scene

from . import common
from .common import (
    resize_like, linear_normalize, false_color_rgb, AHI_TO_FCI,
)
from ._rgb_corrections import apply_rgb_corrections

MTG_COLLECTION = "EO:EUM:DAT:0662"

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

def _load_eumetsat_creds():
    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if key and secret:
        return key, secret

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    for name in ("env.txt", ".env"):
        env_file = os.path.join(root_dir, name)
        if os.path.exists(env_file):
            try:
                pairs = {}
                for line in open(env_file, encoding="utf-8", errors="replace"):
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

    return key, secret


def _ensure_mtg_setup():
    try:
        import eumdac
    except ImportError:
        raise RuntimeError(
            "MTG (EUMETSAT) support requires the 'eumdac' package. "
            "Install it with: pip install eumdac hdf5plugin  "
            "and set EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET as "
            "environment variables or in an env.txt file in the project root."
        )
    key, secret = _load_eumetsat_creds()
    if not key or not secret:
        raise RuntimeError(
            "MTG (EUMETSAT) requires credentials. "
            "Set EUMETSAT_CONSUMER_KEY and EUMETSAT_CONSUMER_SECRET as "
            "environment variables, or put them in env.txt in the project "
            "root (one 'EUMETSAT_CONSUMER_KEY=...' / "
            "'EUMETSAT_CONSUMER_SECRET=...' per line)."
        )
    return eumdac


def eumetsat_creds_available():
    try:
        import eumdac
    except ImportError:
        return False
    if os.environ.get("EUMETSAT_CONSUMER_KEY") and \
       os.environ.get("EUMETSAT_CONSUMER_SECRET"):
        return True
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    env_file = os.path.join(root_dir, "env.txt")
    if os.path.exists(env_file):
        try:
            txt = open(env_file, encoding="utf-8", errors="replace").read()
            if "EUMETSAT_CONSUMER_KEY" in txt and \
               "EUMETSAT_CONSUMER_SECRET" in txt:
                return True
        except Exception:
            pass
    return False

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
                    if total // (128 * 1024 * 1024) > \
                       (total - len(buf)) // (128 * 1024 * 1024):
                        logging.info(f"  MTG download: {total / 1e6:.0f} MB ...")
            logging.info(f"MTG product downloaded: {total / 1e6:.0f} MB")
            return download_path
        except Exception as e:
            last_err = e
            logging.warning(f"MTG download attempt {attempt}/{attempts} failed: {e}")
    raise last_err

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


def _read_mtg_tailored_channel(local_paths, channel, target_area,
                               resample_type="nearest"):
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
        y_vals = np.asarray(ds["y"].values)
        if float(y_vals[0]) < float(y_vals[-1]):
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

def _mtg_tailor_download(datastore, datatailor, product, channels, roi_nswe,
                         temp_dir):
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
                logging.warning(f"MTG tailor submit attempt {attempt}/3 failed: "
                                f"{submit_err}")
                time.sleep(5)
        if custs is None:
            raise RuntimeError("MTG Data Tailor submit failed after 3 attempts")

        for c in custs:
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
                        logging.error(f"MTG tailor job {c._id} did not finish "
                                      f"within the poll deadline (last status "
                                      f"{st}); killing it")
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
                    log_tail = c.logfile[-800:]
                except Exception:
                    log_tail = ""
                raise RuntimeError(f"MTG tailor job {c._id} ended {st}: {log_tail}")

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

def discover_mtg_files(satellite, dt_obj, bands_list,
                       temp_dir="temp_data", roi_nswe=None):
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
        logging.warning(f"No MTG FCI products found near "
                        f"{dt_obj.strftime('%Y-%m-%d %H:%M')}Z")
        return {}

    channels = sorted({AHI_TO_FCI[b] for b in bands_list if AHI_TO_FCI.get(b)})
    if not channels:
        logging.warning(f"No MTG FCI channels requested for bands {bands_list}")
        return {}

    try:
        out_files = _mtg_tailor_download(datastore, datatailor, product,
                                         channels, roi_nswe, temp_dir)
    except Exception as tailor_err:
        logging.warning(f"Data Tailor failed ({tailor_err}); falling back to "
                        f"full package download")
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
            vis = _read_mtg_tailored_channel(all_files, "vis_06",
                                             target_area, "nearest")
        except ValueError:
            vis = None
    return ir, vis

def process_mtg_data(local_files_map, target_area, target_dt,
                     composite_type, resample_type="nearest"):
    all_files = []
    for paths in local_files_map.values():
        all_files.extend(paths)
    if not all_files:
        raise ValueError("No MTG local files")

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
        raise ValueError(f"Unsupported MTG composite: {composite_type}")
    for b in bands_needed:
        if AHI_TO_FCI.get(b) is None:
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

    if composite_type == "firetemp":
        r = linear_normalize(_read(7), 273.0, 350.0)
        g = linear_normalize(_read(6), 0.0, 50.0)
        b = linear_normalize(_read(9), 0.0, 50.0)
        return r, g, b, None

    if composite_type == "fire":
        return _read(7), None, None, None

    if composite_type == "true":
        r = _read(3)
        b = resize_like(_read(1), r.shape)
        v = resize_like(_read(4), r.shape)
        ir = resize_like(_read(13), r.shape)
        g = 0.45 * r + 0.10 * v + 0.45 * b
        r, g, b = apply_rgb_corrections(r, g, b, ir, target_area,
                                        target_dt, mode=1)
        return r, g, b, None

    if composite_type == "dayconv":
        b03 = _read(3); b05 = _read(5); b07 = _read(7)
        b08 = _read(8); b10 = _read(10); b13 = _read(13)
        r = linear_normalize(b08 - b10, -35.0, 5.0)
        g = linear_normalize(b07 - b13, -5.0, 60.0, gamma=0.5)
        b = linear_normalize(b03 - b05, -10.0, 70.0, gamma=0.95, invert=True)
        return r, g, b, None

    raise ValueError(f"Unsupported MTG composite: {composite_type}")