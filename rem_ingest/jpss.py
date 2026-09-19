# =============================================================================
# rem_ingest/jpss.py — JPSS VIIRS façade for MonWatch-CLI
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
import shutil
import logging
import datetime
import tempfile

import numpy as np
import matplotlib.colors as mcolors
from satpy import Scene

from . import common
from .common import (
    resize_like, linear_normalize, normalize_reflectance,
    false_color_rgb, stack_rgb,
    DVORAK_nodes, OTT_nodes, _DVORAK_IR_LUT,
)

from .jpss_common import (
    VIIRS_NORAD_IDS, VIIRS_HALF_SWATH_KM, VIIRS_TLE_CACHE, VIIRS_TLE_MAX_AGE,
    _spacetrack_credentials,
    _jpss_extract_orbit, _jpss_parse_granule_time_from_name,
    _jpss_haversine_km, _jpss_dist_to_box_km,
    _jpss_parse_tle_text, _jpss_fetch_tle, _jpss_fetch_tle_spacetrack,
    _jpss_get_orbital, _jpss_subsat_track, _jpss_find_passes,
    _jpss_rank_passes, _jpss_find_h5_dataset, _jpss_check_geo_coverage,
    _jpss_pds_list_keys,
    jpss_download_tar, jpss_extract_tar, jpss_select_closest_files,
)

from .jpss_pds import (
    JPSS_PDS_BUCKETS, JPSS_PDS_SAT_TOKEN, JPSS_PDS_SAT_LABEL,
    JPSS_PDS_BUCKET_TO_KEY, JPSS_PDS_COMPOSITE_BANDS, VIIRS_PRODUCT_INFO,
    _jpss_pds_resolve_bucket, _jpss_pds_bucket_candidates,
    discover_jpss_pds_files, download_jpss_pds_files,
)

from .jpss_class import (
    JPSS_CLASS_BASE, JPSS_FAMILY_DEFAULT_PRODUCTS,
    _jpss_list_dir, jpss_list_available_dates, jpss_closest_date,
    jpss_list_families, jpss_list_products, jpss_list_sats, jpss_list_tars,
    jpss_resolve_product, jpss_resolve_sat,
    discover_jpss_files, download_jpss_and_extract,
)

def _jpss_satpy_reader_for_files(files):
    names = " ".join(os.path.basename(f).lower() for f in files)
    if any(x in names for x in ("jrr-", "surfref", "lst_", "aod", "adp",
                                "cloudmask", "cloudheight")):
        return "viirs_edr"
    if any(x in names for x in ("svm", "svi", "svdnb", "gimgo",
                                "gitco", "gmodo", "gmtco", "gdnbo")):
        return "viirs_sdr"
    return "viirs_sdr"

def process_jpss_data(local_files, target_area, target_dt, composite_type,
                      resample_type="nearest"):
    if not local_files:
        raise ValueError("No JPSS local files")
    reader = _jpss_satpy_reader_for_files(local_files)
    logging.info(f"JPSS: loading {len(local_files)} file(s) "
                 f"with satpy reader '{reader}'")
    try:
        scn = Scene(filenames=local_files, reader=reader)
    except Exception as e:
        alt = "viirs_edr" if reader == "viirs_sdr" else "viirs_sdr"
        logging.warning(f"JPSS: reader {reader} failed ({e}); trying {alt}")
        scn = Scene(filenames=local_files, reader=alt)

    ir_candidates = ["I05", "M15", "M16",
                     "brightness_temperature_I5",
                     "brightness_temperature_M15",
                     "BT", "BrightnessTemperature",
                     "cloud_top_temperature"]
    vis_candidates = ["I01", "M05", "M03",
                      "reflectance_I1", "reflectance_M5"]

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
                if "i05" in al or "m15" in al \
                        or "brightness" in al or al.endswith("_bt"):
                    scn.load([a])
                    return a
        except Exception as e:
            logging.warning(f"JPSS available_dataset_names failed: {e}")
        return None

    if composite_type in ("infrared", "dvorak", "ir",
                          "z1-ir", "althea-ott2", "bt0", "z1-dvorak"):
        key = _try_load(ir_candidates)
        if key is None:
            raise ValueError("JPSS: could not load an IR "
                             "brightness-temperature dataset")
        data = scn[key]
        if target_area is not None:
            res = scn.resample(target_area, resampler=resample_type,
                               reduce_data=True, radius_of_influence=20000)
            data = res[key]
        arr = np.asarray(
            data.compute() if hasattr(data, "compute") else data,
            dtype=np.float32)
        if np.nanmax(arr) < 100:
            arr = arr + 273.15
        return arr, None, None, None

    if composite_type in ("sandwich", "irv", "falsecolor",
                          "falsecoloradv", "true", "b03", "z1-true"):
        ir_key = _try_load(ir_candidates)
        vis_key = _try_load(vis_candidates)
        if ir_key is None and vis_key is None:
            raise ValueError("JPSS: no VIS/IR datasets available "
                             "for this composite")
        res = (scn.resample(target_area, resampler=resample_type,
                            reduce_data=True, radius_of_influence=20000)
               if target_area is not None else scn)
        ir = vis = None
        if ir_key and ir_key in res:
            ir = np.asarray(
                res[ir_key].compute() if hasattr(res[ir_key], "compute")
                else res[ir_key], dtype=np.float32)
            if np.nanmax(ir) < 100:
                ir = ir + 273.15
        if vis_key and vis_key in res:
            vis = np.asarray(
                res[vis_key].compute() if hasattr(res[vis_key], "compute")
                else res[vis_key], dtype=np.float32)
        if composite_type == "b03" and vis is not None:
            return vis, None, None, None
        if vis is not None and ir is not None:
            if composite_type in ("falsecolor", "falsecoloradv",
                                  "sandwich", "irv"):
                from pyorbital.astronomy import sun_zenith_angle
                if target_area is not None:
                    lons, lats = target_area.get_lonlats()
                    sza = sun_zenith_angle(
                        target_dt or datetime.datetime.utcnow(), lons, lats)
                else:
                    sza = np.zeros(vis.shape, dtype=np.float32)
                r, g, b = false_color_rgb(
                    vis, ir, sza,
                    advanced=(composite_type == "falsecoloradv"))
                return r, g, b, None
            if composite_type in ("true", "z1-true"):
                vis_n = normalize_reflectance(vis)
                ir_n = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
                return vis_n, vis_n, vis_n * 0.85 + ir_n * 0.15, None
        if ir is not None:
            return ir, None, None, None
        if vis is not None:
            return vis, None, None, None

    key = _try_load(ir_candidates + vis_candidates)
    if key is None:
        raise ValueError(f"JPSS: unsupported composite {composite_type} "
                         f"/ no datasets")
    data = scn[key]
    if target_area is not None:
        res = scn.resample(target_area, resampler=resample_type,
                           reduce_data=True, radius_of_influence=20000)
        data = res[key]
    arr = np.asarray(
        data.compute() if hasattr(data, "compute") else data,
        dtype=np.float32)
    return arr, None, None, None

def process_jpss_storm(storm, crop_km, product, output_dir, output_width,
                       family, jpss_product=None, jpss_sat=None,
                       date_str=None, time_str=None,
                       download_workers=4, logo_path=None,
                       grid=False, grid_thick=0.4, grid_color="#00BFFF",
                       grid_style="--",
                       no_coastlines=False, label=False,
                       export_formats=None, floater=False, info=False,
                       project="flat",
                       plotter=None):
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
            target_dt = datetime.datetime.strptime(
                date_str + tpart, "%Y%m%d%H%M")
        except ValueError:
            target_dt = None

    family_u = (family or "").upper()
    use_pds = (family_u in ("VIIRS-SDR", "VIIRS", "VIIRSI-EDR")
               or family_u.startswith("VIIRS"))
    if jpss_sat:
        sk = jpss_sat.strip().lower().replace("_", "-")
        if sk in JPSS_PDS_BUCKETS or sk in ("j01", "j02", "npp",
                                            "n20", "n21", "snpp"):
            use_pds = True

    meta = None
    local_files = None
    work_dir = tempfile.mkdtemp(prefix=f"jpss_{family_u or 'VIIRS'}_")
    try:
        if use_pds:
            logging.info("JPSS: trying NESDIS PDS "
                         "(orbit-aware coverage filter)...")
            meta = discover_jpss_pds_files(
                composite_type=product, target_dt=target_dt,
                date_str=date_str, time_str=time_str, sat_id=jpss_sat,
                center_lat=lat, center_lon=lon, crop_km=crop_km or 1000,
                max_passes=6, search_window_hours=14)
            if meta:
                local_files = download_jpss_pds_files(
                    meta, work_dir, download_workers=download_workers)
                if not local_files:
                    logging.warning("JPSS PDS download empty; "
                                    "falling back to CLASS")
                    meta = None

        if meta is None:
            logging.info("JPSS: using CLASS archive path...")
            cls_family = family
            if family_u in ("N20", "N21", "SNPP", "NOAA-20",
                            "NOAA-21", "NPP"):
                cls_family = "VIIRS-SDR"
            meta = discover_jpss_files(
                cls_family, target_dt=target_dt, date_str=date_str,
                time_str=time_str, product=jpss_product, sat_id=jpss_sat,
                center_lat=lat, center_lon=lon)
            if not meta:
                logging.error(f"JPSS: discovery failed for family={family}")
                return
            local_files = download_jpss_and_extract(
                meta, work_dir, download_workers=download_workers)
            if not local_files:
                logging.error("JPSS: no granules after download/extract")
                return

        half_km = (crop_km or 1000) / 2.0
        lat_deg = half_km / 111.32
        lon_deg = half_km / (111.32 * max(np.cos(np.radians(lat)), 0.05))
        if all(storm.get(k) is not None
               for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
            extent = [storm["lon_min"], storm["lat_min"],
                      storm["lon_max"], storm["lat_max"]]
            width_m = abs(storm["lon_max"] - storm["lon_min"]) * 111320 * \
                max(np.cos(np.radians(lat)), 0.05)
            height_m = abs(storm["lat_max"] - storm["lat_min"]) * 111320
        else:
            extent = [lon - lon_deg, lat - lat_deg,
                      lon + lon_deg, lat + lat_deg]
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
        if composite in ("ir", "infrared", "z1-ir", "althea-ott2",
                         "bt0", "dvorak", "z1-dvorak"):
            ir, _, _, _ = process_jpss_data(local_files, target_area,
                                            obs_dt, "infrared")
            ir = np.nan_to_num(ir, nan=300.0)
            ir_c = ir - 273.15
            if composite in ("dvorak", "z1-dvorak"):
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    "Dvorak", DVORAK_nodes)
                display = "DVORAK (VIIRS)"
            elif composite == "bt0":
                cmap = mcolors.ListedColormap(_DVORAK_IR_LUT,
                                              name="dvorak_ir")
                display = "BT0 VIIRS"
            else:
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    "OTT", OTT_nodes)
                display = "IR (VIIRS)"
            vmin, vmax = -100, 50
            plot_data, is_rgb = ir_c, False
        else:
            r, g, b, _ = process_jpss_data(local_files, target_area,
                                           obs_dt, composite)
            if g is None:
                plot_data = np.nan_to_num(r, nan=0.0)
                if np.nanmax(plot_data) > 200:
                    plot_data = plot_data - 273.15
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        "OTT", OTT_nodes)
                    vmin, vmax = -100, 50
                else:
                    cmap = "gray"
                    vmin, vmax = 0, 1
                display, is_rgb = composite.upper(), False
            else:
                plot_data = stack_rgb(r, g, b)
                cmap = vmin = vmax = None
                display, is_rgb = composite, True

        src_tag = "PDS" if meta.get("source") == "pds" else "CLASS"
        sat_tag = f"VIIRS-{meta['sat']}"
        ts = obs_dt.strftime("%Y%m%d_%H%M")
        out_base = os.path.join(
            output_dir, f"{storm_id}_{ts}_{product}_{sat_tag}")
        metadata = {
            "satellite_name": f"JPSS/{src_tag}/"
                              f"{meta.get('family', 'VIIRS-SDR')}/"
                              f"{meta['sat']}",
            "target_dt": obs_dt, "center_lat": lat, "center_lon": lon,
            "crop_deg": max(lon_deg, lat_deg),
            "crop_lon": lon_deg, "crop_lat": lat_deg,
            "product": display, "storm_id": storm_id,
            "storm_name": storm.get("storm_name", ""),
            "winds": storm.get("winds"), "pressure": storm.get("pressure"),
            "grid": grid, "grid_thick": grid_thick,
            "grid_color": grid_color, "grid_style": grid_style,
            "no_coastlines": no_coastlines, "label": label,
            "par": False, "tcad": False, "tcid": False,
            "ico": False, "invest": False,
            "active_storms": None, "crop_km": crop_km,
            "polygon": storm.get("polygon"),
            "floater": floater, "info": info, "target_extent": extent,
        }

        if plotter is None:
            logging.warning("JPSS: no plotter provided; skipping render.")
        else:
            plotter(plot_data, out_base, metadata,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    logo_path=logo_path,
                    export_formats=export_formats or ["avif"],
                    is_rgb=is_rgb)
            logging.info(f"JPSS ({src_tag}): wrote {out_base}.*")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)