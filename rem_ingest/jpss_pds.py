# =============================================================================
# rem_ingest/jpss_pds.py — NESDIS JPSS PDS (S3) ingest for MonWatch-CLI
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
import logging
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .jpss_common import (
    VIIRS_HALF_SWATH_KM,
    _jpss_pds_list_keys,
    _jpss_extract_orbit,
    _jpss_parse_granule_time_from_name,
    _jpss_dist_to_box_km,
    _jpss_get_orbital,
    _jpss_find_passes,
    _jpss_rank_passes,
    _jpss_check_geo_coverage,
    jpss_download_tar,
    jpss_select_closest_files,
)


JPSS_PDS_BUCKETS = {
    "n20": "noaa-nesdis-n20-pds",
    "n21": "noaa-nesdis-n21-pds",
    "snpp": "noaa-nesdis-snpp-pds",
    "j01": "noaa-nesdis-n20-pds",
    "j02": "noaa-nesdis-n21-pds",
    "npp": "noaa-nesdis-snpp-pds",
    "noaa-20": "noaa-nesdis-n20-pds",
    "noaa-21": "noaa-nesdis-n21-pds",
    "noaa20": "noaa-nesdis-n20-pds",
    "noaa21": "noaa-nesdis-n21-pds",
    "s-npp": "noaa-nesdis-snpp-pds",
}
JPSS_PDS_SAT_TOKEN = {
    "noaa-nesdis-n20-pds": "j01",
    "noaa-nesdis-n21-pds": "j02",
    "noaa-nesdis-snpp-pds": "npp",
}
JPSS_PDS_SAT_LABEL = {"j01": "NOAA-20", "j02": "NOAA-21", "npp": "S-NPP"}
JPSS_PDS_BUCKET_TO_KEY = {
    "noaa-nesdis-n20-pds": "n20",
    "noaa-nesdis-n21-pds": "n21",
    "noaa-nesdis-snpp-pds": "snpp",
}

JPSS_PDS_COMPOSITE_BANDS = {
    "infrared": ["I05"], "ir": ["I05"], "dvorak": ["I05"],
    "z1-ir": ["I05"], "z1-dvorak": ["I05"],
    "althea-ott2": ["I05"], "bt0": ["I05"],
    "b03": ["I01"],
    "sandwich": ["I05", "I01"], "irv": ["I05", "I01"],
    "falsecolor": ["I05", "I01"], "falsecoloradv": ["I05", "I01"],
    "true": ["I05", "I01"], "z1-true": ["I05", "I01"],
}

VIIRS_PRODUCT_INFO = {
    "I01": ("VIIRS-I1-SDR", "SVI01", "VIIRS-IMG-GEO-TC", "GITCO"),
    "I02": ("VIIRS-I2-SDR", "SVI02", "VIIRS-IMG-GEO-TC", "GITCO"),
    "I03": ("VIIRS-I3-SDR", "SVI03", "VIIRS-IMG-GEO-TC", "GITCO"),
    "I04": ("VIIRS-I4-SDR", "SVI04", "VIIRS-IMG-GEO-TC", "GITCO"),
    "I05": ("VIIRS-I5-SDR", "SVI05", "VIIRS-IMG-GEO-TC", "GITCO"),
    "M15": ("VIIRS-M15-SDR", "SVM15", "VIIRS-MOD-GEO-TC", "GMTCO"),
    "M05": ("VIIRS-M5-SDR", "SVM05", "VIIRS-MOD-GEO-TC", "GMTCO"),
    "DNB": ("VIIRS-DNB-SDR", "SVDNB", "VIIRS-DNB-GEO", "GDNBO"),
}

def _jpss_pds_resolve_bucket(sat_id=None):
    if not sat_id:
        return None
    key = sat_id.strip().lower().replace("_", "-")
    if key in JPSS_PDS_BUCKETS:
        return JPSS_PDS_BUCKETS[key]
    alias = {
        "NOAA-21": "n21", "NOAA21": "n21", "J02": "n21", "N21": "n21",
        "NOAA-20": "n20", "NOAA20": "n20", "J01": "n20", "N20": "n20",
        "NPP": "snpp", "SNPP": "snpp", "S-NPP": "snpp",
    }
    mapped = alias.get(sat_id.strip().upper().replace("_", "-"))
    return JPSS_PDS_BUCKETS.get(mapped) if mapped else None

def _jpss_pds_bucket_candidates(sat_id=None):
    preferred = _jpss_pds_resolve_bucket(sat_id)
    order = ["noaa-nesdis-n21-pds",
             "noaa-nesdis-n20-pds",
             "noaa-nesdis-snpp-pds"]
    if preferred:
        return [preferred] + [b for b in order if b != preferred]
    return order

def discover_jpss_pds_files(composite_type="infrared", target_dt=None,
                            date_str=None, time_str=None, sat_id=None,
                            center_lat=None, center_lon=None, crop_km=1000,
                            max_passes=6, search_window_hours=14):
    if target_dt is None:
        if date_str:
            try:
                tpart = (time_str or "1200")[:4]
                target_dt = datetime.datetime.strptime(
                    date_str[:8] + tpart, "%Y%m%d%H%M")
            except ValueError:
                target_dt = datetime.datetime.now(
                    datetime.timezone.utc).replace(tzinfo=None)
        else:
            target_dt = datetime.datetime.now(
                datetime.timezone.utc).replace(tzinfo=None)

    if center_lat is None or center_lon is None:
        logging.warning("JPSS PDS: lat/lon required for pass selection")
        return None

    if crop_km and float(crop_km) > 0:
        crop_deg = float(crop_km) / 111.32 / 2.0
    else:
        crop_deg = 1.0
    crop_deg = max(0.3, crop_deg)
    asr = 1.0

    bands = JPSS_PDS_COMPOSITE_BANDS.get(
        (composite_type or "infrared").lower().strip(), ["I05"])
    primary = bands[0] if bands[0] in VIIRS_PRODUCT_INFO else "I05"
    sdr_product, sdr_prefix, geo_product, geo_prefix = \
        VIIRS_PRODUCT_INFO[primary]

    buckets = _jpss_pds_bucket_candidates(sat_id)
    for bucket in buckets:
        sat_token = JPSS_PDS_SAT_TOKEN.get(bucket, "j01")
        sat_key = JPSS_PDS_BUCKET_TO_KEY.get(bucket, "n21")
        try:
            orb = _jpss_get_orbital(sat_key, dt_obj=target_dt)
        except Exception as e:
            logging.warning(f"JPSS PDS: TLE/orbit unavailable for "
                            f"{sat_key}: {e}")
            continue

        all_passes = _jpss_find_passes(
            orb, target_dt, center_lat, center_lon, crop_deg, asr=asr,
            search_window_hours=search_window_hours, step_seconds=5)
        if not all_passes:
            logging.info(
                f"JPSS PDS: no {sat_key} pass within +/-{search_window_hours}h "
                f"of {target_dt} over ({center_lat:.2f},{center_lon:.2f})")
            continue

        selected = _jpss_rank_passes(
            orb, all_passes, target_dt, center_lat, center_lon, crop_deg,
            asr=asr, max_passes=max_passes, min_coverage=0.97)
        logging.info(
            f"JPSS PDS: {sat_key} selected {len(selected)} prioritized "
            f"pass(es) (of {len(all_passes)} candidates)")

        listing = {}

        def _ls(product, day):
            key = (product, day)
            if key not in listing:
                prefix = f"{product}/{day.year}/{day.month:02d}/{day.day:02d}/"
                listing[key] = _jpss_pds_list_keys(bucket, prefix, max_keys=2000)
            return listing[key]

        days = set()
        for ps, pe in selected:
            days.add(ps.date())
            days.add(pe.date())

        pass_granules_grouped = []
        seen_paths = set()
        for ps, pe in selected:
            pass_valid = []
            for day in days:
                sdr_files = _ls(sdr_product, day)
                geo_files = _ls(geo_product, day)
                for sk in sdr_files:
                    if sk in seen_paths:
                        continue
                    fname = sk.split("/")[-1]
                    if not fname.startswith(sdr_prefix):
                        continue
                    fl = fname.lower()
                    if sat_token == "npp":
                        if "_npp_" not in fl:
                            continue
                    elif f"_{sat_token}_" not in fl:
                        continue
                    g_time = _jpss_parse_granule_time_from_name(fname)
                    orbit = _jpss_extract_orbit(fname)
                    if g_time is None or orbit is None:
                        continue
                    if not (ps <= g_time <= pe):
                        continue
                    seen_paths.add(sk)
                    geo_matches = [
                        g for g in geo_files
                        if g.split("/")[-1].startswith(geo_prefix)
                        and f"_b{orbit}_" in g
                    ]
                    if not geo_matches:
                        continue
                    best_geo = min(
                        geo_matches,
                        key=lambda g: abs(
                            ((_jpss_parse_granule_time_from_name(
                                g.split("/")[-1]) or g_time) - g_time
                             ).total_seconds()),
                    )
                    try:
                        sub_lon, sub_lat, _ = orb.get_lonlatalt(
                            g_time + datetime.timedelta(seconds=43))
                        if float(_jpss_dist_to_box_km(
                                sub_lat, sub_lon, center_lat, center_lon,
                                crop_deg, asr=asr)[0]) > \
                           (VIIRS_HALF_SWATH_KM + 350.0):
                            continue
                    except Exception:
                        continue
                    if _jpss_check_geo_coverage(bucket, best_geo,
                                                center_lat, center_lon,
                                                crop_deg, asr=asr):
                        pass_valid.append((g_time, orbit, sk, best_geo))
            if pass_valid:
                pass_valid.sort(key=lambda c: c[0])
                pass_granules_grouped.append(pass_valid)

        if not pass_granules_grouped:
            logging.info(f"JPSS PDS: {sat_key} passes found but no covering "
                         f"granules")
            continue

        anchor = min(pass_granules_grouped[0],
                     key=lambda c: abs((c[0] - target_dt).total_seconds()))
        anchor_time = anchor[0]

        keep = []
        primary_pass = pass_granules_grouped[0]
        primary_sorted = sorted(
            primary_pass,
            key=lambda c: abs((c[0] - anchor_time).total_seconds()))
        keep.extend(primary_sorted[:3])
        for extra in pass_granules_grouped[1:]:
            if extra:
                keep.append(min(extra,
                                key=lambda c: abs((c[0] - target_dt).total_seconds())))

        remote_keys = []
        for g_time, orbit, sk, gk in keep:
            remote_keys.append(sk)
            remote_keys.append(gk)
            for bt in bands[1:]:
                if bt not in VIIRS_PRODUCT_INFO:
                    continue
                bp, bpre, _gp, _gpre = VIIRS_PRODUCT_INFO[bt]
                day = g_time.date()
                for cand in _ls(bp, day):
                    if cand.split("/")[-1].startswith(bpre) and \
                       f"_b{orbit}_" in cand:
                        remote_keys.append(cand)
                        break

        seen_k, ordered = set(), []
        for k in remote_keys:
            if k not in seen_k:
                seen_k.add(k)
                ordered.append(k)

        logging.info(
            f"JPSS PDS: s3://{bucket}/ {sdr_product}+{geo_product} "
            f"sat={sat_token} ({JPSS_PDS_SAT_LABEL.get(sat_token, sat_token)}) "
            f"files={len(ordered)} "
            f"closest={anchor_time.strftime('%Y-%m-%d %H:%M')}Z "
            f"target=({center_lat:.2f},{center_lon:.2f})")

        return {
            "source": "pds",
            "bucket": bucket,
            "date": anchor_time.strftime("%Y%m%d"),
            "family": "VIIRS-SDR",
            "product": sdr_product,
            "geo_product": geo_product,
            "sat": sat_token.upper() if sat_token != "npp" else "NPP",
            "sat_token": sat_token,
            "remote_keys": ordered,
            "target_dt": anchor_time,
        }

    logging.warning(
        f"JPSS PDS: no covering granules for composite={composite_type} "
        f"dt={target_dt} sat={sat_id or 'auto'} "
        f"at ({center_lat},{center_lon})")
    return None


def download_jpss_pds_files(meta, work_dir, download_workers=8):
    os.makedirs(work_dir, exist_ok=True)
    bucket = meta["bucket"]
    keys = meta.get("remote_keys") or []
    if not keys:
        return None
    tasks, local_paths = [], []
    for key in keys:
        lpath = os.path.join(work_dir, os.path.basename(key))
        local_paths.append(lpath)
        if not os.path.exists(lpath):
            tasks.append((f"https://{bucket}.s3.amazonaws.com/{key}", lpath))
    if tasks:
        logging.info(f"JPSS PDS: downloading {len(tasks)} file(s) "
                     f"from {bucket}...")
        with ThreadPoolExecutor(
                max_workers=min(download_workers, max(len(tasks), 1)),
                thread_name_prefix="JPSS-PDS") as ex:
            futs = [ex.submit(jpss_download_tar, u, p) for u, p in tasks]
            for f in as_completed(futs):
                f.result()
    existing = [p for p in local_paths if os.path.exists(p)]
    return (jpss_select_closest_files(existing, meta.get("target_dt"),
                                      max_files=16)
            if existing else None)