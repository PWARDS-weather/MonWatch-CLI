# =============================================================================
# rem_ingest/jpss_class.py — NOAA CLASS JPSS archive ingest for MonWatch-CLI
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
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import _download_session
from .jpss_common import (
    jpss_download_tar, jpss_extract_tar, jpss_select_closest_files,
)


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
        resp = _download_session.get(
            url if url.endswith("/") else url + "/", timeout=60)
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
    for m in re.finditer(
            r'href="((?:/downloads)?/JPSS/[^"]+\.(?:tar|nc|h5|hdf5|xml)(?:\.gz)?)"',
            html, re.I):
        href = m.group(1)
        name = href.split("/")[-1]
        full = "https://data.class.noaa.gov" + href
        entries.append((name, full))
    if not entries:
        for m in re.finditer(r'href="([^"]+)"', html):
            href = m.group(1)
            if href.startswith("?") or href.startswith("#") \
                    or "javascript" in href:
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
        logging.warning(f"JPSS: invalid date {target_date_str}; "
                        f"using latest {dates[-1]}")
        return dates[-1]
    best = min(dates, key=lambda d: abs(
        (datetime.datetime.strptime(d, "%Y%m%d").date() - target).days))
    if best != target_date_str:
        logging.info(f"JPSS: requested {target_date_str} not present; "
                     f"closest available is {best}")
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
    entries = _jpss_list_dir(
        f"{JPSS_CLASS_BASE}/{date_str}/{family}/{product}/")
    return [n.rstrip("/") for n, _ in entries if n.endswith("/")]


def jpss_list_tars(date_str, family, product, sat_id):
    entries = _jpss_list_dir(
        f"{JPSS_CLASS_BASE}/{date_str}/{family}/{product}/{sat_id}/")
    tars = []
    for name, url in entries:
        if name.lower().endswith(".tar") and "manifest" not in name.lower():
            tars.append((name, url))
    return tars

def jpss_resolve_product(date_str, family, preferred_product=None):
    products = jpss_list_products(date_str, family)
    if not products:
        logging.warning(f"JPSS: no products under {date_str}/{family}")
        return None
    if preferred_product:
        for p in products:
            if p.lower() == preferred_product.lower() \
                    or preferred_product.lower() in p.lower():
                return p
        logging.warning(f"JPSS: product '{preferred_product}' not found "
                        f"under {family}; available: {products[:12]}...")
    defaults = JPSS_FAMILY_DEFAULT_PRODUCTS.get(family, [])
    for d in defaults:
        for p in products:
            if d.lower() in p.lower() or p.lower() in d.lower():
                return p
    return products[0]


def jpss_resolve_sat(date_str, family, product, preferred_sat=None):
    sats = jpss_list_sats(date_str, family, product)
    if not sats:
        logging.warning(f"JPSS: no satellite dirs under "
                        f"{date_str}/{family}/{product}")
        return None

    alias = {
        "NOAA-21": "J02", "NOAA21": "J02", "J02": "J02", "N21": "J02",
        "NOAA-20": "J01", "NOAA20": "J01", "J01": "J01", "N20": "J01",
        "NPP": "NPP", "SNPP": "NPP", "S-NPP": "NPP", "NOAA": None,
    }
    if preferred_sat:
        key = preferred_sat.strip().upper().replace("_", "-")
        want = alias.get(key, key)
        if want is None:
            preferred_sat = None
        else:
            for s in sats:
                if s.upper() == want.upper():
                    logging.info(f"JPSS: selected satellite {s} "
                                 f"({preferred_sat})")
                    return s
            logging.warning(f"JPSS: requested {preferred_sat} ({want}) "
                            f"not under product; available {sats} "
                            f"— falling back to auto")

    for prefer, label in (("J02", "NOAA-21"), ("J01", "NOAA-20"),
                          ("NPP", "S-NPP"), ("npp", "S-NPP")):
        for s in sats:
            if s.upper() == prefer.upper():
                logging.info(f"JPSS: auto-selected satellite {s} ({label})")
                return s
    logging.info(f"JPSS: using first available satellite dir {sats[0]}")
    return sats[0]

def discover_jpss_files(family, target_dt=None, date_str=None, time_str=None,
                        product=None, sat_id=None,
                        center_lat=None, center_lon=None):
    if target_dt is None and date_str:
        try:
            if time_str:
                target_dt = datetime.datetime.strptime(
                    date_str + time_str[:4], "%Y%m%d%H%M")
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
            m = next((f for f in families
                      if family.upper() in f.upper()
                      or f.upper() in family.upper()), None)
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
            f"JPSS family '{family}' not found on CLASS "
            f"(checked {min(21, len(ordered))} day(s)). "
            f"Latest sample: {last_families}. "
            f"Try --JPSS-GRAN or --date with an older day.")
        return None

    avail_date, family = chosen_date, matched_family
    logging.info(
        f"JPSS: using CLASS date {avail_date} / family {family}"
        + (f" (requested {date_str})"
           if date_str and date_str != avail_date else "")
        + (" [nearest day with this family]"
           if not date_str or date_str != avail_date else ""))

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
        logging.warning(f"JPSS: no TAR files under "
                        f"{avail_date}/{family}/{prod}/{sat}")
        return None
    logging.info(f"JPSS: found {len(tars)} TAR(s)")

    return {
        "date": avail_date, "family": family, "product": prod, "sat": sat,
        "tars": tars, "target_dt": target_dt,
    }

def download_jpss_and_extract(meta, work_dir, download_workers=4,
                              max_tars=2):
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
        with ThreadPoolExecutor(
                max_workers=min(download_workers, len(tasks)),
                thread_name_prefix="JPSS") as ex:
            futures = [ex.submit(jpss_download_tar, u, p) for u, p in tasks]
            for f in as_completed(futures):
                f.result()

    all_files = []
    for lt in local_tars:
        if not os.path.exists(lt):
            continue
        extract_dir = os.path.join(
            work_dir, "extract_" + os.path.basename(lt).replace(".tar", ""))
        all_files.extend(jpss_extract_tar(lt, extract_dir))
    if not all_files:
        return None
    return jpss_select_closest_files(all_files, meta.get("target_dt"),
                                     max_files=24)