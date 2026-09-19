#!/usr/bin/env python3
# =============================================================================
# gro_ingest/garbinradar.py — GARBINWX ingest for MonWatch-CLI
# (C) 2025-2026 PWARDS-weather
# SPDX-License-Identifier: Apache-2.0 OR GPL-3.0-or-later
#
# Dual-licensed. You may use, modify, and distribute this package under the
# terms of EITHER the Apache License, Version 2.0, or the GNU General Public
# License, Version 3.0 or later — not both. See LICENSE.txt in the project
# root for the full license texts and the copyright notice.
# =============================================================================

import os
import io
import re
import logging
import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors


GARBIN_RADAR_BASE = "https://data.garbinwx.org/raw"
GARBIN_TIME_REFERENCE_URL = "https://data.garbinwx.org/latest/timeReference10.json"
GARBIN_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/152.0.0.0 Safari/537.36")
GARBIN_BROWSER_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://garbinwx.org",
    "referer": "https://garbinwx.org/",
    "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": GARBIN_USER_AGENT,
}
GARBIN_RADAR_BOUNDS = [115.41549141305251, 3.801613036809332,
                       129.51730887177652, 22.45850950564088]

HEX_COLORS_DBZ = [
    '#535353', '#5b5b5b', '#606060', '#6e6e6e', '#797979', '#828282',
    '#8a8a8a', '#939393', '#9b9b9b', '#a1a1a1', '#aaaaaa', '#b9b9b9',
    '#c1c1c1', '#c8c8c8', '#cecece', '#00ff00', '#00f500', '#00e600',
    '#00dc00', '#00d200', '#00c800', '#00be00', '#00b400', '#00aa00',
    '#00a000', '#009600', '#32aa00', '#64be00', '#96d200', '#cdeb00',
    '#ffff00', '#fff500', '#ffe600', '#ffdc00', '#ffd200', '#ffc800',
    '#ffb900', '#ffaa00', '#ff9600', '#ff8700', '#ff7800', '#ff5f00',
    '#ff4600', '#ff3200', '#ff1900', '#ff0000', '#ff0000', '#e60000',
    '#dc0000', '#d20000', '#c80000', '#be0000', '#b40000', '#aa0000',
    '#a00000', '#960000', '#aa0032', '#be0064', '#d70096', '#eb00cd',
    '#ff00ff', '#eb00ff', '#d200ff', '#be00ff', '#aa00ff', '#9600ff',
]


_download_session = requests.Session()
_dl_retries = Retry(total=5, backoff_factor=0.5,
                    status_forcelist=[500, 502, 503, 504])
_download_session.mount(
    "https://",
    HTTPAdapter(pool_connections=32, pool_maxsize=64,
                max_retries=_dl_retries),
)


def dbz_cmap():
    return mcolors.ListedColormap(HEX_COLORS_DBZ, name="garbin_dbz")


def identity():
    gid = (
        os.environ.get("GARBINWXID")
        or os.environ.get("GARBINWX_ID")
        or os.environ.get("garbinwxid")
    )
    ua = (
        os.environ.get("GARBINWXUSER")
        or os.environ.get("GARBINWX_USER")
        or GARBIN_USER_AGENT
    )
    return (gid.strip() if gid else None), (ua.strip() if ua else GARBIN_USER_AGENT)


def time_reference():
    url = GARBIN_TIME_REFERENCE_URL
    try:
        resp = _download_session.get(url, headers=GARBIN_BROWSER_HEADERS, timeout=30)
        if resp.status_code == 200:
            refs = resp.json().get("timeReference", [])
            out, seen = [], set()
            for r in refs:
                if not isinstance(r, str):
                    continue
                name = r[:-4] if r.endswith(".png") else r
                digits = re.sub(r"\D", "", name)
                if len(digits) < 12:
                    continue
                try:
                    dt = datetime.datetime.strptime(digits[:12], "%Y%m%d%H%M")
                except ValueError:
                    continue
                dt = dt.replace(minute=(dt.minute // 10) * 10,
                                second=0, microsecond=0)
                ts = dt.strftime("%Y%m%d%H%M")
                if ts not in seen:
                    seen.add(ts)
                    out.append(ts)
            return out
    except Exception as e:
        logging.warning(f"Could not fetch GarbinWx time reference: {e}")
    return []


def candidate_timestamps(max_n=18, step_min=10):
    refs = time_reference() or []
    now_pht = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8))
    ).replace(second=0, microsecond=0)
    now_pht = now_pht.replace(minute=(now_pht.minute // step_min) * step_min)

    synth = [
        (now_pht - datetime.timedelta(minutes=step_min * i)).strftime("%Y%m%d%H%M")
        for i in range(max_n)
    ]

    seen, out = set(), []
    for t in refs + synth:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_n:
            break
    return out


def fetch_radar(radar_type, ts, gid, output_dir):
    _, user_agent = identity()
    url = f"{GARBIN_RADAR_BASE}/{radar_type}-{ts}.png"
    headers = {"garbinwxid": gid, "user-agent": user_agent}
    out_path = os.path.join(output_dir, f"radar_{radar_type}_{ts}.png")
    try:
        resp = _download_session.get(url, headers=headers, stream=True, timeout=30)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(1024):
                    if chunk:
                        f.write(chunk)
            logging.info(f"Saved GarbinWx radar: {out_path}")
            return True
        logging.error(f"Failed to fetch GarbinWx radar {ts}: status {resp.status_code}")
        return False
    except Exception as e:
        logging.error(f"Failed to fetch GarbinWx radar {ts}: {e}")
        return False


def process_radar(output_dir, radar_type="DBZ", date_str=None, time_str=None):
    gid = os.environ.get("GARBINWX_ID") or os.environ.get("garbinwxid")
    if not gid:
        logging.error("--garbinradar requires a garbinwxid. Set GARBINWX_ID in a .env file "
                      "(e.g. GARBINWX_ID=YOUR-ID).")
        return False

    radar_type = (radar_type or "DBZ").upper()

    if date_str and time_str:
        try:
            ts = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M").strftime("%Y%m%d%H%M")
        except ValueError:
            logging.error(f"Invalid --date/--time for --garbinradar: {date_str} {time_str}")
            return False
        return fetch_radar(radar_type, ts, gid, output_dir)

    for ts in candidate_timestamps(max_n=18):
        if fetch_radar(radar_type, ts, gid, output_dir):
            return True
    return False


def radar_bytes(radar_type, date_str, time_str, gid):
    radar_type = (radar_type or "DBZ").upper()

    if time_str:
        if date_str:
            raw = date_str + time_str
        else:
            raw = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d") + time_str
        try:
            dt_utc = datetime.datetime.strptime(raw, "%Y%m%d%H%M").replace(tzinfo=datetime.timezone.utc)
            dt_pht = dt_utc + datetime.timedelta(hours=8)
            ts_list = [dt_pht.strftime("%Y%m%d%H%M")]
        except ValueError:
            logging.error(f"Invalid --date/--time for radar: date={date_str} time={time_str}")
            return None, None
    else:
        ts_list = candidate_timestamps(max_n=18)

    _gid, user_agent = identity()
    headers = {"garbinwxid": gid, "user-agent": user_agent}

    for i, ts in enumerate(ts_list):
        url = f"{GARBIN_RADAR_BASE}/{radar_type}-{ts}.png"
        try:
            resp = _download_session.get(url, headers=headers, stream=True, timeout=30)
            if resp.status_code == 200:
                if i > 0:
                    logging.info(f"GarbinWx radar: fell back {i} slot(s) to {ts}")
                return resp.content, ts
            logging.debug(f"GarbinWx radar {ts}: status {resp.status_code}")
        except Exception as e:
            logging.warning(f"GarbinWx radar {ts}: {e}")
    return None, None


def radar_overlay(radar_type, date_str, time_str):
    gid, _ua = identity()
    if not gid:
        logging.error("--garbinradar overlay requires a garbinwxid. Set GARBINWX_ID in a .env file.")
        return None
    png_bytes, ts = radar_bytes(radar_type, date_str, time_str, gid)
    if png_bytes is None:
        logging.error("No GarbinWx radar composite available for overlay.")
        return None
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        logging.error(f"Failed to decode radar composite: {e}")
        return None
    arr = np.asarray(img).astype(np.float32) / 255.0
    radar_type = (radar_type or "DBZ").upper()

    ts_utc = ts
    if ts:
        try:
            dt_pht = datetime.datetime.strptime(ts, "%Y%m%d%H%M")
            ts_utc = (dt_pht - datetime.timedelta(hours=8)).strftime("%Y%m%d%H%M")
        except ValueError:
            ts_utc = ts

    logging.info(f"GarbinWx radar overlay fetched (timestamp: {ts} PHT / {ts_utc} UTC)")
    return {
        "rgb": arr,
        "bounds": list(GARBIN_RADAR_BOUNDS),
        "ts": ts_utc,
        "source": "GarbinWx",
        "type": radar_type,
    }