#!/usr/bin/env python3
# =============================================================================
# gro_ingest/phradar.py — PAGASA ingest for MonWatch-CLI
# (C) 2025-2026 PWARDS-weather
# SPDX-License-Identifier: Apache-2.0 OR GPL-3.0-or-later
#
# Dual-licensed. You may use, modify, and distribute this package under the
# terms of EITHER the Apache License, Version 2.0, or the GNU General Public
# License, Version 3.0 or later — not both. See LICENSE.txt in the project
# root for the full license texts and the copyright notice.
# =============================================================================
import io
import re
import hmac
import hashlib
import secrets
import logging
import datetime

import requests
import numpy as np
from PIL import Image

from .garbinradar import HEX_COLORS_DBZ


PHRADAR_ORIGIN = "https://www.panahon.gov.ph"
PHRADAR_TOKEN_FALLBACK = "bH2qMl5ZJsRZEcgo32fk8VQlRN5X6K6eEGBVcOCm"
PHRADAR_BOUNDS = [115.41549141305251, 3.801613036809332,
                  129.51730887177652, 22.45850950564088]
PHRADAR_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
)


def colorize_la(png_bytes, scale=None, radar_type="DBZ"):
    try:
        img = Image.open(io.BytesIO(png_bytes))
    except Exception as e:
        logging.error(f"Panahon: cannot open radar image: {e}")
        return None
    if img.mode not in ("LA", "L", "RGBA", "RGB"):
        img = img.convert("LA")
    arr = np.asarray(img)
    if arr.ndim == 2:
        L = arr.astype(np.float32)
        A = np.full(L.shape, 255, dtype=np.float32)
    elif arr.shape[-1] == 2:
        L = arr[..., 0].astype(np.float32)
        A = arr[..., 1].astype(np.float32)
    elif arr.shape[-1] >= 3:
        rgba = arr.astype(np.float32)
        if rgba.shape[-1] == 3:
            alpha = np.ones(rgba.shape[:2] + (1,), dtype=np.float32) * 255.0
            rgba = np.concatenate([rgba, alpha], axis=-1)
        return np.clip(rgba / 255.0, 0.0, 1.0)
    else:
        logging.error(f"Panahon: unexpected image array shape {arr.shape}")
        return None

    scale = scale or {}
    vmax = float(scale.get("max") or 80.0)
    vmin = float(scale.get("min") or 0.0)
    use_sqrt = bool(scale.get("sqrt"))
    r = L / 255.0
    if use_sqrt:
        dbz = (r * r) * vmax
    else:
        dbz = vmin + r * (vmax - vmin)

    colors = np.zeros((66, 4), dtype=np.float32)
    for i, hx in enumerate(HEX_COLORS_DBZ):
        hx = hx.lstrip("#")
        colors[i, 0] = int(hx[0:2], 16) / 255.0
        colors[i, 1] = int(hx[2:4], 16) / 255.0
        colors[i, 2] = int(hx[4:6], 16) / 255.0
        colors[i, 3] = 1.0
    idx = np.clip(np.floor(dbz).astype(np.int32) - 1, 0, 65)
    rgba = colors[idx].copy()
    alpha = np.clip(A / 255.0, 0.0, 1.0)
    alpha = np.where(dbz < 1.0, 0.0, alpha)
    rgba[..., 3] = alpha
    return rgba


def hmac_sha256_hex(key, message):
    return hmac.new(key.encode("utf-8"), message.encode("utf-8"),
                    hashlib.sha256).hexdigest()


def session():
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": PHRADAR_USER_AGENT,
        "Accept": "*/*",
        "Referer": PHRADAR_ORIGIN + "/",
        "Origin": PHRADAR_ORIGIN,
    })
    home = sess.get(PHRADAR_ORIGIN + "/", timeout=30)
    home.raise_for_status()
    metas = {}
    for m in re.finditer(r"<meta[^>]+>", home.text, re.I):
        tag = m.group(0)
        name = re.search(r'name=["\']([^"\']+)["\']', tag, re.I)
        content = re.search(r'content=["\']([^"\']*)["\']', tag, re.I)
        if name and content:
            metas[name.group(1)] = content.group(1)
    csrf = metas.get("csrf-token")
    handle = metas.get("api-sig-handle")
    if not csrf or not handle:
        raise RuntimeError("Panahon: missing csrf-token / api-sig-handle meta tags")
    sig_url = f"{PHRADAR_ORIGIN}/api/v1/sig?token={csrf}"
    sr = sess.get(sig_url,
                  headers={"X-Sig-Handle": handle,
                           "X-Requested-With": "XMLHttpRequest"},
                  timeout=30)
    sr.raise_for_status()
    secret = (sr.json() or {}).get("secret")
    if not secret:
        raise RuntimeError("Panahon: /api/v1/sig returned no secret")
    sess._ph_csrf = csrf
    sess._ph_secret = secret

    def _signed_get(path, params=None, timeout=60):
        path_clean = path.strip("/")
        ts = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp()))
        nonce = secrets.token_hex(16)
        msg = f"GET\n{path_clean}\n{ts}\n{nonce}"
        headers = {
            "X-Ts": ts,
            "X-Nonce": nonce,
            "X-Sig": hmac_sha256_hex(secret, msg),
            "X-Requested-With": "XMLHttpRequest",
        }
        url = f"{PHRADAR_ORIGIN}/{path_clean}"
        return sess.get(url, params=params or {}, headers=headers, timeout=timeout)

    sess.ph_get = _signed_get
    return sess


def mode(radar_type):
    rt = (radar_type or "DBZ").upper()
    if rt in ("RR", "RAIN", "RAINRATE"):
        return "rain", "mosaic-rainrate"
    return "dbz", "mosaic-reflectivity"


def timeline(sess, sublayer="mosaic-reflectivity"):
    token = sess._ph_csrf or PHRADAR_TOKEN_FALLBACK
    resp = sess.ph_get("api/v1/radar/timeline",
                       {"token": token, "sublayer": sublayer}, timeout=30)
    if resp.status_code != 200:
        logging.warning(f"Panahon timeline HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    try:
        payload = resp.json()
    except Exception as e:
        logging.warning(f"Panahon timeline JSON error: {e}")
        return None
    if not payload.get("success"):
        logging.warning(f"Panahon timeline unsuccessful: {payload}")
        return None
    return payload.get("data") or {}


def pick_unix(data, date_str=None, time_str=None):
    timeline_entries = list(data.get("timeline") or [])
    if not timeline_entries:
        return None, None
    target = None
    if date_str and time_str:
        try:
            target = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M").replace(
                tzinfo=datetime.timezone.utc)
        except ValueError:
            target = None
    if target is not None:
        t_unix = int(target.timestamp())
        best = min(timeline_entries,
                   key=lambda e: abs(int(e.get("observed_at_unix") or 0) - t_unix))
        return int(best["observed_at_unix"]), best.get("observed_at")
    best = max(timeline_entries, key=lambda e: int(e.get("observed_at_unix") or 0))
    return int(best["observed_at_unix"]), best.get("observed_at")


def fetch_image_bytes(radar_type="DBZ", date_str=None, time_str=None, size=1536):
    mode_str, sublayer = mode(radar_type)
    try:
        sess = session()
    except Exception as e:
        logging.error(f"Panahon session bootstrap failed: {e}")
        return None, None, None, None
    data = timeline(sess, sublayer=sublayer)
    if not data:
        return None, None, None, None
    bounds = data.get("bounds") or PHRADAR_BOUNDS
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4):
        bounds = PHRADAR_BOUNDS
    t_unix, observed_at = pick_unix(data, date_str, time_str)
    if t_unix is None:
        logging.error("Panahon: no timeline frames available")
        return None, None, None, None
    token = sess._ph_csrf or PHRADAR_TOKEN_FALLBACK
    last_err = None
    for sz in (size, 1536, 1024, 512):
        try:
            resp = sess.ph_get("api/v1/radar-data-image", {
                "token": token, "t": t_unix, "mode": mode_str, "size": sz, "v": 5,
            }, timeout=90)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                logging.info(f"Panahon radar image: t={t_unix} ({observed_at}) "
                             f"mode={mode_str} size={sz} ({len(resp.content) / 1024:.0f} KB)")
                scale = data.get("scale") or {"mode": mode_str, "max": 80, "sqrt": False, "unit": "dBZ"}
                return resp.content, t_unix, list(bounds), scale
            last_err = f"HTTP {resp.status_code} {resp.headers.get('content-type')} {resp.text[:120]}"
        except Exception as e:
            last_err = str(e)
    logging.error(f"Panahon radar image failed: {last_err}")
    return None, None, None, None


def radar_bytes(radar_type, date_str, time_str):
    png, t_unix, bounds, scale = fetch_image_bytes(radar_type, date_str, time_str)
    if png is None:
        return None, None, None, None
    try:
        ts = datetime.datetime.fromtimestamp(t_unix, tz=datetime.timezone.utc).strftime("%Y%m%d%H%M")
    except Exception:
        ts = str(t_unix)
    return png, ts, bounds, scale


def radar_overlay(radar_type, date_str, time_str):
    png_bytes, ts, bounds, scale = radar_bytes(radar_type, date_str, time_str)
    if png_bytes is None:
        logging.error("No Panahon radar composite available for overlay.")
        return None
    rgba = colorize_la(png_bytes, scale=scale, radar_type=radar_type)
    if rgba is None:
        logging.error("Failed to colorize Panahon radar composite for overlay.")
        return None
    if not bounds or len(bounds) != 4:
        bounds = list(PHRADAR_BOUNDS)
    radar_type = (radar_type or "DBZ").upper()
    logging.info(f"Panahon radar overlay fetched (timestamp: {ts})")
    return {
        "rgb": rgba,
        "bounds": list(bounds),
        "ts": ts,
        "source": "PAGASA/Panahon",
        "type": radar_type,
    }