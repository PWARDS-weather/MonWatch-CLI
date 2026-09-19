#!/usr/bin/env python3
"""
ahi_segment_latitudes.py

Download every Himawari-8/9 AHI full-disk segment for one slot and print
the actual latitude range each segment covers.

Usage:
    python ahi_segment_latitudes.py                     # latest slot
    python ahi_segment_latitudes.py 20260919 0820       # explicit (UTC)
    python ahi_segment_latitudes.py 20260919 0820 B03   # other band
"""

import os
import re
import sys
import bz2
import shutil
import tempfile
import datetime
import logging

import numpy as np
import s3fs
import requests
from satpy import Scene


DEFAULT_BAND   = "B13"
DEFAULT_BUCKET = "noaa-himawari9"
SEGMENTS       = [f"S{i:02d}" for i in range(1, 11)]
MAX_BACK       = 60

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

fs = s3fs.S3FileSystem(anon=True)


def _folder(bucket, dt):
    return (f"{bucket}/AHI-L1b-FLDK/"
            f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/"
            f"{dt.hour:02d}{dt.minute:02d}/")


def _has_all_segments(files, band):
    for n in range(1, 11):
        seg = f"S{n:02d}"
        if not any(re.search(rf"_{seg}\d{{2}}\.DAT", f) and f"_{band}_" in f
                   for f in files):
            return False
    return True


def find_latest(bucket, band, max_back=MAX_BACK):
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    dt = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
    for i in range(max_back):
        try:
            files = fs.ls(_folder(bucket, dt))
            if _has_all_segments(files, band):
                return dt, files
        except FileNotFoundError:
            pass
        dt -= datetime.timedelta(minutes=10)
        if i % 6 == 5:
            logging.info(f"  ...searched {i + 1} slot(s), now at "
                         f"{dt:%Y-%m-%d %H:%M}Z")
    return None, None


def discover(bucket, dt, band):
    files = fs.ls(_folder(bucket, dt))
    out = {}
    for f in files:
        if f"_{band}_" not in f:
            continue
        for n in range(1, 11):
            seg = f"S{n:02d}"
            if re.search(rf"_{seg}\d{{2}}\.DAT", f):
                out[seg] = f
                break
    return out


def _https_url(s3_path):
    bucket, _, key = s3_path.partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def fetch(s3_path, out_dir):
    base = os.path.basename(s3_path)
    bz2_path = os.path.join(out_dir, base)
    dat_path = os.path.join(out_dir, base[:-4])
    if os.path.exists(dat_path):
        return dat_path

    logging.info(f"downloading {base}")
    r = requests.get(_https_url(s3_path), stream=True, timeout=120)
    r.raise_for_status()
    with open(bz2_path, "wb") as fh:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if chunk:
                fh.write(chunk)

    logging.info(f"decompressing {base}")
    with bz2.open(bz2_path, "rb") as fin, open(dat_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(bz2_path)
    return dat_path


def lat_range(dat_path, band):
    """
    The ahi_hsd reader pads each segment into the full-disk 5500x5500 grid
    and fills the rest with NaN.  We must therefore use the *data* mask,
    not the lat array, to find the pixels belonging to this segment.
    """
    scn = Scene(filenames=[dat_path], reader="ahi_hsd")
    scn.load([band])
    data = scn[band]
    area = data.attrs.get("area")
    if area is None:
        return None

    d = data.data
    if hasattr(d, "compute"):
        d = d.compute()
    d = np.asarray(d)

    _, lats = area.get_lonlats()
    lats = np.asarray(lats, dtype=np.float64)

    if d.shape != lats.shape:
        logging.warning(f"shape mismatch: data {d.shape} vs lats {lats.shape}")
        return None

    mask = np.isfinite(d)
    if not mask.any():
        return None

    row_has = mask.any(axis=1)
    col_has = mask.any(axis=0)
    r_lo = int(np.argmax(row_has))
    r_hi = int(len(row_has) - np.argmax(row_has[::-1]) - 1)
    c_lo = int(np.argmax(col_has))
    c_hi = int(len(col_has) - np.argmax(col_has[::-1]) - 1)

    seg_lats = lats[mask]
    return (float(seg_lats.min()), float(seg_lats.max()),
            d.shape, (r_lo, r_hi, c_lo, c_hi))


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    date_str = time_str = None
    band = DEFAULT_BAND
    bucket = DEFAULT_BUCKET

    if len(positional) >= 2:
        date_str, time_str = positional[0], positional[1]
    rest = positional[2:]
    if rest and rest[0].upper().startswith("B"):
        band = rest[0].upper()
        rest = rest[1:]
    if rest:
        bucket = rest[0]

    if date_str and time_str:
        dt = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
        logging.info(f"using requested slot {dt:%Y-%m-%d %H:%M}Z")
    else:
        logging.info(f"searching for latest slot on {bucket} ...")
        dt, _ = find_latest(bucket, band)
        if dt is None:
            logging.error("no complete slot found in search window")
            sys.exit(1)
        logging.info(f"latest complete slot: {dt:%Y-%m-%d %H:%M}Z")

    paths = discover(bucket, dt, band)
    missing = [s for s in SEGMENTS if s not in paths]
    if missing:
        logging.warning(f"missing segments: {missing}")
    if not paths:
        logging.error("nothing found")
        sys.exit(1)

    logging.info(f"{len(paths)} file(s) discovered")

    tmpdir = tempfile.mkdtemp(prefix="ahi_segs_")
    results = {}
    try:
        for seg in SEGMENTS:
            if seg not in paths:
                continue
            try:
                dat = fetch(paths[seg], tmpdir)
                rng = lat_range(dat, band)
                if rng is None:
                    logging.warning(f"{seg}: no usable area definition")
                    continue
                lo, hi, shape, rc = rng
                results[seg] = (lo, hi, shape, rc)
                logging.info(
                    f"{seg}: lat {lo:+7.3f}° to {hi:+7.3f}°  "
                    f"rows {rc[0]:>4}..{rc[1]:<4}  "
                    f"cols {rc[2]:>4}..{rc[3]:<4}"
                )
            except Exception as e:
                logging.error(f"{seg}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    print(f"AHI segment latitude coverage — {bucket} — "
          f"{dt:%Y-%m-%d %H:%M}Z — {band}")
    print("-" * 78)
    print(f"{'Segment':<8}{'Lat min':>10}{'Lat max':>10}"
          f"{'Centre':>10}{'Span':>8}"
          f"{'Row range':>18}{'Col range':>18}")
    print("-" * 78)
    for seg in SEGMENTS:
        if seg not in results:
            print(f"{seg:<8}{'-- missing --':>54}")
            continue
        lo, hi, _, rc = results[seg]
        c = 0.5 * (lo + hi)
        s = hi - lo
        print(f"{seg:<8}{lo:>+10.3f}{hi:>+10.3f}{c:>+10.3f}{s:>8.3f}"
              f"{f'{rc[0]}..{rc[1]}':>18}"
              f"{f'{rc[2]}..{rc[3]}':>18}")
    print("-" * 78)


if __name__ == "__main__":
    main()