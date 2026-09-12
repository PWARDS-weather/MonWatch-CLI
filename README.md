# MonWatch-CLI

**A.S.T.I.G. — Automated Satellite Tracking & Imagery Generator**

CLI tool for servers and automated systems. Part of the PWARDS ecosystem. Generates storm-centered (or region-centered) satellite imagery from multiple geostationary satellites without a GUI.

> **Developed by [PWARDS-weather](https://github.com/PWARDS-weather)** — Pasacao Weather Atmospheric and Real-Time Data System  
> **Established**: 2025  
> Dual-licensed Apache-2.0 / GPL-3.0 (same as MonWatch-UI)

---

<img width="1280" height="640" alt="MONWATCH-CLI_POSTER" src="https://github.com/user-attachments/assets/154cb991-a5e8-482d-ab85-bd42a21de623" />

---

<h1 align="center" style="font-size: 3rem; font-weight: 900;">
  <img 
    width="32" 
    height="32" 
    alt="MONWATCH-CLI" 
    src="https://github.com/user-attachments/assets/c6a21b61-6238-44b8-8719-26bb166c45bf"
    style="vertical-align: middle; margin-right: 8px;"
  >
  <img 
    width="32" 
    height="32" 
    alt="splash" 
    src="https://github.com/user-attachments/assets/a7821feb-77fc-4c9d-906a-3cb48aa0e555"
    style="vertical-align: middle; margin-right: 8px;"
  >
 MonWatch-CLI SERVER AUTOMATION SYSTEM
</h1>

**Release date:** September 9, 2026<br>
**Last Updated:** September 11, 2026<br>
**Version:** 1.3

---

## What's New in 1.3

- **JPSS / VIIRS polar-orbiting support** — New `--VIIRS-SDR`, `--VIIRS-EDR`, `--JPSS-GRAN`, and `--VIIRSI-EDR` families sourced from NOAA CLASS, with optional `--jpss`, `--jpss-product`, and `--jpss-sat` filters.
- **NOAA-20 / NOAA-21 / S-NPP platform flags** — `--NOAA-20`, `--NOAA-21`, `--NOAA` (auto), `--NPP` for quick VIIRS platform selection.
- **Smarter `--auto-satellite`** — Probes every candidate satellite and picks the one with the *newest* available observation timestamp instead of the first responder.
- **GarbinWx identity resolution** — Now reads the API id from `GARBINWXID`, `GARBINWX_ID`, or `garbinwxid`, and a custom user-agent from `GARBINWXUSER` / `GARBINWX_USER`.
- **Radar type normalisation** — `RAIN` / `RAINRATE` aliases now correctly map to `RR`.

---

## Features

### Satellites

#### Geostationary
- **Himawari-8/9** (AHI) — Full Disk + Target (Region 3)
- **GK-2A** (AMI)
- **GOES-16/17/18/19** (auto East/West selection by longitude)
- **MTG** (EUMETSAT FCI, requires credentials)
- **MTSAT-1R / MTSAT-2** (historical HRIT from CEReS)

#### Polar-Orbiting (NEW)
- **JPSS / VIIRS** via NOAA CLASS:
  - `--VIIRS-SDR` — VIIRS Sensor Data Records (native radiances)
  - `--VIIRS-EDR` — VIIRS Environmental Data Records (cloud mask, surface reflectance)
  - `--JPSS-GRAN` — JPSS granule EDRs (VIIRS + ATMS + OMPS bundles)
  - `--VIIRSI-EDR` — VIIRS Imagery EDRs
  - `--jpss <FAMILY>` — explicit CLASS family name
  - `--jpss-product <NAME>` — pin a specific CLASS product subfolder
  - `--jpss-sat <ID>` — pin a satellite (`J01` = NOAA-20, `J02` = NOAA-21, `NPP` = S-NPP)

#### Auto-Selection
- **`--auto-satellite`** — Ranks candidate satellites (Himawari, GK-2A, GOES, MTG) by viewing geometry, probes **all** of them, then picks the one with the **newest available observation** at (or near) the requested timestamp.

### Products
`sandwich`, `true`, `dvorak`, `ir` (BT PWARDS), `infrared`, `z1-ir`, `althea-ott2`, `z1-true`, `z1-dvorak`, `b03`, `irv`, `bt0`, `falsecolor`, `falsecoloradv`, `firetemp`, `dayconv`, `fire`

Batch mode via `--products`.  
Polar-orbiting (VIIRS) composites fall back gracefully to the closest available granule time.

### Storm & Region Targeting
- Live ATCF via KnackWx API
- Historical tracks via IBTrACS (`--year` / date-based)
- **Basin filtering** (`--filter`) — Process only storms in specified basins (e.g. `WP`, `EPAC`, `AL`, `IO`, `SH`); accepts aliases and normalizes input.
- **All active storms processed by default** — no implicit WPAC-only restriction; use `--filter` to narrow.
- Named regions: Philippines, WestPac, CONUS, East/West Coast, Gulf, Caribbean, Europe, Mediterranean, Africa, etc.
- Custom lat/lon, polygon crops (PAR, TCAD, TCID, North/South Luzon, Manila, etc.)
- Peak-intensity time (`--peak`)

### Radar
- **GarbinWx** composite (`--garbinradar`, requires `GARBINWX_ID` / `GARBINWXID` / `garbinwxid`)
- **PAGASA Panahon** national mosaic (`--phradar`)
- Overlay on satellite or standalone floater view

### Export
- Formats: AVIF, PNG, JPG, WebP, MP4
- Configurable FPS, width, floater mode (colorbar + edge coords)
- Logo + info overlay, lat/lon grid, coastlines

### Automation
- Date/time or range (`--datefrom`/`--dateto`/`--timefrom`/`--timeto`)
- Multi-worker download + decompress
- Prefetch cache
- Global multi-satellite stacked mode (`--global`)
- Multi-sat run (`--multi`)

---

## Quick Usage

### Geostationary
```bash
# Latest sandwich for all active storms
python MonWatch-CLI.py --product sandwich --him

# Only Western Pacific storms
python MonWatch-CLI.py --product sandwich --him --filter WP

# Let MonWatch-CLI pick the best satellite per storm (newest observation wins)
python MonWatch-CLI.py --auto-satellite --product sandwich

# Specific storm + historical year
python MonWatch-CLI.py --storm HAIYAN --year 2013 --product dvorak --him

# GOES CONUS true color
python MonWatch-CLI.py --region conus --product true --goes19

# Radar floater
python MonWatch-CLI.py --phradar --floater --lat 14.5 --lon 121.0

# Batch products + MP4
python MonWatch-CLI.py --storm 01W --products sandwich,ir,dvorak --export avif,mp4 --fps 15
```
---
## Polar-Orbiting (JPSS / VIIRS) — NEW
```
# VIIRS SDR IR for a storm (auto picks NOAA-21 → NOAA-20 → S-NPP)
python MonWatch-CLI.py --storm 01W --VIIRS-SDR --product infrared

# Pin a specific satellite and product
python MonWatch-CLI.py --storm 01W --jpss VIIRS-SDR \
    --jpss-sat J02 --jpss-product VIIRS-Moderate-Resolution-Band-15-SDR \
    --product infrared

# VIIRS EDR composites
python MonWatch-CLI.py --storm 01W --VIIRS-EDR --products falsecolor,true

# Platform shortcut (implies VIIRS-SDR)
python MonWatch-CLI.py --storm 01W --NOAA-21 --product ir

# Auto JPSS satellite selection
python MonWatch-CLI.py --storm 01W --NOAA --product sandwich
```
---
# Environment Variables

MonWatch-CLI reads credentials and API identifiers from environment variables. A `.env` file placed in the same directory as the script is **loaded automatically** at startup, so you can keep secrets out of your shell history.

## Supported Variables

| Variable | Purpose |
|---|---|
| `EUMETSAT_CONSUMER_KEY` | MTG / EUMETSAT API key |
| `EUMETSAT_CONSUMER_SECRET` | MTG / EUMETSAT API secret |
| `GARBINWXID` / `GARBINWX_ID` / `garbinwxid` | GarbinWx radar composite id |
| `GARBINWXUSER` / `GARBINWX_USER` | Optional custom user-agent for GarbinWx |

> **Note:** Any one of the aliases listed for GarbinWx is accepted. If multiple are set, `GARBINWXID` takes precedence over `GARBINWX_ID`, which takes precedence over `garbinwxid`.

---

## `.env` File

Place a `.env` file next to `MonWatch-CLI.py` (or in the current working directory). Lines use `KEY=VALUE` format; blank lines and lines starting with `#` are ignored. Quotes around values are optional.

```dotenv
# EUMETSAT (MTG)
EUMETSAT_CONSUMER_KEY=your-consumer-key
EUMETSAT_CONSUMER_SECRET=your-consumer-secret

# GarbinWx radar
GARBINWX_ID=your-garbinwx-id
GARBINWXUSER=your-garbinwx-user-agent
```
---

## Acknowledgments

- **NOAA** — National Oceanic and Atmospheric Administration for Himawari data via AWS
- **JMA** — Japan Meteorological Agency for tropical cyclone forecast data
- **NHC** — National Hurricane Center for Atlantic/EPAC storm data
- **JTWC** — Joint Typhoon Warning Center for Western Pacific cyclone data
- **PAGASA** — Philippine Atmospheric, Geophysical and Astronomical Services Administration (including Panahon radar)
- **CWA** — Taiwan Central Weather Administration
- **EUMETSAT / OSI SAF** — SATELLITE DATA
- **GarbinWx** — Radar composite data (data.garbinwx.org)
- **CEReS / Chiba University** — Historical MTSAT HRIT archive
- **KnackWx** — ATCF tropical cyclone track API

---

> ### Please note that the program is still under active development, and some features are still being improved. If you encounter any issues, reporting them would be greatly appreciated.

---

> This is a branch of the traditional UI version. While the original UI was developed first, this version was created to accelerate R&D. Some features were temporarily removed during this shift, but they will be restored shortly.

*MonWatch-CLI Server Automation System — PWARDS ECOSYSTEM — © 2025-2026 PWARDS-weather*
