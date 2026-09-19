# MonWatch-CLI

**A.S.T.I.G. — Automated Satellite Tracking & Imagery Generator**

CLI tool for servers and automated systems. Part of the PWARDS ecosystem. Generates storm-centered (or region-centered) satellite imagery from multiple geostationary and polar-orbiting satellites without a GUI.

> **Developed by [PWARDS-weather](https://github.com/PWARDS-weather)** — Pasacao Weather Atmospheric and Real-Time Data System  
> **Established**: 2025  
> Dual-licensed Apache-2.0 / GPL-3.0 (same as MonWatch-UI)

---

<img width="1280" height="640" alt="MONWATCH-CLI_POSTER" src="https://github.com/user-attachments/assets/154cb991-a5e8-482d-ab85-bd42a21de623" />

---

<p align="center">
  <img width="32" height="32" alt="MONWATCH-CLI" src="https://github.com/user-attachments/assets/c6a21b61-6238-44b8-8719-26bb166c45bf" style="vertical-align: middle; margin-right: 8px;" />
  <img width="32" height="32" alt="splash" src="https://github.com/user-attachments/assets/a7821feb-77fc-4c9d-906a-3cb48aa0e555" style="vertical-align: middle; margin-right: 8px;" />
  <strong style="font-size: 1.75rem;">MonWatch-CLI SERVER AUTOMATION SYSTEM</strong>
</p>

**Release date:** September 9, 2026  
**Last Updated:** September 19, 2026  
**Version:** 1.4

---

## What's New in 1.4

- **Modular package layout** — Core logic split into dedicated packages for maintainability and future expansion:
  - `rem_ingest/` — Satellite data discovery, download, and processing (Himawari, GK-2A, GOES, MTG, MTSAT, JPSS/VIIRS, ASCAT, microwave)
  - `gro_ingest/` — Ground/radar observations (GarbinWx, PAGASA Panahon, NEXRAD, dropsonde, recon, surface obs)
  - `controllers/` — Generation controllers (image, track, forecast, all) — *scaffolding under construction*
  - `styles/` + `track_style/` — Agency-specific rendering styles (JMA, JTWC, NHC, PAGASA, McIDAS, modern)
  - `tracks/` — Track ingest modules (JMA, JTWC, NHC, PAGASA, CWA)
  - `forecast/` — Numerical forecast backends (AIFS, ECMWF, GFS) — *scaffolding*
- **Expanded JPSS support** — Further split of CLASS / PDS logic (`jpss_class.py`, `jpss_common.py`, `jpss_pds.py`)
- **New ingest stubs** — ASCAT, microwave satellites, NEXRAD, dropsonde, recon, and surface observations prepared for future wiring
- **Note:** The main entry point (`MonWatch-CLI.py`) still drives the full 1.3 feature set. Controller modules are placeholders marked “UNDER CONSTRUCTION”.

### Carried forward from 1.3

- JPSS / VIIRS polar-orbiting support (`--VIIRS-SDR`, `--VIIRS-EDR`, `--JPSS-GRAN`, `--VIIRSI-EDR`, `--jpss`, `--jpss-product`, `--jpss-sat`)
- NOAA-20 / NOAA-21 / S-NPP platform flags (`--NOAA-20`, `--NOAA-21`, `--NOAA`, `--NPP`)
- Smarter `--auto-satellite` (newest available observation wins)
- GarbinWx identity resolution (`GARBINWXID` / `GARBINWX_ID` / `garbinwxid` + optional user-agent)
- Radar type normalisation (`RAIN` / `RAINRATE` → `RR`)

---

## Project Structure

```
MonWatch-CLI/
├── MonWatch-CLI.py          # Main entry point (full 1.3 feature set)
├── args.json                # CLI argument definitions
├── requirements.txt
├── LICENSE.txt / LICENSE
├── .env                     # Credentials (not committed)
├── logo/
│   └── splash.png
├── rem_ingest/              # Remote satellite ingest & processing
│   ├── himawari.py / gk2a.py / goes.py / mtg.py / mtsat.py
│   ├── jpss.py / jpss_class.py / jpss_common.py / jpss_pds.py
│   ├── ascat_sat.py / mw_sats.py
│   ├── common.py / _rgb_corrections.py / ahi_segment_latitudes.py
│   └── __init__.py
├── gro_ingest/              # Ground / radar observations
│   ├── garbinradar.py / phradar.py / NEXRAD.py
│   ├── dropsonde.py / recon.py / surfaceobs.py
│   └── __init__.py
├── controllers/             # Generation controllers (scaffolding)
│   ├── image_gen.py / track_gen.py / forecast_gen.py / all_gen.py
├── styles/                  # Agency rendering styles
│   ├── jma.py / jtwc.py / nhc.py / pagasa.py / mcidas.py / modern.py
├── track_style/             # Track drawing styles
│   ├── jma_style.py / jtwc_style.py / nhc_style.py / pagasa_style.py
├── tracks/                  # Track data ingest
│   ├── jma_ingest.py / jtwc_ingest.py / nhc_ingest.py
│   ├── pagasa_ingest.py / cwa_ingest.py
└── forecast/                # NWP backends (scaffolding)
    ├── aifs.py / ecmwf.py / gfs.py
```

---

## Features

### Satellites

#### Geostationary
- **Himawari-8/9** (AHI) — Full Disk + Target (Region 3)
- **GK-2A** (AMI)
- **GOES-16/17/18/19** (auto East/West selection by longitude)
- **MTG** (EUMETSAT FCI, requires credentials)
- **MTSAT-1R / MTSAT-2** (historical HRIT from CEReS)

#### Polar-Orbiting
- **JPSS / VIIRS** via NOAA CLASS / NESDIS PDS:
  - `--VIIRS-SDR` — VIIRS Sensor Data Records (native radiances)
  - `--VIIRS-EDR` — VIIRS Environmental Data Records
  - `--JPSS-GRAN` — JPSS granule EDRs (VIIRS + ATMS + OMPS)
  - `--VIIRSI-EDR` — VIIRS Imagery EDRs
  - `--jpss <FAMILY>` / `--jpss-product <NAME>` / `--jpss-sat <ID>`
  - Platform shortcuts: `--NOAA-20`, `--NOAA-21`, `--NOAA` (auto), `--NPP`

#### Auto-Selection
- **`--auto-satellite`** — Ranks candidates (Himawari, GK-2A, GOES, MTG) by viewing geometry, probes all, and selects the one with the newest available observation.

### Products
`sandwich`, `true`, `dvorak`, `ir` (BT PWARDS), `infrared`, `z1-ir`, `althea-ott2`, `z1-true`, `z1-dvorak`, `b03`, `irv`, `bt0`, `falsecolor`, `falsecoloradv`, `firetemp`, `dayconv`, `fire`

Batch mode via `--products`.  
Polar-orbiting (VIIRS) composites fall back to the closest available granule time.

### Storm & Region Targeting
- Live ATCF via KnackWx API
- Historical tracks via IBTrACS (`--year` / date-based)
- Basin filtering (`--filter`) — e.g. `WP`, `EPAC`, `AL`, `IO`, `SH`
- Named regions: Philippines, WestPac, CONUS, East/West Coast, Gulf, Caribbean, Europe, Mediterranean, Africa, etc.
- Custom lat/lon, polygon crops (PAR, TCAD, TCID, North/South Luzon, Manila, etc.)
- Peak-intensity time (`--peak`)

### Radar & Ground Data
- **GarbinWx** composite (`--garbinradar`, requires `GARBINWX_ID` / aliases)
- **PAGASA Panahon** national mosaic (`--phradar`)
- Overlay on satellite or standalone floater view
- Additional ingest prepared: NEXRAD, dropsonde, recon, surface observations

### Export
- Formats: AVIF, PNG, JPG, WebP, MP4
- Configurable FPS, width, floater mode (colorbar + edge coords)
- Logo + info overlay, lat/lon grid, coastlines

### Automation
- Date/time or range (`--datefrom` / `--dateto` / `--timefrom` / `--timeto`)
- Multi-worker download + decompress
- Prefetch cache
- Global multi-satellite stacked mode (`--global`)
- Multi-sat run (`--multi`)

### Special Modes
- **BEYEV** (`--beyev`) — Red/cyan anaglyph pseudo-3D from Himawari + GK-2A parallax
- **BEYEVS** (`--beyevs`) — SATAID-style perspective view with tunable camera parameters

---

## Quick Usage

### Geostationary

```bash
# Latest sandwich for all active storms
python MonWatch-CLI.py --product sandwich --him

# Only Western Pacific storms
python MonWatch-CLI.py --product sandwich --him --filter WP

# Auto-pick best satellite (newest observation wins)
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

### Polar-Orbiting (JPSS / VIIRS)

```bash
# VIIRS SDR IR (auto prefers NOAA-21 → NOAA-20 → S-NPP)
python MonWatch-CLI.py --storm 01W --VIIRS-SDR --product infrared

# Pin satellite and product
python MonWatch-CLI.py --storm 01W --jpss VIIRS-SDR \
    --jpss-sat J02 --jpss-product VIIRS-Moderate-Resolution-Band-15-SDR \
    --product infrared

# VIIRS EDR composites
python MonWatch-CLI.py --storm 01W --VIIRS-EDR --products falsecolor,true

# Platform shortcut
python MonWatch-CLI.py --storm 01W --NOAA-21 --product ir

# Auto JPSS selection
python MonWatch-CLI.py --storm 01W --NOAA --product sandwich
```

---

## Environment Variables

MonWatch-CLI reads credentials and API identifiers from environment variables. A `.env` file placed next to the script is loaded automatically at startup.

### Supported Variables

| Variable | Purpose |
|---|---|
| `EUMETSAT_CONSUMER_KEY` | MTG / EUMETSAT API key |
| `EUMETSAT_CONSUMER_SECRET` | MTG / EUMETSAT API secret |
| `GARBINWXID` / `GARBINWX_ID` / `garbinwxid` | GarbinWx radar composite id |
| `GARBINWXUSER` / `GARBINWX_USER` | Optional custom user-agent for GarbinWx |

> **Note:** Any of the GarbinWx aliases is accepted. Precedence: `GARBINWXID` > `GARBINWX_ID` > `garbinwxid`.

### `.env` File Example

```dotenv
# EUMETSAT (MTG)
EUMETSAT_CONSUMER_KEY=your-consumer-key
EUMETSAT_CONSUMER_SECRET=your-consumer-secret

# GarbinWx radar
GARBINWX_ID=your-garbinwx-id
GARBINWXUSER=your-garbinwx-user-agent

# StarTracker
star-trackuser=email/username
star-trackpass=password
```

---

## Acknowledgments

- **NOAA** — Himawari data via AWS, JPSS/VIIRS CLASS & PDS
- **JMA** — Tropical cyclone forecast data
- **NHC** — Atlantic / EPAC storm data
- **JTWC** — Western Pacific cyclone data
- **PAGASA** — Philippine radar (Panahon) and advisories
- **CWA** — Taiwan Central Weather Administration
- **EUMETSAT / OSI SAF** — MTG / FCI data
- **GarbinWx** — Radar composite data (data.garbinwx.org)
- **CEReS / Chiba University** — Historical MTSAT HRIT archive
- **KnackWx** — ATCF tropical cyclone track API

---

> ### Please note that the program is under active development.  
> Controllers, forecast backends, and several new ingest modules are scaffolding only. Core satellite + radar imagery generation (the 1.3 feature set) remains fully operational. Bug reports and feedback are greatly appreciated.

---

> This is a branch of the traditional UI version (MonWatch-UI). The CLI was created to accelerate R&D and automation. Some advanced UI features are still being restored or re-implemented in modular form.

*MonWatch-CLI Server Automation System — PWARDS ECOSYSTEM — © 2025-2026 PWARDS-weather*
