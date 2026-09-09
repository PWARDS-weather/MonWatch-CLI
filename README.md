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
    src="https://github.com/user-attachments/assets/9952b772-2285-4eff-8ea9-43ff862789af"
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

**Release date:** September 9, 2026

---

## Features

### Satellites
- **Himawari-8/9** (AHI) — Full Disk + Target (Region 3)
- **GK-2A** (AMI)
- **GOES-16/17/18/19** (auto East/West selection by longitude)
- **MTG** (EUMETSAT FCI, requires credentials)
- **MTSAT-1R / MTSAT-2** (historical HRIT from CEReS)

### Products
`sandwich`, `true`, `dvorak`, `ir` (BT PWARDS), `infrared`, `z1-ir`, `althea-ott2`, `z1-true`, `z1-dvorak`, `b03`, `irv`, `bt0`, `falsecolor`, `falsecoloradv`, `firetemp`, `dayconv`, `fire`

Batch mode via `--products`.

### Storm & Region Targeting
- Live ATCF via KnackWx API
- Historical tracks via IBTrACS (`--year` / date-based)
- Named regions: Philippines, WestPac, CONUS, East/West Coast, Gulf, Caribbean, Europe, Mediterranean, Africa, etc.
- Custom lat/lon, polygon crops (PAR, TCAD, TCID, North/South Luzon, Manila, etc.)
- Peak-intensity time (`--peak`)

### Radar
- **GarbinWx** composite (`--garbinradar`, requires `GARBINWX_ID`)
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

```bash
# Latest sandwich for active WPAC storms
python MonWatch-CLI.py --product sandwich --him

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
