# =============================================================================
# rem_ingest/_rgb_corrections.py — shared true-color atmospheric correction
# =============================================================================
import numpy as np
from pyorbital.astronomy import sun_zenith_angle


def apply_rgb_corrections(r, g, b, ir, target_area, target_dt, mode=1,
                          saturation_factor=1.5, gamma_cor=0.88):
    r_norm = np.clip(r / 100.0 if np.nanmax(r) > 1.0 else r, 0.0, 1.0)
    g_norm = np.clip(g / 100.0 if np.nanmax(g) > 1.0 else g, 0.0, 1.0)
    b_norm = np.clip(b / 100.0 if np.nanmax(b) > 1.0 else b, 0.0, 1.0)

    if target_area is not None:
        lons, lats = target_area.get_lonlats()
        sza = sun_zenith_angle(target_dt, lons, lats)
    else:
        sza = np.zeros_like(r_norm, dtype=np.float32)
        
    cos_sza = np.clip(np.cos(np.radians(sza)), 0.33, 1.0)
    cos2_sza = np.clip(np.cos(np.radians(sza)), 0.38, 1.0)

    path_sun = 1.0 / cos2_sza
    path_sun_a = 1.0 / cos_sza

    r_bright = r_norm * path_sun_a * 0.9 + 0.01
    g_bright = g_norm * path_sun_a * 0.9 + 0.01
    b_bright = b_norm * path_sun_a * 0.9 + 0.01

    if mode == 1:
        rayleigh_r = 0.011 * path_sun + 0.001
        rayleigh_g = 0.031 * path_sun + 0.001
        rayleigh_b = 0.051 * path_sun + 0.002
    else:
        rayleigh_r = 0.011 * path_sun + 0.001
        rayleigh_g = 0.031 * path_sun + 0.004
        rayleigh_b = 0.051 * path_sun + 0.005

    r_corr = np.clip(r_bright - rayleigh_r, 0.0, 1.0)
    g_corr = np.clip(g_bright - rayleigh_g, 0.0, 1.0)
    b_corr = np.clip(b_bright - rayleigh_b, 0.0, 1.0)

    day_weight = np.clip((91.0 - sza) / 5.0, 0.0, 1.0)
    night_weight = 1.0 - day_weight

    r_final_vis = r_corr * day_weight
    g_final_vis = g_corr * day_weight
    b_final_vis = b_corr * day_weight

    if gamma_cor != 1.0:
        r_final_vis = np.clip(np.power(r_final_vis, gamma_cor), 0.0, 1.0)
        g_final_vis = np.clip(np.power(g_final_vis, gamma_cor), 0.0, 1.0)
        b_final_vis = np.clip(np.power(b_final_vis, gamma_cor), 0.0, 1.0)

    ir_norm = np.clip((313.15 - ir) / (313.15 - 173.15), 0.0, 1.0)
    ir_layer = np.power(ir_norm, 1.5) * 0.66

    r_final = r_final_vis + ir_layer * night_weight
    g_final = g_final_vis + ir_layer * night_weight
    b_final = b_final_vis + ir_layer * night_weight

    if saturation_factor != 1.0:
        luminance = 0.2989 * r_final + 0.5870 * g_final + 0.1140 * b_final
        r_final = np.clip(luminance + saturation_factor * (r_final - luminance), 0.0, 1.0)
        g_final = np.clip(luminance + saturation_factor * (g_final - luminance), 0.0, 1.0)
        b_final = np.clip(luminance + saturation_factor * (b_final - luminance), 0.0, 1.0)

    return r_final, g_final, b_final