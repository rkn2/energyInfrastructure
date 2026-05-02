"""
Site-specific hazard analysis: illustrative Gulf Coast industrial site.
Reference location: Plant Daniel vicinity, Escatawpa MS (~30.40°N, 88.47°W),
Jackson County — used here as a well-documented Gulf Coast industrial site with
publicly available ASCE 7-22 wind and FEMA FIRM flood data.

For the AMMTO HPC4EI concept paper this is a demonstration/validation site.
Replace with the SMM's actual facility coordinates for the full application.

NOTE FOR STEAM PLANT VARIANT (DOE CESER / GMI): restore SITE_NAME to
"Victor J. Daniel Jr. Power Plant vicinity" and frame as the specific facility
under study rather than an illustrative example.

Replaces generic ASCE 7-22 return-period tables with site-specific values
and compares resulting AFP against the generic-table baseline.

Key site characteristics:
  - Gulf Coast, open terrain (Exposure C)
  - FEMA Zone AE (high-risk coastal flood zone), BFE ~12 ft NAVD88
  - High hurricane wind hazard: ASCE 7-22 700-yr wind ~150 mph
  - NOAA SLOSH max surge for Cat 5 at this location: ~25 ft
    (surge_from_wind(157) ≈ 13.6 ft — documented underestimate; not corrected
     here for consistency; see SURGE_BIAS_FACTOR below)
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from urm_wall import ARCHETYPES
from fragility import (
    annual_failure_probability,
    hurricane_fragility, flood_fragility, combined_hurricane_fragility,
    RETURN_PERIOD_WIND, RETURN_PERIOD_FLOOD,
)
from scipy.interpolate import interp1d

# ── Site identity ──────────────────────────────────────────────────────────────
SITE_NAME = "Victor J. Daniel Jr. Power Plant vicinity"
SITE_CITY = "Escatawpa, MS"   # Jackson County; corrected from prior Gulfport/Harrison County
SITE_LAT  = 30.35
SITE_LON  = -88.87

# ── Site-specific wind hazard (mph, 3-s gust, Risk Cat II) ───────────────────
# Source: ASCE 7-22 Figure 26.5-1B, Exposure C, coastal Harrison County MS.
# Representative values; use ASCE 7 Hazard Tool for precise site coordinates.
SITE_WIND_HAZARD = {
     10:  82,
     25:  96,
     50: 108,
    100: 118,
    700: 150,
   1700: 165,
}

# ── Site-specific flood hazard (ft inundation above grade) ───────────────────
# Source: FEMA FIRM for Harrison County MS (Panel 28047C).
# Zone AE, BFE approximately 12 ft NAVD88; site grade assumed at ~0 ft NAVD88.
SITE_FLOOD_HAZARD = {
     10:  3.0,
     25:  5.5,
     50:  8.0,
    100: 12.0,
    250: 14.5,
    500: 16.5,
}

# ── Storm surge note ──────────────────────────────────────────────────────────
# NOAA SLOSH maximum envelope (MEOW) for Gulf of Mexico basin, Cat 5:
# Mississippi Sound coast typically sees 20-25 ft (Category 5, direct hit).
# The surge_from_wind() function uses the Gulf Coast median (~13.6 ft at 157 mph).
# Documenting this discrepancy — ratio is ~1.84 for Cat 5 at this site.
SLOSH_MAX_SURGE_CAT5_FT   = 25.0
SURGE_FROM_WIND_CAT5_FT   = 0.14 * (157.0 - 60.0)   # ≈ 13.6 ft
SURGE_BIAS_FACTOR          = SLOSH_MAX_SURGE_CAT5_FT / SURGE_FROM_WIND_CAT5_FT


def _afp_from_fragility(frag_dict: dict, hazard_key: str, hazard_table: dict) -> float:
    """Helper: integrate a fragility curve over a given hazard table."""
    interp_fn = interp1d(
        frag_dict[hazard_key], frag_dict["p_fail"],
        bounds_error=False, fill_value=(0.0, 1.0),
    )
    return float(annual_failure_probability(interp_fn, hazard_table))


def compute_site_afp() -> dict:
    """
    For each archetype and hazard, compute both generic and site-specific AFP.
    The fragility curves are identical (site-independent); only the hazard
    integration tables differ.

    Returns
    -------
    dict with structure:
        {archetype_key: {hazard: {"generic_afp": float, "site_afp": float}}}
    """
    results = {}
    for key, arch in ARCHETYPES.items():
        print(f"    Site AFP: {key} ...", flush=True)

        # Fragility curves (same for generic and site — only AFP integral differs)
        frag_h = hurricane_fragility(arch)
        frag_f = flood_fragility(arch)
        frag_c = combined_hurricane_fragility(arch)

        results[key] = {
            "hurricane": {
                "generic_afp": _afp_from_fragility(frag_h, "V_mph",   RETURN_PERIOD_WIND),
                "site_afp":    _afp_from_fragility(frag_h, "V_mph",   SITE_WIND_HAZARD),
            },
            "flood": {
                "generic_afp": _afp_from_fragility(frag_f, "depth_ft", RETURN_PERIOD_FLOOD),
                "site_afp":    _afp_from_fragility(frag_f, "depth_ft", SITE_FLOOD_HAZARD),
            },
            "combined": {
                "generic_afp": _afp_from_fragility(frag_c, "V_mph",   RETURN_PERIOD_WIND),
                "site_afp":    _afp_from_fragility(frag_c, "V_mph",   SITE_WIND_HAZARD),
            },
        }
    return results


def run_site_analysis() -> dict:
    """Top-level entry point called from run_analysis.py."""
    afp = compute_site_afp()
    return {
        "site_name":        SITE_NAME,
        "city":             SITE_CITY,
        "lat":              SITE_LAT,
        "lon":              SITE_LON,
        "wind_hazard":      SITE_WIND_HAZARD,
        "flood_hazard":     SITE_FLOOD_HAZARD,
        "surge_bias_factor": SURGE_BIAS_FACTOR,
        "afp":              afp,
    }
