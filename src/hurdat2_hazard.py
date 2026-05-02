"""
HURDAT2-based empirical wind hazard estimation for the Plant Daniel site.

Downloads the NOAA Atlantic hurricane track database and builds an empirical
annual exceedance probability (AEP) curve by applying a parametric wind field
model (modified Rankine vortex) to each historical storm track 1851–2023.

Compares the resulting empirical hazard to the ASCE 7-22 site-specific table
to validate the design wind speed assumptions used in the fragility analysis.

Reference:
  Landsea, C. W. & Franklin, J. L. (2013). Atlantic hurricane database uncertainty
  and presentation of a new database format. Monthly Weather Review, 141(10), 3576–3592.

Site: Plant Daniel vicinity, Escatawpa MS (~30.40°N, 88.47°W, Jackson County).
"""
import numpy as np
import os
import urllib.request

HURDAT2_URL   = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
HURDAT2_CACHE = os.path.join(os.path.dirname(__file__), "..", "outputs", "hurdat2_cache.txt")

# Site (more precise than prior 30.35°N/88.87°W which was Harrison County)
SITE_LAT = 30.40
SITE_LON = -88.47

# Wind field model parameters
R_MAX_KM       = 50.0    # radius of max winds (km), Gulf Coast median
R_DECAY_KM     = 80.0    # e-folding decay length beyond R_max (km)
SEARCH_RADIUS_KM = 350.0 # include storms with any point within this radius

# Wind unit conversion: HURDAT2 uses 1-min sustained knots; model needs 3-s gust mph
# 1 kt = 1.15078 mph; gust factor ≈ 1.28 for open coastal terrain (ASCE 7-22 Sec. 26.5)
KT_TO_MPH_GUST = 1.15078 * 1.28  # ≈ 1.473

# Only consider observations with hurricane-force winds (≥64 kt)
MIN_WIND_KT  = 64
YEARS_RECORD = 2023 - 1851 + 1   # 173 years


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _wind_at_site_kt(v_max_kt, dist_km):
    """Modified Rankine wind profile: flat inside R_max, exponential decay outside."""
    if dist_km <= R_MAX_KM:
        return float(v_max_kt)
    return float(v_max_kt * np.exp(-(dist_km - R_MAX_KM) / R_DECAY_KM))


def _download_hurdat2():
    os.makedirs(os.path.dirname(HURDAT2_CACHE), exist_ok=True)
    if not os.path.exists(HURDAT2_CACHE):
        print("  Downloading HURDAT2 from NOAA NHC (~3 MB)...", flush=True)
        urllib.request.urlretrieve(HURDAT2_URL, HURDAT2_CACHE)
        print(f"  Cached at {HURDAT2_CACHE}", flush=True)
    return HURDAT2_CACHE


def _parse_hurdat2(path):
    """
    Returns list of storms: each {'id', 'name', 'obs'}.
    obs entries: (year, lat, lon, wind_kt).
    """
    storms, current = [], None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0][:2] in ("AL", "EP", "CP"):
                if current:
                    storms.append(current)
                current = {"id": parts[0], "name": parts[1], "obs": []}
            elif current is not None and len(parts) >= 7:
                try:
                    date = parts[0]
                    year = int(date[:4])
                    lat_s = parts[4]
                    lon_s = parts[5].strip()
                    lat = float(lat_s.replace("N", "").replace("S", "")) * (-1 if "S" in lat_s else 1)
                    lon = float(lon_s.replace("W", "").replace("E", "")) * (-1 if "W" in lon_s else 1)
                    wind_kt = int(parts[6])
                    if wind_kt >= MIN_WIND_KT:
                        current["obs"].append((year, lat, lon, wind_kt))
                except (ValueError, IndexError):
                    pass
    if current:
        storms.append(current)
    return storms


def build_hurdat2_hazard(
    site_lat: float = SITE_LAT,
    site_lon: float = SITE_LON,
    search_km: float = SEARCH_RADIUS_KM,
) -> dict:
    """
    Build empirical annual exceedance probability curve from HURDAT2 record.

    Returns
    -------
    dict with keys:
        v_mph           : ndarray of wind speed thresholds (mph, 3-s gust)
        aep             : annual exceedance probability at each threshold
        return_period   : 1/aep (years)
        storm_events    : list of (year, v_mph_at_site, dist_km, name)
        n_storms        : int, storms within search_km
        years           : float, record length
        rp_table        : dict {RP_yr: V_mph} at standard return periods
        asce_comparison : dict {RP_yr: {'hurdat2': V, 'asce': V, 'ratio': r}}
    """
    from site_specific import SITE_WIND_HAZARD  # compare against ASCE table

    path = _download_hurdat2()
    storms = _parse_hurdat2(path)

    events = []
    for s in storms:
        best_v, best_dist, best_year = 0.0, np.inf, None
        for (year, lat, lon, wind_kt) in s["obs"]:
            dist = _haversine(site_lat, site_lon, lat, lon)
            if dist < search_km:
                v_site = _wind_at_site_kt(wind_kt, dist)
                if v_site > best_v:
                    best_v, best_dist, best_year = v_site, dist, year
        if best_v > 0:
            events.append((best_year, best_v * KT_TO_MPH_GUST, best_dist, s["name"]))

    if not events:
        return {}

    v_mph_arr = np.array([e[1] for e in events])
    v_grid    = np.linspace(40, 220, 200)
    aep       = np.array([np.sum(v_mph_arr >= v) / YEARS_RECORD for v in v_grid])
    rp_grid   = np.where(aep > 0, 1.0 / aep, np.inf)

    rp_table = {}
    for rp in sorted(SITE_WIND_HAZARD.keys()):
        target = 1.0 / rp
        idx = np.searchsorted(-aep, -target)
        if 0 < idx < len(v_grid):
            # Linear interpolation
            f = (target - aep[idx]) / (aep[idx - 1] - aep[idx] + 1e-12)
            rp_table[rp] = float(v_grid[idx] + f * (v_grid[idx - 1] - v_grid[idx]))
        elif idx == 0:
            rp_table[rp] = float(v_grid[0])
        else:
            rp_table[rp] = float(v_grid[-1])

    asce_comp = {
        rp: {
            "hurdat2": rp_table.get(rp, np.nan),
            "asce":    SITE_WIND_HAZARD[rp],
            "ratio":   rp_table.get(rp, np.nan) / SITE_WIND_HAZARD[rp] if rp in rp_table else np.nan,
        }
        for rp in SITE_WIND_HAZARD
    }

    return {
        "v_mph":         v_grid,
        "aep":           aep,
        "return_period": rp_grid,
        "storm_events":  events,
        "n_storms":      len(events),
        "years":         float(YEARS_RECORD),
        "rp_table":      rp_table,
        "asce_comparison": asce_comp,
    }


def run_hurdat2_analysis() -> dict:
    """Top-level entry point called from run_analysis.py."""
    print("  Building HURDAT2 empirical hazard...", flush=True)
    result = build_hurdat2_hazard()
    n = result.get("n_storms", 0)
    print(f"  {n} hurricane events within {SEARCH_RADIUS_KM:.0f} km "
          f"of site in {int(YEARS_RECORD)}-yr HURDAT2 record.", flush=True)
    if result.get("rp_table"):
        print(f"  HURDAT2 700-yr wind: {result['rp_table'].get(700, 'n/a'):.0f} mph  "
              f"|  ASCE 7-22 site: 150 mph", flush=True)
    return result
