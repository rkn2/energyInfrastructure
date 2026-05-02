"""
Hazard demand calculations: hurricane wind, tornado wind, coastal/riverine flood.
References: ASCE 7-22 Chapters 26-27 (wind), ASCE 7-22 Chapter 5 / FEMA P-55 (flood).
All forces in lbf, pressures in psf (lbf/ft²), lengths in ft.
"""
import numpy as np


# ── Wind ──────────────────────────────────────────────────────────────────────

def velocity_pressure_psf(V_mph: float,
                           Kz: float = 0.98,   # Exposure C, z≈30 ft; use kz_exposure_c() for height-specific
                           Kzt: float = 1.0,
                           Kd: float = 1.0) -> float:    # C&C wall panels: Kd=1.0 per ASCE 7-22 §26.6.1 Table 26.6-1
    """
    Velocity pressure per ASCE 7-22 Eq. 26.10-1:
        qz = 0.00256 · Kz · Kzt · Kd · V²   (psf)
    Kd = 1.0 for Components & Cladding (wall panels); MWFRS uses 0.85.
    """
    return 0.00256 * Kz * Kzt * Kd * V_mph**2


def kz_exposure_c(z_ft: float) -> float:
    """
    Velocity pressure exposure coefficient Kz for Exposure Category C.
    ASCE 7-22 Table 26.10-1 / Eq. 26.10-1:
        Kz = 2.01 · (z/zg)^(2/α)   where α = 9.5, zg = 900 ft
    Clipped to z ≥ 15 ft (code minimum reference height).
    """
    z = max(float(z_ft), 15.0)
    return 2.01 * (z / 900.0) ** (2.0 / 9.5)


def wind_pressure_psf(V_mph: float | np.ndarray,
                       Cp_windward: float = 0.8,
                       Cp_leeward: float = 0.5,
                       Kz: float = 0.98) -> float | np.ndarray:
    """
    Net out-of-plane wall pressure (windward + leeward contribution).
    p_net = qz · (Cp_windward + Cp_leeward)  — conservative for enclosed industrial bldg.
    """
    qz = velocity_pressure_psf(V_mph, Kz=Kz)
    return qz * (Cp_windward + Cp_leeward)


# EF-scale peak 3-second gust wind speed ranges (mph), ASCE 7-22 App. CC Table CC-1
EF_WIND_SPEED = {
    0: (65,  85),
    1: (86, 110),
    2: (111, 135),
    3: (136, 165),
    4: (166, 200),
    5: (201, 250),
}

EF_LABELS = ["EF0", "EF1", "EF2", "EF3", "EF4", "EF5"]

EF_MID_SPEEDS = np.array([
    (lo + hi) / 2.0 for lo, hi in EF_WIND_SPEED.values()
])


def tornado_pressure_psf(EF_speed_mph: float | np.ndarray,
                          Kz: float = 0.98) -> float | np.ndarray:
    """
    Apply an internal pressure amplification factor of 1.5× for tornado
    (ASCE 7-22 App. CC §CC.5: breaching of envelope creates +/- internal pressure).
    """
    return wind_pressure_psf(EF_speed_mph, Kz=Kz) * 1.5


# ── Flood ─────────────────────────────────────────────────────────────────────

WATER_DENSITY_PCF = 62.4      # freshwater; most thermal plants are riverine, not coastal
FLOOD_Cd = 2.0                # drag coefficient for flat wall, FEMA P-55 §3.4.2


def hydrostatic_resultant(depth_ft: float | np.ndarray,
                           width_ft: float = 20.0) -> tuple:
    """
    Total hydrostatic force on a rectangular wall and its moment arm above base.
    F = 0.5 · ρ · h² · w       (lbf)
    arm = h / 3                 (ft above base — triangular pressure dist.)
    """
    F = 0.5 * WATER_DENSITY_PCF * depth_ft**2 * width_ft
    arm = depth_ft / 3.0
    return F, arm


def hydrodynamic_force(depth_ft: float | np.ndarray,
                        V_water_fps: float = 6.0,
                        width_ft: float = 20.0) -> float | np.ndarray:
    """
    Hydrodynamic drag per FEMA P-55 Eq. 3-4:
        F_dyn = 0.5 · (ρ/g) · Cd · V² · A
    Assumes V_water_fps ≈ flood velocity at impact surface.
    A = depth × width (submerged area of wall).
    """
    rho_slug = WATER_DENSITY_PCF / 32.2   # slugs/ft³
    A = depth_ft * width_ft               # ft²
    return 0.5 * rho_slug * FLOOD_Cd * V_water_fps**2 * A


def flood_total_force(depth_ft: float | np.ndarray,
                       V_water_fps: float = 6.0,
                       width_ft: float = 20.0) -> tuple:
    """
    Combined hydrostatic + hydrodynamic flood force and effective moment arm.
    Returns (F_total_lbf, effective_arm_ft).
    """
    F_hs, arm_hs = hydrostatic_resultant(depth_ft, width_ft)
    F_hd = hydrodynamic_force(depth_ft, V_water_fps, width_ft)
    # Hydrostatic arm = h/3; hydrodynamic acts at centroid ≈ h/2
    arm_hd = depth_ft / 2.0
    F_total = F_hs + F_hd
    # Weighted combined moment arm
    arm_total = np.where(F_total > 0,
                         (F_hs * arm_hs + F_hd * arm_hd) / F_total,
                         depth_ft / 3.0)
    # Effective uniform pressure for flexure check (average over submerged height)
    p_eff_psf = np.where(depth_ft > 0, F_total / (depth_ft * width_ft), 0.0)
    return F_total, arm_total, p_eff_psf


# ── Combined hurricane landfall (wind + flood simultaneously) ─────────────────

def hurricane_landfall_forces(V_wind_mph: float | np.ndarray,
                               storm_surge_ft: float | np.ndarray,
                               V_water_fps: float = 8.0,
                               width_ft: float = 20.0,
                               Kz: float = 0.98) -> dict:
    """
    Simultaneous wind and storm-surge flood loading during hurricane landfall.
    Returns a dict of load components.
    """
    p_wind = wind_pressure_psf(V_wind_mph, Kz=Kz)
    F_flood, arm_flood, p_flood_eff = flood_total_force(storm_surge_ft, V_water_fps, width_ft)

    return {
        "p_wind_psf": p_wind,
        "F_flood_lbf": F_flood,
        "arm_flood_ft": arm_flood,
        "p_flood_eff_psf": p_flood_eff,
        "surge_ft": storm_surge_ft,
    }


# ── Return-period wind speed table (approximation for ASCE 7-22 Risk Cat II) ──
# Representative Gulf/Atlantic Coast exposure; used for annual risk calculation.
# Short-return-period entries (1–5 yr) are approximate; they anchor the lower
# tail of the integral and prevent truncation errors for archetypes that start
# failing below the 10-yr wind speed.  Values from NWS climatology for coastal
# SE US; treat as order-of-magnitude estimates.
RETURN_PERIOD_WIND = {
    1:    55,    # mph, 3-s gust (approximate calm-year wind)
    2:    65,
    5:    76,
    10:   85,
    25:   100,
    50:   115,
    100:  130,
    250:  150,
    500:  165,
    700:  175,
    1700: 190,
}

# Representative coastal/riverine flood depths by return period (FEMA Zone AE typical)
RETURN_PERIOD_FLOOD = {
    10:   3.0,   # ft of inundation above grade
    25:   5.0,
    50:   7.5,
    100:  10.0,  # 100-yr FEMA base flood elevation
    250:  13.0,
    500:  15.0,
}
