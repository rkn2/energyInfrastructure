"""
Fragility curve generation for URM thermal plant building wall assemblies.
Uses vectorized Monte Carlo with independent parameter sampling.

Hazards:
    wind_hurricane  — 3-s gust V (mph), 60–200 mph
    wind_tornado    — EF-scale peak gust mapped through EF_MID_SPEEDS
    flood           — inundation depth h (ft), 0–16 ft

Outputs: P(failure | intensity) for each archetype and hazard.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from urm_wall import ARCHETYPES, sample_walls
from hazard_loads import (
    wind_pressure_psf, tornado_pressure_psf, flood_total_force,
    hurricane_landfall_forces, EF_MID_SPEEDS,
    RETURN_PERIOD_WIND, RETURN_PERIOD_FLOOD,
)
from limit_states import governing_dc

N_SAMPLES = 12_000   # per archetype; balance between accuracy and speed

# ── Intensity ranges ───────────────────────────────────────────────────────────
V_HURRICANE = np.linspace(60, 200, 50)    # mph, 3-s gust
V_TORNADO   = EF_MID_SPEEDS              # representative mid-range per EF category
DEPTH_FLOOD = np.linspace(0.5, 16.0, 50) # ft

# Storm-surge depth correlated with hurricane wind speed (simplified linear fit)
# Category 1 (~80 mph) → ~3 ft surge;  Category 5 (~180 mph) → ~20 ft surge
def surge_from_wind(V_mph: np.ndarray) -> np.ndarray:
    """
    Illustrative linear fit: Cat 1 (~80 mph) → ~3 ft, Cat 5 (~180 mph) → ~17 ft.
    Based on median NOAA SLOSH model outputs for Gulf Coast landfalls; see
    Irish et al. (2008) "The Influence of Storm Size on Hurricane Surge."
    Use only for relative compound-hazard demonstration, not site-specific design.
    """
    return np.clip((V_mph - 60.0) * 0.14, 0.5, 20.0)


def _run_hazard(walls: dict,
                lateral_force: np.ndarray,
                pressure: np.ndarray,
                arm: np.ndarray) -> np.ndarray:
    """Thin wrapper: compute failure probability vector for one hazard."""
    result = governing_dc(walls, lateral_force, pressure, arm)
    return result["p_fail"], result["governing_mode"]


def hurricane_fragility(archetype: dict) -> dict:
    """
    P(wall failure | V_hurricane) for a single archetype.
    Force computed over full panel height; arm at mid-panel.
    """
    walls = sample_walls(archetype, N_SAMPLES, seed=1)
    ph = archetype["panel_height_ft_mean"]   # effective structural span
    p_net = wind_pressure_psf(V_HURRICANE)   # (nV,) psf
    F   = p_net * ph * archetype["width_ft"] # (nV,) lbf — over panel span
    arm = np.full(len(V_HURRICANE), ph / 2.0)

    p_fail, modes = _run_hazard(walls, F, p_net, arm)
    return {"V_mph": V_HURRICANE, "p_fail": p_fail, "modes": modes}


def tornado_fragility(archetype: dict) -> dict:
    """
    P(wall failure | EF category) for a single archetype.
    Tornado pressure includes 1.5× internal pressure amplification (ASCE 7-22 App. CC).
    """
    walls = sample_walls(archetype, N_SAMPLES, seed=2)
    ph = archetype["panel_height_ft_mean"]
    p_net = tornado_pressure_psf(V_TORNADO)  # (6,) psf
    F   = p_net * ph * archetype["width_ft"]
    arm = np.full(len(V_TORNADO), ph / 2.0)

    p_fail, modes = _run_hazard(walls, F, p_net, arm)
    return {"EF_speed_mph": V_TORNADO, "p_fail": p_fail, "modes": modes}


def flood_fragility(archetype: dict) -> dict:
    """
    P(wall failure | flood depth in ft) for a single archetype.
    Overturning uses total wall height; sliding uses degraded base area.
    Flood force and arm computed from hydrostatic + hydrodynamic resultant.
    """
    walls = sample_walls(archetype, N_SAMPLES, seed=3)
    F, arm, p_eff = flood_total_force(DEPTH_FLOOD,
                                       V_water_fps=6.0,
                                       width_ft=archetype["width_ft"])
    p_fail, modes = _run_hazard(walls, F, p_eff, arm)
    return {"depth_ft": DEPTH_FLOOD, "p_fail": p_fail, "modes": modes}


def combined_hurricane_fragility(archetype: dict) -> dict:
    """
    P(wall failure | V_hurricane) under simultaneous storm surge + wind (landfall).
    Surge depth correlated with wind speed via surge_from_wind().
    """
    walls = sample_walls(archetype, N_SAMPLES, seed=4)
    surge = surge_from_wind(V_HURRICANE)
    loads = hurricane_landfall_forces(V_HURRICANE, surge,
                                       width_ft=archetype["width_ft"])

    ph = archetype["panel_height_ft_mean"]
    F_wind  = loads["p_wind_psf"] * ph * archetype["width_ft"]
    F_flood = loads["F_flood_lbf"]
    F_total = F_wind + F_flood

    # Effective uniform pressure for flexure check (over panel height)
    p_combined = F_total / (ph * archetype["width_ft"])

    # Weighted moment arm for overturning
    arm_wind  = ph / 2.0
    arm_combined = np.where(
        F_total > 0,
        (F_wind * arm_wind + F_flood * loads["arm_flood_ft"]) / F_total,
        arm_wind,
    )

    p_fail, modes = _run_hazard(walls, F_total, p_combined, arm_combined)
    return {"V_mph": V_HURRICANE, "surge_ft": surge,
            "p_fail": p_fail, "modes": modes}


def run_all_fragility() -> dict:
    """
    Compute hurricane, tornado, flood, and combined fragility for all archetypes.
    Returns nested dict: results[archetype_key][hazard] = fragility_dict.
    """
    results = {}
    for key, arch in ARCHETYPES.items():
        print(f"  Computing fragility: {key} ...", flush=True)
        results[key] = {
            "hurricane":          hurricane_fragility(arch),
            "tornado":            tornado_fragility(arch),
            "flood":              flood_fragility(arch),
            "hurricane_combined": combined_hurricane_fragility(arch),
        }
    return results


# ── Annual failure probability ─────────────────────────────────────────────────

def annual_failure_probability(fragility_interp, hazard_table: dict) -> float:
    """
    Annual failure probability via PEER PBEE numerical integration:
        λ_f = Σᵢ P(fail | IMᵢ) × Δλᵢ
    where Δλᵢ = λ(IMᵢ₋₁) - λ(IMᵢ) is the annual rate of occurrence in bin i.
    fragility_interp: callable P(failure | intensity)
    hazard_table: {return_period: intensity}  (sorted ascending return period → decreasing rate)
    """
    rp = np.array(sorted(hazard_table.keys()), dtype=float)  # ascending RP → descending rate
    intensities = np.array([hazard_table[r] for r in rp])
    lambdas = 1.0 / rp                        # annual exceedance rates (descending)

    pf = fragility_interp(intensities)

    # Bin rates: Δλᵢ = λ(RPᵢ) - λ(RPᵢ₊₁) = 1/RPᵢ - 1/RPᵢ₊₁  (positive)
    # Each bin contributes P_fail(midpoint) × probability mass in that bin.
    d_lambda = lambdas[:-1] - lambdas[1:]     # positive since lambdas is descending
    pf_mid   = (pf[:-1] + pf[1:]) / 2.0

    return float(np.sum(pf_mid * d_lambda))
