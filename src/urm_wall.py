"""
URM wall panel geometry and material properties for thermal power plant buildings.
Archetypes: boiler house, turbine hall, powerhouse (1890-1950 era construction).
All units: inches/psi for material props; feet/lbf for structural calculations.
"""
import numpy as np
from dataclasses import dataclass, field


# ── Archetype definitions ──────────────────────────────────────────────────────
# Three representative building types from U.S. thermal generation fleet.
# Parameters: mean values for Monte Carlo sampling.

ARCHETYPES = {
    "boiler_house": {
        "label": "Boiler House\n(1895–1915)",
        "thickness_in_mean": 24.0,   # 6-wythe solid brick
        "thickness_in_std": 2.0,
        "height_ft_mean": 30.0,      # total building wall height ~28-32 ft
        "height_ft_std": 4.0,
        # Effective structural span for out-of-plane flexure (floor-to-floor ~12 ft;
        # multi-story building with intermediate floors at ~12 ft).
        "panel_height_ft_mean": 12.0,
        "panel_height_ft_std": 1.5,
        "width_ft": 20.0,            # representative wall panel bay
        "year_built": 1905,
        "f_m_mean_psi": 1100.0,      # brick compressive strength, lime mortar
        "f_m_cov": 0.28,
        "f_v_mean_psi": 38.0,        # bed-joint shear cohesion
        "f_v_cov": 0.45,
        # Modulus of rupture (flexural tensile strength, tension normal to bed joints).
        # TMS 402 Table 9.1.9.2 for clay units, Type N mortar: ~30 psi; degraded for
        # 1905 pure-lime mortar. Mean 25 psi, CoV 0.50.
        "f_r_mean_psi": 25.0,
        "f_r_cov": 0.50,
        "phi_deg": 28.0,             # mortar friction angle
        "unit_weight_pcf": 120.0,
    },
    "turbine_hall": {
        "label": "Turbine Hall\n(1920–1940)",
        "thickness_in_mean": 16.0,   # 4-wythe
        "thickness_in_std": 1.5,
        "height_ft_mean": 45.0,      # total height — open to roof, no intermediate floors
        "height_ft_std": 6.0,
        # No intermediate floor; wall spans between grade slab and roof girder.
        # Vertical pilasters reduce effective free-standing span; use ~20 ft (≈ half-height).
        "panel_height_ft_mean": 20.0,
        "panel_height_ft_std": 2.5,
        "width_ft": 20.0,
        "year_built": 1930,
        "f_m_mean_psi": 1500.0,
        "f_m_cov": 0.25,
        "f_v_mean_psi": 50.0,
        "f_v_cov": 0.40,
        # Type N portland-lime mortar, 1930s: ~45 psi mean (TMS 402).
        "f_r_mean_psi": 45.0,
        "f_r_cov": 0.40,
        "phi_deg": 30.0,
        "unit_weight_pcf": 120.0,
    },
    "powerhouse": {
        "label": "Powerhouse\n(1910–1930)",
        "thickness_in_mean": 20.0,   # 5-wythe
        "thickness_in_std": 2.0,
        "height_ft_mean": 36.0,
        "height_ft_std": 5.0,
        # Ledger beam or partial mezzanine at ~16 ft provides lateral support.
        "panel_height_ft_mean": 16.0,
        "panel_height_ft_std": 2.0,
        "width_ft": 20.0,
        "year_built": 1920,
        "f_m_mean_psi": 1300.0,
        "f_m_cov": 0.27,
        "f_v_mean_psi": 44.0,
        "f_v_cov": 0.42,
        # Mixed lime/portland mortar, ~38 psi mean.
        "f_r_mean_psi": 38.0,
        "f_r_cov": 0.45,
        "phi_deg": 29.0,
        "unit_weight_pcf": 120.0,
    },
}


def degradation_factor(year_built: int, scatter: float = 0.0) -> float:
    """
    Strength reduction from aging: 0.5% per year beyond 50 years of service,
    floored at 0.50. Optional Gaussian scatter (std) for Monte Carlo.
    """
    age = 2026 - year_built
    base = np.clip(1.0 - 0.005 * max(0, age - 50), 0.50, 1.00)
    if scatter > 0.0:
        return float(np.clip(np.random.normal(base, scatter), 0.40, 1.00))
    return float(base)


# ── Vectorised property sampler ────────────────────────────────────────────────

def sample_walls(archetype: dict, n: int, seed: int = 42) -> dict:
    """
    Draw n correlated Monte Carlo samples of wall properties.
    Returns a dict of 1-D numpy arrays, one entry per sample.
    """
    rng = np.random.default_rng(seed)

    # Thickness and heights: truncated normal (physical lower bounds)
    t_in = rng.normal(archetype["thickness_in_mean"],
                      archetype["thickness_in_std"], n).clip(8.0, 36.0)
    h_ft = rng.normal(archetype["height_ft_mean"],
                      archetype["height_ft_std"], n).clip(10.0, 70.0)
    # Panel height = effective flexural span (floor-to-floor or to roof support)
    ph_ft = rng.normal(archetype["panel_height_ft_mean"],
                       archetype["panel_height_ft_std"], n).clip(6.0, 40.0)

    # Material properties: lognormal (always positive, right-skewed)
    f_m = rng.lognormal(np.log(archetype["f_m_mean_psi"]),
                        archetype["f_m_cov"], n)
    f_v = rng.lognormal(np.log(archetype["f_v_mean_psi"]),
                        archetype["f_v_cov"], n)
    # Modulus of rupture: sampled independently (primary control is mortar quality,
    # not brick strength). TMS 402 values for clay units; degraded per archetype era.
    f_r = rng.lognormal(np.log(archetype["f_r_mean_psi"]),
                        archetype["f_r_cov"], n).clip(5.0, 150.0)

    # Degradation: Gaussian scatter around deterministic mean
    deg_mean = degradation_factor(archetype["year_built"])
    deg = rng.normal(deg_mean, 0.05, n).clip(0.40, 1.00)

    t_ft = t_in / 12.0
    w = archetype["width_ft"]
    phi = np.radians(archetype["phi_deg"])

    return {
        "t_in": t_in,
        "t_ft": t_ft,
        "h_ft": h_ft,           # total wall height (used for self-weight, overturning)
        "ph_ft": ph_ft,         # panel height = effective flexural span
        "w_ft": np.full(n, w),
        "f_m": f_m,
        "f_v": f_v,
        "f_r": f_r,             # modulus of rupture, psi
        "deg": deg,
        "phi": np.full(n, phi),
        "unit_wt": np.full(n, archetype["unit_weight_pcf"]),
        # Derived quantities used repeatedly in limit-state checks
        "f_m_eff": f_m * deg,
        "f_v_eff": f_v * deg,
        "f_r_eff": f_r * deg,   # degraded modulus of rupture
        # Section modulus per panel (ft³): S = w * t² / 6
        "S_ft3": w * t_ft**2 / 6.0,
        # Self-weight (lbf): γ * t * h_total * w  (total wall above base)
        "W_lbf": archetype["unit_weight_pcf"] * t_ft * h_ft * w,
        # Physical base area (ft²) — degradation is captured in f_v_eff, NOT here.
        # Applying deg to area would double-count it in the cohesion sliding term.
        "A_eff_ft2": t_ft * w,
    }
