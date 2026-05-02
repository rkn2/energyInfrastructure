"""
Structural limit-state checks for URM wall panels.
All checks are fully vectorized — inputs are numpy arrays from sample_walls().
Returns demand-to-capacity ratio (D/C); failure when D/C >= 1.0.

Limit states:
    1. Out-of-plane flexure (governs: tall/slender walls under wind)
    2. Base sliding       (governs: flood with high horizontal force)
    3. Overturning        (governs: flood on thin/lightweight walls)
"""
import numpy as np


def dc_out_of_plane_flexure(walls: dict,
                             pressure_psf: np.ndarray) -> np.ndarray:
    """
    Simply-supported URM wall under uniform lateral pressure.

    Demand:   M_d = (p · w · h²) / 8           (lbf·ft)
    Capacity: M_c = f_r · S   where
              f_r = modulus of rupture (psi → psf by ×144)
              S   = section modulus (ft³)

    Returns D/C ratio array shape (n_intensity, n_samples).
    """
    # pressure_psf: (nI,); ph_ft = panel height (effective flexural span); w: scalar
    # ph_ft is the floor-to-floor (or floor-to-roof) span — NOT total building height.
    ph = walls["ph_ft"]        # (nS,) effective flexural span
    w  = walls["w_ft"]         # (nS,)
    S  = walls["S_ft3"]        # (nS,)
    f_r_psf = walls["f_r_eff"] * 144.0   # degraded MOR: psi → psf;  (nS,)

    p = pressure_psf[:, np.newaxis]   # (nI, 1)

    M_demand = (p * w) * ph**2 / 8.0  # (nI, nS) lbf·ft — simply supported over panel height
    M_cap = f_r_psf * S                # (nS,) lbf·ft  (broadcast ok)

    return M_demand / np.maximum(M_cap, 1e-6)


def dc_base_sliding(walls: dict,
                    lateral_force_lbf: np.ndarray) -> np.ndarray:
    """
    Sliding along base mortar bed joint.

    Demand:   F_lateral                                          (lbf)
    Capacity: V_c = f_v_eff · A_eff · 144 + W · tan(φ)         (lbf)
              f_v_eff in psi; A_eff in ft² → ×144 to get in²; W in lbf

    Returns D/C ratio array shape (n_intensity, n_samples).
    """
    f_v_eff = walls["f_v_eff"]    # (nS,) psi
    A_eff   = walls["A_eff_ft2"]  # (nS,) ft²
    W       = walls["W_lbf"]      # (nS,) lbf
    phi     = walls["phi"]        # (nS,) radians

    # Cohesion component: f_v [psi] × A [in²] = lbf  →  A_ft2 × 144 = A_in2
    V_cohesion = f_v_eff * (A_eff * 144.0)    # (nS,) lbf
    V_friction = W * np.tan(phi)               # (nS,) lbf
    V_cap = V_cohesion + V_friction            # (nS,) lbf

    F = lateral_force_lbf[:, np.newaxis]       # (nI, 1)
    return F / np.maximum(V_cap, 1e-6)


def dc_overturning(walls: dict,
                   lateral_force_lbf: np.ndarray,
                   moment_arm_ft: np.ndarray) -> np.ndarray:
    """
    Overturning about base (conservative: no uplift resistance from connections).

    Demand:   M_ot = F · arm                           (lbf·ft)
    Capacity: M_stab = W · t/2                         (lbf·ft)

    NOTE: W uses full dry self-weight. For flood loading, partial submersion
    reduces effective weight by buoyancy (γ_w · V_sub). Omitting buoyancy
    overpredicts M_stab by ~8–12% at 10 ft submersion, making results
    slightly non-conservative (understates failure probability for flood).

    Returns D/C ratio array shape (n_intensity, n_samples).
    """
    W   = walls["W_lbf"]     # (nS,) lbf
    t   = walls["t_ft"]      # (nS,) ft

    F    = lateral_force_lbf[:, np.newaxis]    # (nI, 1)
    arm  = moment_arm_ft[:, np.newaxis]        # (nI, 1)

    M_ot   = F * arm                           # (nI, nS) lbf·ft
    M_stab = W * (t / 2.0)                    # (nS,) lbf·ft

    return M_ot / np.maximum(M_stab, 1e-6)


def dc_out_of_plane_flexure_combined(walls: dict,
                                      p_wind_psf: np.ndarray,
                                      F_flood_lbf: np.ndarray,
                                      h_surge_ft: np.ndarray) -> np.ndarray:
    """
    Flexure D/C for simultaneous wind + partial-depth flood on a simply-supported panel.

    Wind acts uniformly over the full panel height ph.
    Flood acts only over the lower h_surge portion — using a concentrated-load
    approximation (F at h_surge/2 from base on SS beam of span ph) rather than
    the incorrect equivalent-uniform-pressure-over-ph approach, which over-
    estimates the flood moment contribution by ~2x when h_surge << ph.

    M_wind  = p_wind · w · ph² / 8           (uniform load)
    M_flood = F_flood · (h_s/2) · (ph − h_s/2) / ph   (conc. load at midpoint of surge)
    M_cap   = f_r_eff · 144 · S
    """
    ph  = walls["ph_ft"]                      # (nS,) sampled panel height
    w   = walls["w_ft"]                       # (nS,)
    S   = walls["S_ft3"]                      # (nS,)
    f_r_psf = walls["f_r_eff"] * 144.0        # (nS,) psi → psf

    p   = p_wind_psf[:, np.newaxis]           # (nI, 1)
    F_f = F_flood_lbf[:, np.newaxis]          # (nI, 1)
    h_s = np.minimum(h_surge_ft[:, np.newaxis], ph)  # surge cannot exceed panel height

    M_wind  = (p * w) * ph**2 / 8.0
    # SS beam concentrated-load moment: F·a·b/L, a=h_s/2 from base, b=ph−a
    a       = h_s / 2.0
    M_flood = F_f * a * (ph - a) / np.maximum(ph, 1e-6)

    M_cap = f_r_psf * S
    return (M_wind + M_flood) / np.maximum(M_cap, 1e-6)


def governing_dc_combined(walls: dict,
                           F_wind_lbf: np.ndarray,
                           p_wind_psf: np.ndarray,
                           F_flood_lbf: np.ndarray,
                           h_surge_ft: np.ndarray,
                           arm_combined_ft: np.ndarray) -> dict:
    """
    D/C ratios for simultaneous wind + storm-surge flood loading.

    Flexure uses the corrected partial-load moment (flood acts only over surge depth).
    Sliding and overturning use the total lateral force F_wind + F_flood.
    """
    F_total = F_wind_lbf + F_flood_lbf

    dc_f = dc_out_of_plane_flexure_combined(walls, p_wind_psf, F_flood_lbf, h_surge_ft)
    dc_s = dc_base_sliding(walls, F_total)
    dc_o = dc_overturning(walls, F_total, arm_combined_ft)

    dc_gov = np.maximum(dc_f, np.maximum(dc_s, dc_o))
    p_fail = (dc_gov >= 1.0).mean(axis=1)

    mean_f = dc_f.mean(axis=1)
    mean_s = dc_s.mean(axis=1)
    mean_o = dc_o.mean(axis=1)
    mode_idx = np.argmax(np.stack([mean_f, mean_s, mean_o], axis=1), axis=1)
    modes = np.array(["flexure", "sliding", "overturning"])[mode_idx]

    return {"dc_flex": dc_f, "dc_slide": dc_s, "dc_ot": dc_o,
            "dc_gov": dc_gov, "p_fail": p_fail, "governing_mode": modes}


def governing_dc(walls: dict,
                 lateral_force_lbf: np.ndarray,
                 pressure_psf: np.ndarray,
                 moment_arm_ft: np.ndarray) -> dict:
    """
    Compute all three D/C ratios and return the governing (max) value.

    Inputs (all numpy arrays):
        lateral_force_lbf : (nI,)  total lateral force per intensity level
        pressure_psf      : (nI,)  equivalent uniform pressure for flexure
        moment_arm_ft     : (nI,)  moment arm for overturning

    Returns dict with keys:
        dc_flex, dc_slide, dc_ot  — shape (nI, nS)
        dc_gov                    — shape (nI, nS) element-wise maximum
        p_fail                    — shape (nI,) fraction of samples with dc_gov ≥ 1
        governing_mode            — shape (nI,) dominant failure mode label per level
    """
    dc_f = dc_out_of_plane_flexure(walls, pressure_psf)
    dc_s = dc_base_sliding(walls, lateral_force_lbf)
    dc_o = dc_overturning(walls, lateral_force_lbf, moment_arm_ft)

    dc_gov = np.maximum(dc_f, np.maximum(dc_s, dc_o))
    p_fail = (dc_gov >= 1.0).mean(axis=1)

    # Dominant mode by sample-averaged D/C across each intensity
    mean_f = dc_f.mean(axis=1)
    mean_s = dc_s.mean(axis=1)
    mean_o = dc_o.mean(axis=1)
    mode_idx = np.argmax(np.stack([mean_f, mean_s, mean_o], axis=1), axis=1)
    modes = np.array(["flexure", "sliding", "overturning"])[mode_idx]

    return {
        "dc_flex": dc_f,
        "dc_slide": dc_s,
        "dc_ot": dc_o,
        "dc_gov": dc_gov,
        "p_fail": p_fail,
        "governing_mode": modes,
    }
