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

    Returns D/C ratio array shape (n_intensity, n_samples).
    """
    W   = walls["W_lbf"]     # (nS,) lbf
    t   = walls["t_ft"]      # (nS,) ft

    F    = lateral_force_lbf[:, np.newaxis]    # (nI, 1)
    arm  = moment_arm_ft[:, np.newaxis]        # (nI, 1)

    M_ot   = F * arm                           # (nI, nS) lbf·ft
    M_stab = W * (t / 2.0)                    # (nS,) lbf·ft

    return M_ot / np.maximum(M_stab, 1e-6)


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
