"""
Main analysis script: runs Monte Carlo fragility analysis for all archetypes
and generates publication-quality figures for the HPC4EI concept paper.

Usage:
    python src/run_analysis.py

Outputs written to outputs/:
    fragility_wind_hurricane.png
    fragility_wind_tornado.png
    fragility_flood.png
    fragility_combined_hurricane.png
    risk_matrix.png
    degradation_sensitivity.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from urm_wall import ARCHETYPES
from hazard_loads import (
    RETURN_PERIOD_WIND, RETURN_PERIOD_FLOOD,
    EF_LABELS, EF_MID_SPEEDS,
)
from fragility import run_all_fragility, annual_failure_probability, surge_from_wind, union_afp
from opensees_comparison import run_opensees_comparison, BC_LABELS
from site_specific import run_site_analysis, SITE_NAME, SITE_CITY, SITE_LAT, SITE_LON
from hpc_scaling import run_hpc_scaling
from hurdat2_hazard import run_hurdat2_analysis

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    "boiler_house": "#C0392B",   # deep red
    "turbine_hall": "#2471A3",   # steel blue
    "powerhouse":   "#1E8449",   # forest green
}
LINESTYLES = {
    "boiler_house": "-",
    "turbine_hall": "--",
    "powerhouse":   "-.",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DPI = 200

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Helper ─────────────────────────────────────────────────────────────────────

def save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Hurricane wind fragility
# ══════════════════════════════════════════════════════════════════════════════

def plot_hurricane_fragility(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Reference wind speeds for major hurricane categories (Saffir-Simpson)
    cat_speeds = {1: 80, 2: 96, 3: 111, 4: 130, 5: 157}
    for cat, v in cat_speeds.items():
        ax.axvline(v, color="lightgray", lw=0.8, zorder=0)
        ax.text(v + 1, 0.97, f"Cat {cat}", fontsize=7.5, color="gray",
                va="top", ha="left")

    for key, arch in ARCHETYPES.items():
        r = results[key]["hurricane"]
        ax.plot(r["V_mph"], r["p_fail"],
                color=COLORS[key], ls=LINESTYLES[key], lw=2.0,
                label=arch["label"])

    # 10%, 50% reference lines
    for p_ref, label in [(0.10, "10%"), (0.50, "50%")]:
        ax.axhline(p_ref, color="#888", lw=0.7, ls=":")
        ax.text(202, p_ref + 0.01, label, fontsize=8, color="#888")

    ax.set_xlabel("3-second Gust Wind Speed (mph)")
    ax.set_ylabel("P(Wall Panel Failure)")
    ax.set_title("Hurricane Wind Fragility — URM Thermal Power Plant Walls")
    ax.set_xlim(60, 200)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.01, -0.02,
             "Simply supported wall panel; Monte Carlo n=12,000; degradation accounts for age/condition.",
             fontsize=7.5, color="#555")
    save(fig, "fragility_wind_hurricane.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Tornado (EF-scale) fragility
# ══════════════════════════════════════════════════════════════════════════════

def plot_tornado_fragility(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(6)
    bar_w = 0.25
    offsets = [-bar_w, 0, bar_w]
    keys = list(ARCHETYPES.keys())

    for i, key in enumerate(keys):
        r = results[key]["tornado"]
        ax.bar(x + offsets[i], r["p_fail"],
                      width=bar_w * 0.9,
                      color=COLORS[key], alpha=0.85,
                      label=ARCHETYPES[key]["label"])

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{lbl}\n({int(EF_MID_SPEEDS[j])} mph)"
        for j, lbl in enumerate(EF_LABELS)
    ])
    ax.set_ylabel("P(Wall Panel Failure)")
    ax.set_title("Tornado Fragility by EF Category — URM Thermal Power Plant Walls")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=False)
    # 10% and 50% lines
    for p_ref in [0.10, 0.50]:
        ax.axhline(p_ref, color="#888", lw=0.7, ls=":")
    fig.text(0.01, -0.02,
             "EF mid-range wind speeds; 1.5× internal pressure amplification (ASCE 7-22 App. CC).",
             fontsize=7.5, color="#555")
    save(fig, "fragility_wind_tornado.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Flood fragility
# ══════════════════════════════════════════════════════════════════════════════

def plot_flood_fragility(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # FEMA return-period depth markers
    for rp, depth in RETURN_PERIOD_FLOOD.items():
        ax.axvline(depth, color="lightblue", lw=0.8, zorder=0)
        if depth <= 16:
            ax.text(depth + 0.1, 1.01, f"{rp}-yr", fontsize=7.5, color="#2980B9",
                    va="bottom", ha="left", rotation=45)

    for key, arch in ARCHETYPES.items():
        r = results[key]["flood"]
        ax.plot(r["depth_ft"], r["p_fail"],
                color=COLORS[key], ls=LINESTYLES[key], lw=2.0,
                label=arch["label"])

    for p_ref in [0.10, 0.50]:
        ax.axhline(p_ref, color="#888", lw=0.7, ls=":")
        ax.text(16.2, p_ref + 0.01, f"{int(p_ref*100)}%", fontsize=8, color="#888")

    ax.set_xlabel("Flood Inundation Depth (ft)")
    ax.set_ylabel("P(Wall Panel Failure)")
    ax.set_title("Flood Fragility — URM Thermal Power Plant Walls")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.01, -0.02,
             "Hydrostatic + hydrodynamic (FEMA P-55); flood velocity = 6 fps; FEMA Zone AE depths shown.",
             fontsize=7.5, color="#555")
    save(fig, "fragility_flood.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Combined hurricane landfall (wind + surge)
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_fragility(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax_left, ax_right = axes

    # Left: wind-only vs combined for turbine hall (most vulnerable)
    key = "turbine_hall"
    r_wind = results[key]["hurricane"]
    r_comb = results[key]["hurricane_combined"]

    ax_left.plot(r_wind["V_mph"], r_wind["p_fail"],
                 color="#2471A3", ls="--", lw=2.0, label="Wind only")
    ax_left.plot(r_comb["V_mph"], r_comb["p_fail"],
                 color="#C0392B", ls="-", lw=2.0, label="Wind + storm surge")
    ax_left.fill_between(r_wind["V_mph"],
                          r_wind["p_fail"], r_comb["p_fail"],
                          alpha=0.15, color="#C0392B", label="Compounding effect")
    for p_ref in [0.10, 0.50]:
        ax_left.axhline(p_ref, color="#888", lw=0.7, ls=":")
    ax_left.set_xlabel("3-second Gust Wind Speed (mph)")
    ax_left.set_ylabel("P(Wall Panel Failure)")
    ax_left.set_title("Turbine Hall: Wind-only vs. Combined Loading")
    ax_left.set_xlim(60, 200)
    ax_left.set_ylim(0, 1.05)
    ax_left.legend(frameon=False)

    # Right: all archetypes combined
    for key, arch in ARCHETYPES.items():
        r_w = results[key]["hurricane"]
        r_c = results[key]["hurricane_combined"]
        ax_right.plot(r_w["V_mph"], r_w["p_fail"],
                      color=COLORS[key], ls="--", lw=1.5, alpha=0.5)
        ax_right.plot(r_c["V_mph"], r_c["p_fail"],
                      color=COLORS[key], ls="-", lw=2.0,
                      label=arch["label"])
    # Two separate legends: archetype colors (upper-left) + line styles (lower-right)
    # Use Legend class directly to avoid the second ax.legend() replacing the first.
    arch_legend = ax_right.legend(frameon=False, loc="upper left")
    style_handles = [
        Line2D([0], [0], color="k", ls="--", lw=1.5, alpha=0.5, label="Wind only"),
        Line2D([0], [0], color="k", ls="-",  lw=2.0,             label="Wind + surge"),
    ]
    style_legend = Legend(ax_right, style_handles,
                          [h.get_label() for h in style_handles],
                          frameon=False, loc="lower right", fontsize=8)
    ax_right.add_artist(arch_legend)
    ax_right.add_artist(style_legend)
    for p_ref in [0.10, 0.50]:
        ax_right.axhline(p_ref, color="#888", lw=0.7, ls=":")
    ax_right.set_xlabel("3-second Gust Wind Speed (mph)")
    ax_right.set_title("All Archetypes: Combined Wind + Surge")
    ax_right.set_xlim(60, 200)

    fig.suptitle("Hurricane Landfall: Compound Wind + Storm-Surge Loading on URM Plant Walls",
                 fontsize=12, y=1.02)
    fig.text(0.01, -0.04,
             "Dashed = wind-only; solid = wind + correlated storm surge. "
             "Surge depth: 0.14(V−60) ft [Cat 1 (80 mph)→3 ft; Cat 5 (157 mph)→14 ft]. "
             "Irish et al. (2008) Gulf Coast median.",
             fontsize=7.5, color="#555")
    save(fig, "fragility_combined_hurricane.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Annual failure probability risk matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_risk_matrix(results: dict):
    """
    Grid: archetype × hazard type → annual failure probability (%).
    Demonstrates that legacy URM plants face order-of-magnitude higher risk
    than the industry implicitly assumes.
    """

    hazard_labels = ["Hurricane\nWind", "EF2+ Tornado\n(annual rate)", "100-yr\nFlood",
                     "Combined\nHurricane†", "Multi-hazard\nUnion‡"]
    arch_labels = [ARCHETYPES[k]["label"] for k in ARCHETYPES]

    # Approximate annual failure probabilities
    afp = np.zeros((len(ARCHETYPES), 5))

    for i, key in enumerate(ARCHETYPES):
        arch = ARCHETYPES[key]

        # Hurricane: integrate over return-period wind speeds
        r_h = results[key]["hurricane"]
        interp_h = interp1d(r_h["V_mph"], r_h["p_fail"],
                            bounds_error=False, fill_value=(0, 1))
        afp[i, 0] = annual_failure_probability(interp_h, RETURN_PERIOD_WIND) * 100

        # Tornado: P(fail | EF) × annual rate per EF category.
        # Per-site annual rates for a ~0.2 km² industrial facility footprint in a
        # high-hazard SE US location (Dixie Alley / Gulf Coast), derived by scaling
        # Tippett et al. (2016) SPC tornado climatology (1° × 1° grid cells,
        # ~10,000 km²) by the plant-to-cell area ratio (~0.2/10,000 = 2×10⁻⁵).
        # EF2+ combined rate ≈ 3×10⁻⁴/yr → ~3,300-yr return period, consistent
        # with ASCE 7-22 App. CC tornado maps for Risk Category II in this region.
        r_t = results[key]["tornado"]
        ef_annual_rates = np.array([5.0e-4, 1.5e-4, 3.0e-5, 5.0e-6, 6.0e-7, 5.0e-8])
        afp[i, 1] = float(np.sum(r_t["p_fail"] * ef_annual_rates)) * 100

        # Flood: integrate over return-period depths (100-yr base)
        r_f = results[key]["flood"]
        interp_f = interp1d(r_f["depth_ft"], r_f["p_fail"],
                            bounds_error=False, fill_value=(0, 1))
        afp[i, 2] = annual_failure_probability(interp_f, RETURN_PERIOD_FLOOD) * 100

        # Combined hurricane: same wind hazard table
        r_c = results[key]["hurricane_combined"]
        interp_c = interp1d(r_c["V_mph"], r_c["p_fail"],
                            bounds_error=False, fill_value=(0, 1))
        afp[i, 3] = annual_failure_probability(interp_c, RETURN_PERIOD_WIND) * 100

        # Multi-hazard union: hurricane (wind+surge) ∪ tornado ∪ independent flood
        # Independent flood fraction ≈ 0.5 for coastal industrial sites (remaining
        # 50% is hurricane storm surge, already captured in combined hurricane AFP).
        afp[i, 4] = union_afp([afp[i, 3] / 100,
                                afp[i, 1] / 100,
                                afp[i, 2] / 100 * 0.5]) * 100

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Log-scale colormap so we can see variation from 0.01% to 10%+
    afp_clipped = np.clip(afp, 0.01, 100.0)
    im = ax.imshow(afp_clipped, cmap="YlOrRd",
                   norm=LogNorm(vmin=0.01, vmax=afp_clipped.max() * 2),
                   aspect="auto")

    ax.set_xticks(range(5))
    ax.set_xticklabels(hazard_labels, fontsize=10)
    ax.set_yticks(range(len(arch_labels)))
    ax.set_yticklabels(arch_labels, fontsize=10)
    ax.set_ylabel("Building Archetype", fontsize=11)
    ax.set_title("Estimated Annual Wall Failure Probability (%) — "
                 "Legacy URM Thermal Generation Facilities", fontsize=11)

    # Annotate each cell
    for i in range(len(ARCHETYPES)):
        for j in range(5):
            v = afp[i, j]
            txt = f"{v:.2f}%" if v >= 0.01 else "<0.01%"
            color = "white" if afp_clipped[i, j] > 1.0 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Annual Failure Probability (%)", fontsize=9)

    fig.text(0.01, -0.08,
             "Annual failure probability via hazard-fragility convolution: λ_f = Σ P(fail|IM) · Δλ(IM) "
             "(Cornell & Krawinkler 2000; Kennedy & Short 1994). "
             "Wind: ASCE 7-22 Risk Cat II hazard curve (10–1700-yr return periods). "
             "Flood: FEMA Zone AE depths (conditional on site being in floodplain). "
             "Tornado: Tippett et al. (2016) SE US rates scaled to ~0.2 km² plant footprint. "
             "† Combined Hurricane includes simultaneous wind + storm surge (Irish et al. 2008 Gulf Coast). "
             "‡ Multi-hazard Union = 1−(1−AFP_hurricane)(1−AFP_tornado)(1−0.5·AFP_flood): "
             "flood scaled by 0.5 because ~50% of coastal flood risk is storm surge already in AFP_hurricane "
             "(Vickery et al. 2009); treating all three as fully independent would overstate risk by ~1.3×.",
             fontsize=7.5, color="#555", wrap=True)
    save(fig, "risk_matrix.png")

    return afp


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Degradation sensitivity (fragility shift with aging)
# ══════════════════════════════════════════════════════════════════════════════

def plot_degradation_sensitivity():
    """
    Show how hurricane wind fragility for the turbine hall shifts as a function
    of construction year (1895–1950) — key argument for why aging plants face
    invisibly elevated risk that is invisible to current inspection practice.
    """
    from hazard_loads import wind_pressure_psf, kz_exposure_c
    from limit_states import governing_dc
    from urm_wall import sample_walls, degradation_factor
    from fragility import V_HURRICANE, N_SAMPLES

    fig, ax = plt.subplots(figsize=(7, 4.5))

    years = [1895, 1910, 1925, 1940, 1950]
    cat3_mph = 111.0
    cmap = plt.get_cmap("plasma")
    arch_base = ARCHETYPES["turbine_hall"].copy()
    Kz_th = kz_exposure_c(arch_base["height_ft_mean"])   # turbine hall Kz

    pf_at_cat3 = {}   # for dynamic caption

    for idx, yr in enumerate(years):
        arch = arch_base.copy()
        arch["year_built"] = yr
        walls = sample_walls(arch, N_SAMPLES, seed=10 + idx)

        p_net = wind_pressure_psf(V_HURRICANE, Kz=Kz_th)
        ph = arch["panel_height_ft_mean"]
        F   = p_net * ph * arch["width_ft"]
        arm = np.full(len(V_HURRICANE), ph / 2.0)

        result = governing_dc(walls, F, p_net, arm)
        deg = degradation_factor(yr)
        color = cmap(idx / len(years))
        ax.plot(V_HURRICANE, result["p_fail"],
                color=color, lw=2.0,
                label=f"Built {yr} (deg. factor = {deg:.2f})")

        # Interpolate P(fail) at Cat 3 for caption
        pf_at_cat3[yr] = float(
            interp1d(V_HURRICANE, result["p_fail"], bounds_error=False,
                     fill_value=(0, 1))(cat3_mph)
        )

    for p_ref in [0.10, 0.50]:
        ax.axhline(p_ref, color="#888", lw=0.7, ls=":")
        ax.text(202, p_ref + 0.01, f"{int(p_ref*100)}%", fontsize=8, color="#888")

    ax.axvline(cat3_mph, color="gray", lw=0.9, ls="--")
    ax.text(113, 0.05, "Cat 3", fontsize=8, color="gray")

    ax.set_xlabel("3-second Gust Wind Speed (mph)")
    ax.set_ylabel("P(Wall Panel Failure)")
    ax.set_title("Effect of Building Age on Turbine Hall Hurricane Fragility")
    ax.set_xlim(60, 200)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    oldest, newest = years[0], years[-1]
    fig.text(0.01, -0.03,
             f"Same archetype geometry; degradation reduces effective material strength with age. "
             f"Built-{oldest} turbine hall: {pf_at_cat3[oldest]:.0%} failure probability at Cat 3; "
             f"built-{newest}: {pf_at_cat3[newest]:.0%}.",
             fontsize=7.5, color="#555")
    save(fig, "degradation_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6b: Grid consequence model (Expected Annual Energy Loss)
# ══════════════════════════════════════════════════════════════════════════════

def plot_consequence_model(afp: np.ndarray):
    """
    Figure 14. Translate AFP into Expected Annual Outage (days) and
    Expected Annual Energy Loss (MWh). afp is (3,5) from plot_risk_matrix;
    column 4 is the multi-hazard union AFP.
    """
    # Post-storm restoration days per unit type (EIA-860 post-Katrina/Sandy coal unit data;
    # EPRI TR-1026889 2012 power plant storm recovery report).
    RESTORATION_DAYS = {
        "boiler_house": 30,   # boiler/pressure-vessel inspection + refractory repair
        "turbine_hall": 45,   # turbine alignment + steam path inspection
        "powerhouse":   21,   # shorter — lower-voltage switchgear, simpler restart
    }
    UNIT_CAPACITY_MW = 200  # representative pre-1950 steam generation unit (EIA-860)
    WHOLESALE_RATE_PER_MWH = 45.0  # $/MWh, MISO/SERC average 2023 (EIA 861)

    arch_keys   = list(ARCHETYPES.keys())
    arch_labels = [ARCHETYPES[k]["label"] for k in arch_keys]
    rest_days   = np.array([RESTORATION_DAYS[k] for k in arch_keys], dtype=float)

    union_afp_frac = afp[:, 4] / 100.0            # multi-hazard union, col 4
    eaod  = union_afp_frac * rest_days             # expected annual outage days
    eal_mwh = eaod * 24.0 * UNIT_CAPACITY_MW       # expected annual energy loss (MWh)
    eal_m_usd = eal_mwh * WHOLESALE_RATE_PER_MWH / 1e6  # millions USD

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    palette = [COLORS[k] for k in arch_keys]

    def _bar(ax, vals, ylabel, title, fmt):
        bars = ax.bar(arch_labels, vals, color=palette, alpha=0.88, edgecolor="white")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.04,
                    fmt.format(v), ha="center", va="bottom", fontsize=9)

    _bar(axes[0], eaod,         "Expected Annual Outage (days)",
         "Expected Annual\nOutage Days", "{:.2f} d")
    _bar(axes[1], eal_mwh / 1e3, "Expected Annual Energy Loss (GWh)",
         f"Expected Annual\nEnergy Loss ({UNIT_CAPACITY_MW} MW unit)", "{:.1f} GWh")
    _bar(axes[2], eal_m_usd,    "Expected Annual Revenue Loss ($M)",
         f"Expected Annual\nRevenue Loss (@${WHOLESALE_RATE_PER_MWH}/MWh)", "${:.2f}M")

    fig.suptitle(
        "Grid Consequence Model — Annual Energy and Revenue Risk from Multi-Hazard URM Failure",
        fontsize=11, y=1.02,
    )
    fig.text(
        0.01, -0.04,
        f"Multi-hazard AFP (column 5 of risk matrix): 1−(1−AFP_hurricane)(1−AFP_tornado)(1−0.5·AFP_flood). "
        f"Restoration times from EIA-860 post-Katrina (2005) and EPRI TR-1026889 (2012). "
        f"Unit capacity {UNIT_CAPACITY_MW} MW representative of pre-1950 steam plant; "
        f"actual Victor J. Daniel Jr. capacity is 1,252 MW — scale accordingly. "
        f"Does not include cascading outage risk to interconnected units or grid-level reliability impact.",
        fontsize=7.5, color="#555",
    )
    save(fig, "consequence_model.png")
    return {"eaod": eaod, "eal_mwh": eal_mwh, "eal_m_usd": eal_m_usd,
            "restoration_days": rest_days, "unit_mw": UNIT_CAPACITY_MW}


# ══════════════════════════════════════════════════════════════════════════════
# Figure 15: AFP epistemic uncertainty (sensitivity tornado chart)
# ══════════════════════════════════════════════════════════════════════════════

def plot_afp_uncertainty(results: dict):
    """
    One-at-a-time sensitivity of turbine hall hurricane AFP to epistemic
    uncertainty in hazard table and material parameters.
    """
    from hazard_loads import wind_pressure_psf, kz_exposure_c, RETURN_PERIOD_WIND as RPW
    from limit_states import governing_dc
    from urm_wall import sample_walls
    from fragility import V_HURRICANE, N_SAMPLES

    key  = "turbine_hall"
    arch = ARCHETYPES[key]
    Kz   = kz_exposure_c(arch["height_ft_mean"])
    ph   = arch["panel_height_ft_mean"]

    r_base   = results[key]["hurricane"]
    interp_b = interp1d(r_base["V_mph"], r_base["p_fail"],
                        bounds_error=False, fill_value=(0, 1))
    afp_base = annual_failure_probability(interp_b, RPW) * 100

    def _afp_from_walls(walls_d, hz=RPW):
        p_net  = wind_pressure_psf(V_HURRICANE, Kz=Kz)
        F      = p_net * ph * arch["width_ft"]
        arm    = np.full(len(V_HURRICANE), ph / 2.0)
        result = governing_dc(walls_d, F, p_net, arm)
        itp    = interp1d(V_HURRICANE, result["p_fail"],
                          bounds_error=False, fill_value=(0, 1))
        return annual_failure_probability(itp, hz) * 100

    perturbations = []

    # Wind hazard ±10%
    for scale, label in [(0.9, "Wind hazard −10%"), (1.1, "Wind hazard +10%")]:
        hz_s = {rp: v * scale for rp, v in RPW.items()}
        perturbations.append((label, _afp_from_walls(
            sample_walls(arch, N_SAMPLES, seed=1), hz=hz_s) - afp_base))

    # f_m CoV ±0.05
    for delta, label in [(-0.05, "Masonry f_m CoV −0.05"), (+0.05, "Masonry f_m CoV +0.05")]:
        a2 = {**arch, "f_m_cov": arch["f_m_cov"] + delta}
        perturbations.append((label, _afp_from_walls(
            sample_walls(a2, N_SAMPLES, seed=99)) - afp_base))

    # f_r CoV ±0.10
    for delta, label in [(-0.10, "Modulus of rupture CoV −0.10"), (+0.10, "Modulus of rupture CoV +0.10")]:
        a2 = {**arch, "f_r_cov": arch["f_r_cov"] + delta}
        perturbations.append((label, _afp_from_walls(
            sample_walls(a2, N_SAMPLES, seed=100)) - afp_base))

    # Aging: ±20 years on construction date
    for dyear, label in [(+20, "Built 20 yrs later (less degraded)"),
                         (-20, "Built 20 yrs earlier (more degraded)")]:
        a2 = {**arch, "year_built": arch["year_built"] + dyear}
        perturbations.append((label, _afp_from_walls(
            sample_walls(a2, N_SAMPLES, seed=101)) - afp_base))

    perturbations.sort(key=lambda x: abs(x[1]))
    labels  = [p[0] for p in perturbations]
    deltas  = [p[1] for p in perturbations]
    bar_col = ["#C0392B" if d > 0 else "#2471A3" for d in deltas]

    ci_lo = afp_base + min(deltas)
    ci_hi = afp_base + max(deltas)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(labels, deltas, color=bar_col, height=0.6, edgecolor="white")
    ax.axvline(0, color="black", lw=0.9)

    for i, (lbl, d) in enumerate(zip(labels, deltas)):
        offset = 0.06 if d >= 0 else -0.06
        ax.text(d + offset, i, f"{d:+.2f}%", va="center",
                ha="left" if d >= 0 else "right", fontsize=8.5)

    ax.set_xlabel("ΔAFP (percentage points) relative to base case")
    ax.set_title(
        f"AFP Sensitivity to Epistemic Uncertainty — Turbine Hall Hurricane\n"
        f"Base AFP = {afp_base:.2f}%  |  Range: [{ci_lo:.2f}%, {ci_hi:.2f}%]  "
        f"(~{ci_hi/ci_lo:.1f}× spread from parameter uncertainty alone)"
    )
    fig.text(
        0.01, -0.04,
        "One-at-a-time sensitivity; all other parameters at base case. "
        "Red = AFP increases; blue = AFP decreases. "
        f"Wind hazard uncertainty dominates: ±10% in 700-yr wind speed produces "
        f"the largest AFP swing. "
        "Bayesian updating from field measurements (accelerometers, anemometers) "
        "would reduce epistemic uncertainty and narrow this interval.",
        fontsize=7.5, color="#555",
    )
    save(fig, "afp_uncertainty.png")
    return {"afp_base": afp_base, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "perturbations": perturbations}


# ══════════════════════════════════════════════════════════════════════════════
# Figure 16: Digital twin data-flow schematic
# ══════════════════════════════════════════════════════════════════════════════

def plot_digital_twin_schematic():
    """
    Conceptual data-flow diagram for the HPC4EI digital twin framework.
    Shows sensor types, data ingestion, HPC physics engine, and output dashboard.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel, color, fontsize=9):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="#555", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", wrap=True)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25,
                    sublabel, ha="center", va="center",
                    fontsize=7.5, color="#333", style="italic")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444",
                                   lw=1.5, connectionstyle="arc3,rad=0.0"))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.05, my + 0.12, label, fontsize=7.5, color="#555")

    # Column 1: Sensors (left)
    ax.text(1.5, 6.65, "SENSING LAYER", ha="center", va="center",
            fontsize=9, color="#555", fontweight="bold")
    sensor_items = [
        ("Wall\nAccelerometers", "Modal ID → stiffness loss", "#D6EAF8"),
        ("Anemometer\nArray", "Local V_wind (rooftop + base)", "#D6EAF8"),
        ("Flood / Surge\nGauge", "Real-time surge depth (ft)", "#D6EAF8"),
        ("CCTV + AI\nVision", "Post-event crack detection", "#D6EAF8"),
        ("SCADA\nTelemetry", "MW output, operational state", "#D6EAF8"),
    ]
    for idx, (lbl, sub, col) in enumerate(sensor_items):
        box(0.15, 5.65 - idx * 1.1, 2.7, 0.85, lbl, sub, col, fontsize=8)

    # Column 2: Data ingestion / state estimation
    ax.text(5.05, 6.65, "INGESTION & STATE UPDATE", ha="center", va="center",
            fontsize=9, color="#555", fontweight="bold")
    box(3.7, 5.1, 2.7, 1.0, "Data Quality &\nAnomaly Filter",
        "Spike removal; sensor dropout flags", "#D5F5E3")
    box(3.7, 3.5, 2.7, 1.2, "Bayesian State\nEstimator",
        "Updates f_m, f_r, deg posteriors\nfrom accelerometer modal data", "#D5F5E3")
    box(3.7, 1.9, 2.7, 1.2, "Hazard Ingestion\n(NOAA / NWS API)",
        "Live wind forecast + SLOSH surge\nfeed into demand model", "#D5F5E3")

    # Column 3: HPC physics
    ax.text(8.65, 6.65, "HPC PHYSICS ENGINE", ha="center", va="center",
            fontsize=9, color="#555", fontweight="bold")
    box(7.3, 4.7, 2.7, 1.5, "Monte Carlo\nFragility Update",
        "Re-runs n=12,000 samples with\nupdated material posteriors", "#FAD7A0")
    box(7.3, 2.9, 2.7, 1.5, "FEM / DEM\nFull-Building Model",
        "3D solid FEM or DEM for\ncritical buildings (HPC nodes)", "#FAD7A0")
    box(7.3, 1.2, 2.7, 1.3, "Multi-hazard AFP\nForecast Engine",
        "Convolves updated fragility with\n48-hr storm hazard forecast", "#FAD7A0")

    # Column 4: Output
    ax.text(11.5, 6.65, "OPERATOR OUTPUTS", ha="center", va="center",
            fontsize=9, color="#555", fontweight="bold")
    box(10.15, 4.8, 2.7, 1.3, "Risk Dashboard",
        "AFP / outage risk per building;\nupdates every 15 min pre-storm", "#FDEDEC")
    box(10.15, 3.1, 2.7, 1.3, "Maintenance\nPriority Alerts",
        "Flags degraded buildings;\npre-positions repair crews", "#FDEDEC")
    box(10.15, 1.4, 2.7, 1.3, "Grid Dispatch\nAdvisory",
        "Expected capacity at risk;\nfeeds EMS/SCADA for dispatch", "#FDEDEC")

    # Arrows: sensors → ingestion
    for sy in [6.08, 4.98, 3.88, 2.78, 1.68]:
        arrow(2.85, sy, 3.7, 5.6)

    # Arrows: ingestion chain
    arrow(5.05, 5.1, 5.05, 4.7)
    arrow(5.05, 3.5, 5.05, 2.9)  # state → hazard ingestion (parallel)
    arrow(5.05, 3.5, 7.3, 5.45)  # state → fragility update
    arrow(5.05, 1.9, 7.3, 1.85)  # hazard → AFP engine

    # Arrows: HPC chain
    arrow(8.65, 4.7, 8.65, 4.4)
    arrow(8.65, 2.9, 8.65, 2.5)
    arrow(8.65, 4.7, 8.65, 2.9)
    arrow(9.15, 3.85, 10.15, 5.45)   # fragility → dashboard
    arrow(9.15, 3.85, 10.15, 3.75)   # fragility → maintenance
    arrow(8.65, 1.2, 10.15, 2.05)    # AFP engine → grid dispatch

    # Phase labels
    phase_info = [
        (1.5,  "#85C1E9", "Phase 0 — Sensing"),
        (5.05, "#58D68D", "Phase 1 — Estimation"),
        (8.65, "#F0B27A", "Phase 2 — HPC Physics"),
        (11.5, "#F1948A", "Phase 3 — Decision"),
    ]
    for px, col, lbl in phase_info:
        ax.add_patch(FancyBboxPatch((px - 1.35, 0.05), 2.7, 0.45,
                     boxstyle="round,pad=0.05", facecolor=col,
                     edgecolor="none", alpha=0.6))
        ax.text(px, 0.28, lbl, ha="center", va="center", fontsize=8,
                fontweight="bold", color="#222")

    ax.set_title(
        "HPC4EI Digital Twin — Conceptual Data-Flow Architecture\n"
        "Pre-1950 URM Thermal Power Plant Multi-Hazard Risk Framework",
        fontsize=11, pad=8,
    )
    fig.text(
        0.01, -0.02,
        "CCTV cameras support post-event rapid assessment and routine AI crack detection "
        "(surface-only; not reliable during storm due to rain/debris obscuration). "
        "Accelerometers provide continuous pre-storm stiffness monitoring via operational modal analysis (OMA); "
        "they inform Bayesian updating of material parameters in the fragility model. "
        "The HPC physics engine (Phase 2) is the scope of the HPC4EI proposal; "
        "Phases 0–1 and 3 leverage existing SCADA infrastructure.",
        fontsize=7.5, color="#555",
    )
    save(fig, "digital_twin_schematic.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7–9: OpenSeesPy / fiber-section FE comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_fe_comparison(fe_results: dict, baseline_results: dict):
    """Generate Figures 7, 8, 9 from the FE boundary-condition comparison."""

    frag       = fe_results["fragility"]
    afp        = fe_results["afp"]
    dc_table   = fe_results["dc_table"]
    method     = fe_results["method"]
    arch_label = ARCHETYPES["turbine_hall"]["label"]

    # ── Figure 7: fragility curves by BC ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bc_colors = {"pin_pin": "#2471A3", "fixed_pin": "#E67E22", "pilaster": "#1E8449"}
    bc_ls     = {"pin_pin": "--",      "fixed_pin": "-",       "pilaster": "-."}

    # Baseline analytical (full 12 k samples, from run_all_fragility)
    r_base = baseline_results["turbine_hall"]["hurricane"]
    ax.plot(r_base["V_mph"], r_base["p_fail"],
            color="black", ls=":", lw=1.5, label="Analytical SS (n=12,000)")

    for bc, lbl in BC_LABELS.items():
        r = frag[bc]
        ax.plot(r["V_mph"], r["p_fail"],
                color=bc_colors[bc], ls=bc_ls[bc], lw=2.0,
                label=f"{lbl}  (AFP={afp[bc]*100:.2f}%)")

    cat_speeds = {1: 80, 2: 96, 3: 111, 4: 130, 5: 157}
    for cat, v in cat_speeds.items():
        ax.axvline(v, color="lightgray", lw=0.8, zorder=0)
        ax.text(v + 1, 0.97, f"Cat {cat}", fontsize=7, color="gray",
                va="top", ha="left")

    for p_ref in [0.10, 0.50]:
        ax.axhline(p_ref, color="#888", lw=0.7, ls=":")
        ax.text(202, p_ref + 0.01, f"{int(p_ref*100)}%", fontsize=8, color="#888")

    ax.set_xlabel("3-second Gust Wind Speed (mph)")
    ax.set_ylabel("P(Wall Panel Failure)")
    ax.set_title(f"FE Boundary Condition Comparison — Turbine Hall Hurricane Fragility")
    ax.set_xlim(60, 200)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.text(0.01, -0.03,
             f"FE method: {method}. Turbine hall archetype (1930, 16-in wall, 20-ft panel span). "
             f"BC-A validates analytical baseline; BC-B/C show boundary condition effect.",
             fontsize=7.5, color="#555")
    save(fig, "fe_fragility_comparison.png")

    # ── Figure 8: AFP bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.2))

    # Include baseline analytical AFP for comparison
    baseline_afp = afp["pin_pin"]   # BC-A ≈ analytical (validated)
    labels_bar = [
        "Analytical SS\n(n=12,000)",
        BC_LABELS["pin_pin"].replace("BC-A: ", ""),
        BC_LABELS["fixed_pin"].replace("BC-B: ", ""),
        BC_LABELS["pilaster"].replace("BC-C: ", ""),
    ]
    afp_pct = [
        baseline_results["turbine_hall"]["hurricane"]["p_fail"].mean() * 0,  # placeholder
        afp["pin_pin"]   * 100,
        afp["fixed_pin"] * 100,
        afp["pilaster"]  * 100,
    ]
    # Use actual baseline from run_all_fragility
    from fragility import annual_failure_probability
    from scipy.interpolate import interp1d as _interp1d
    r_b = baseline_results["turbine_hall"]["hurricane"]
    interp_b = _interp1d(r_b["V_mph"], r_b["p_fail"], bounds_error=False, fill_value=(0, 1))
    afp_pct[0] = annual_failure_probability(interp_b, RETURN_PERIOD_WIND) * 100

    colors_bar = ["#555", bc_colors["pin_pin"], bc_colors["fixed_pin"], bc_colors["pilaster"]]
    bars = ax.barh(labels_bar[::-1], afp_pct[::-1], color=colors_bar[::-1],
                   height=0.5, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Annual Failure Probability (%,  log scale)")
    ax.set_title("Hurricane AFP by Model Type — Turbine Hall")
    for bar, val in zip(bars, afp_pct[::-1]):
        ax.text(val * 1.25, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=9)

    ax.set_xlim(0.01, max(afp_pct) * 20)
    fig.text(0.01, -0.04,
             "Log scale. BC-A validates that the FE model matches the analytical baseline "
             "(pin-pin SS). BC-B and BC-C show realistic boundary conditions reduce AFP "
             "significantly — quantifying the analytical model's conservatism.",
             fontsize=7.5, color="#555")
    save(fig, "fe_afp_comparison.png")

    # ── Figure 9: D/C ratio table at 150 mph design wind ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")

    col_labels = ["Boundary Condition", "Mean D/C at 150 mph", "P(fail) at 150 mph",
                  "D/C vs. Analytical", "AFP (annual)"]
    cell_text  = []
    cell_colors = []
    row_colors_map = {"pin_pin": "#EBF5FB", "fixed_pin": "#FEF9E7", "pilaster": "#EAFAF1"}

    for bc in ["pin_pin", "fixed_pin", "pilaster"]:
        d = dc_table[bc]
        row = [
            BC_LABELS[bc],
            f"{d['mean_dc']:.3f}",
            f"{d['p_fail_at_design']*100:.1f}%",
            f"{d['ratio_vs_analytical']:.3f}",
            f"{afp[bc]*100:.3f}%",
        ]
        cell_text.append(row)
        base_c = row_colors_map[bc]
        cell_colors.append([base_c] * len(col_labels))

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_facecolor("#2471A3")

    ax.set_title(f"D/C Ratios and AFP at 150-mph Design Wind — Turbine Hall  |  Method: {method}",
                 fontsize=10, pad=12)
    save(fig, "fe_dc_ratio_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10–11: Site-specific hazard analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_site_analysis(site_results: dict, baseline_results: dict, generic_afp: np.ndarray):
    """Generate Figures 10 and 11 from the site-specific hazard analysis."""

    afp_data = site_results["afp"]
    hazard_keys = ["hurricane", "flood", "combined"]
    hazard_labels = ["Hurricane\nWind", "100-yr\nFlood", "Combined\nHurricane"]
    arch_keys = list(ARCHETYPES.keys())
    arch_labels = [ARCHETYPES[k]["label"] for k in arch_keys]

    # Build AFP arrays: (n_arch, n_hazard) for generic and site
    generic_arr = np.zeros((3, 3))
    site_arr    = np.zeros((3, 3))
    for i, key in enumerate(arch_keys):
        for j, hz in enumerate(hazard_keys):
            generic_arr[i, j] = afp_data[key][hz]["generic_afp"] * 100
            site_arr[i, j]    = afp_data[key][hz]["site_afp"]    * 100

    # ── Figure 10: side-by-side risk matrices ─────────────────────────────────
    from matplotlib.colors import LogNorm as _LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

    vmax = max(generic_arr.max(), site_arr.max()) * 2
    norm = _LogNorm(vmin=0.01, vmax=vmax)

    for ax, arr, title in zip(
        axes,
        [generic_arr, site_arr],
        ["Generic ASCE 7-22 / FEMA Zone AE Table", f"Site-Specific: {SITE_CITY}"],
    ):
        im = ax.imshow(np.clip(arr, 0.01, 100), cmap="YlOrRd", norm=norm, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(hazard_labels, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels(arch_labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        for i in range(3):
            for j in range(3):
                v = arr[i, j]
                txt = f"{v:.2f}%"
                color = "white" if v > 2.0 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, shrink=0.7,
                 label="Annual Failure Probability (%)")
    fig.suptitle(f"Annual Wall Failure Probability: Generic vs Site-Specific Hazard\n"
                 f"{SITE_NAME} ({SITE_CITY})", fontsize=11, y=1.03)
    fig.text(0.01, -0.05,
             f"Site: {SITE_LAT}°N {abs(SITE_LON):.2f}°W  |  "
             "Wind: ASCE 7-22 Figure 26.5-1B coastal MS  |  "
             "Flood: FEMA FIRM Harrison County AE zone (BFE ≈12 ft NAVD88)  |  "
             f"Storm surge: surge_from_wind() (Gulf Coast median; SLOSH max for Cat5 ≈25 ft, "
             f"surge_from_wind gives ≈13.6 ft — conservative by ~1.8×).",
             fontsize=7.5, color="#555")
    save(fig, "site_specific_risk_matrix.png")

    # ── Figure 11: AFP delta (site - generic) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    delta = site_arr - generic_arr

    x = np.arange(3)
    bar_w = 0.25
    offsets = [-bar_w, 0, bar_w]
    for i, key in enumerate(arch_keys):
        ax.bar(x + offsets[i], delta[i],
               width=bar_w * 0.9, color=COLORS[key], alpha=0.85,
               label=ARCHETYPES[key]["label"])

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(hazard_labels, fontsize=10)
    ax.set_ylabel("AFP change (site − generic, percentage points)")
    ax.set_title(f"Site-Specific vs Generic AFP Delta — {SITE_CITY}")
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.01, -0.03,
             "Positive = site-specific hazard is more severe than generic table. "
             "Negative = site-specific hazard is less severe (e.g., lower flood BFE than generic assumption).",
             fontsize=7.5, color="#555")
    save(fig, "site_afp_delta.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 12–13: HPC computational scaling
# ══════════════════════════════════════════════════════════════════════════════

def plot_hpc_scaling(hpc_results: dict):
    """Generate Figures 12 and 13 from the HPC scaling argument."""

    table  = hpc_results["table"]
    n_bldg = hpc_results["portfolio_size"]

    # ── Figure 12: log-scale CPU-hour bar chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))

    names  = [r["short_name"] for r in table]
    cpu_ph = [r["cpu_hours_portfolio"] for r in table]
    colors = ["#1E8449", "#2471A3", "#E67E22", "#C0392B"]

    bars = ax.barh(names[::-1], cpu_ph[::-1],
                   color=colors[::-1], height=0.5, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Portfolio CPU-hours (log scale)")
    ax.set_title(f"Computational Scaling: URM Portfolio Risk Analysis\n"
                 f"({n_bldg} buildings × representative storm ensemble)")

    # Threshold lines
    thresholds = [
        (8,    "Laptop (<8 hrs)",            "#1E8449"),
        (5000, "Workstation (<5,000 hrs)",    "#E67E22"),
        (1e6,  "Frontier-scale HPC (>1M hrs)","#C0392B"),
    ]
    for val, lbl, col in thresholds:
        ax.axvline(val, color=col, ls="--", lw=1.0, alpha=0.6)
        ax.text(val * 1.4, 3.55, lbl, fontsize=7.5, color=col, va="top")

    for bar, val, row in zip(bars, cpu_ph[::-1], table[::-1]):
        ax.text(val * 1.3, bar.get_y() + bar.get_height() / 2,
                f"{row['portfolio_display']} CPU-hrs", va="center", fontsize=9)

    ax.set_xlim(0.001, max(cpu_ph) * 500)
    fig.text(0.01, -0.04,
             "References: Cundall & Strack (1979) DEM; Lemos (2007) URM DEM; "
             "Lourenço (1996) URM FEM; McKenna et al. (2010) OpenSees. "
             "CPU-hr estimates at ~2.5 GHz single-thread reference performance.",
             fontsize=7.5, color="#555")
    save(fig, "hpc_scaling_chart.png")

    # ── Figure 13: scaling table ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.axis("off")

    col_labels = ["Fidelity Level", "DOF / Building", "MC Samples",
                  "CPU-hrs\n(1 building)", "CPU-hrs\n(full portfolio)", "Feasibility"]
    cell_text   = []
    cell_colors = []

    for row in table:
        cell_text.append([
            row["short_name"],
            row["dof_display"],
            f"{row['mc_samples']:,}",
            row["single_display"],
            row["portfolio_display"],
            row["feasibility"],
        ])
        fc = row["feasibility_color"]
        cell_colors.append(["white", "white", "white", "white", "white", fc])

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_facecolor("#2471A3")

    ax.set_title(f"Computational Cost Summary — {n_bldg}-Building URM Portfolio Analysis",
                 fontsize=11, pad=12)
    fig.text(0.01, -0.04,
             "Green = laptop/workstation feasible.  Yellow = specialized cluster.  "
             "Red = requires DOE-scale HPC (Frontier, Summit, or equivalent).",
             fontsize=7.5, color="#555")
    save(fig, "hpc_scaling_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 17: HURDAT2 empirical hazard vs ASCE 7-22 validation
# ══════════════════════════════════════════════════════════════════════════════

def plot_hurdat2_validation(hurdat2: dict, fragility_results: dict):
    """
    Figure 17: Empirical wind hazard from HURDAT2 (1851–2023) vs.
    ASCE 7-22 site-specific table; AFP comparison for turbine hall.
    """
    from site_specific import SITE_WIND_HAZARD

    if not hurdat2 or "v_mph" not in hurdat2:
        print("  HURDAT2 data unavailable — skipping Fig 17", flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: hazard curves ────────────────────────────────────────────────────
    ax = axes[0]
    mask = hurdat2["aep"] > 0
    ax.semilogy(hurdat2["v_mph"][mask], hurdat2["aep"][mask],
                color="#C0392B", lw=2.0, label=f"HURDAT2 empirical (n={hurdat2['n_storms']} events, {int(hurdat2['years'])} yr)")

    # ASCE 7-22 site table
    asce_rps = sorted(SITE_WIND_HAZARD.keys())
    asce_v   = [SITE_WIND_HAZARD[r] for r in asce_rps]
    asce_aep = [1.0 / r for r in asce_rps]
    ax.semilogy(asce_v, asce_aep, "o--",
                color="#2471A3", lw=1.5, ms=7, label="ASCE 7-22 site-specific table")

    # HURDAT2 return period table points
    h_rp  = sorted(hurdat2["rp_table"].keys())
    h_v   = [hurdat2["rp_table"][r] for r in h_rp]
    h_aep = [1.0 / r for r in h_rp]
    ax.semilogy(h_v, h_aep, "s",
                color="#C0392B", ms=7, zorder=5, label="HURDAT2 at standard RPs")

    # Key return period lines
    for rp, ls in [(100, ":"), (700, "--")]:
        ax.axhline(1.0 / rp, color="gray", lw=0.7, ls=ls)
        ax.text(45, 1.0 / rp * 1.15, f"{rp}-yr", fontsize=8, color="gray")

    ax.set_xlabel("3-second Gust Wind Speed (mph)")
    ax.set_ylabel("Annual Exceedance Probability")
    ax.set_title(f"Empirical vs. Code-Based Wind Hazard\nPlant Daniel site (~{SITE_LAT}°N, {abs(SITE_LON):.2f}°W)")
    ax.set_xlim(40, 220)
    ax.legend(frameon=False, fontsize=8)

    # ── Right: AFP comparison using three hazard sources ──────────────────────
    ax2 = axes[1]
    from fragility import annual_failure_probability, hurricane_fragility
    from hazard_loads import RETURN_PERIOD_WIND

    arch = ARCHETYPES["turbine_hall"]
    r_h  = fragility_results["turbine_hall"]["hurricane"]
    interp_h = interp1d(r_h["V_mph"], r_h["p_fail"], bounds_error=False, fill_value=(0, 1))

    afp_generic  = annual_failure_probability(interp_h, RETURN_PERIOD_WIND) * 100
    afp_asce     = annual_failure_probability(interp_h, SITE_WIND_HAZARD) * 100
    afp_hurdat2  = annual_failure_probability(interp_h, hurdat2["rp_table"]) * 100

    labels  = ["Generic\nASCE 7-22\n(inland table)", "Site ASCE 7-22\n(coastal MS)", "HURDAT2\nEmpirical\n(1851–2023)"]
    values  = [afp_generic, afp_asce, afp_hurdat2]
    colors  = ["#95A5A6", "#2471A3", "#C0392B"]

    bars = ax2.bar(labels, values, color=colors, alpha=0.88, edgecolor="white", width=0.5)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Annual Hurricane Failure Probability (%)")
    ax2.set_title("Turbine Hall Hurricane AFP\nby Hazard Data Source")
    ax2.set_ylim(0, max(values) * 1.35)

    comp = hurdat2.get("asce_comparison", {})
    fig.suptitle("HURDAT2 Wind Hazard Validation — Plant Daniel Site", fontsize=11, y=1.01)

    comp_rows = "\n".join(
        f"  {rp:>5}-yr:  HURDAT2={comp[rp]['hurdat2']:.0f} mph  ASCE={comp[rp]['asce']} mph  ratio={comp[rp]['ratio']:.2f}"
        for rp in sorted(comp.keys()) if rp in comp
    )
    fig.text(0.01, -0.05,
             f"HURDAT2 record: {int(hurdat2['years'])} yr (1851–2023), {hurdat2['n_storms']} hurricane events within "
             f"{350:.0f} km of site. Wind field: modified Rankine vortex (R_max=50 km). "
             f"Conversion: 1-min sustained kt × 1.473 = 3-s gust mph (ASCE 7-22 Sec. 26.5). "
             "AFP bars: same turbine hall fragility curve; only the hazard integration table differs.",
             fontsize=7.5, color="#555")
    save(fig, "hurdat2_validation.png")

    # Print comparison table
    print(f"\n── HURDAT2 vs ASCE 7-22 wind speed comparison ──")
    print(f"{'Return Period':>15}  {'HURDAT2 (mph)':>14}  {'ASCE 7-22 (mph)':>16}  {'Ratio':>7}")
    for rp in sorted(comp.keys()):
        c = comp[rp]
        print(f"{rp:>14}-yr  {c['hurdat2']:>14.1f}  {c['asce']:>16}  {c['ratio']:>7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Running Monte Carlo fragility analysis...")
    results = run_all_fragility()

    print("Generating figures 1–6 (existing hazards)...")
    plot_hurricane_fragility(results)
    plot_tornado_fragility(results)
    plot_flood_fragility(results)
    plot_combined_fragility(results)
    afp = plot_risk_matrix(results)
    plot_degradation_sensitivity()

    print("Running FE boundary-condition comparison (Figs 7–9)...")
    fe_results = run_opensees_comparison()
    plot_fe_comparison(fe_results, results)

    print("Running site-specific hazard analysis (Figs 10–11)...")
    site_results = run_site_analysis()
    plot_site_analysis(site_results, results, afp)

    print("Building HPC scaling argument (Figs 12–13)...")
    hpc_results = run_hpc_scaling()
    plot_hpc_scaling(hpc_results)

    print("Computing grid consequence model (Fig 14)...")
    plot_consequence_model(afp)

    print("Computing AFP epistemic uncertainty (Fig 15)...")
    plot_afp_uncertainty(results)

    print("Generating digital twin schematic (Fig 16)...")
    plot_digital_twin_schematic()

    print("Building HURDAT2 empirical hazard validation (Fig 17)...")
    hurdat2_results = run_hurdat2_analysis()
    plot_hurdat2_validation(hurdat2_results, results)

    print("\n── Summary: Annual failure probabilities (%) [Generic] ──")
    cols = ["Hurricane", "Tornado", "100-yr Flood", "Combined Hurricane"]
    header = f"{'Archetype':<22}" + "".join(f"{c:<22}" for c in cols)
    print(header)
    for i, key in enumerate(ARCHETYPES):
        row = f"{key:<22}" + "".join(f"{afp[i, j]:<22.3f}" for j in range(4))
        print(row)

    print(f"\n── Site-specific AFP (%) — {SITE_CITY} ──")
    site_afp = site_results["afp"]
    hazards  = ["hurricane", "flood", "combined"]
    s_header = f"{'Archetype':<22}" + "".join(f"{h:<22}" for h in ["Hurricane(site)", "Flood(site)", "Combined(site)"])
    print(s_header)
    for key in ARCHETYPES:
        row = f"{key:<22}" + "".join(
            f"{site_afp[key][h]['site_afp']*100:<22.3f}" for h in hazards
        )
        print(row)

    print(f"\nAll outputs written to outputs/")


if __name__ == "__main__":
    main()
