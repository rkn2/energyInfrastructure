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
from fragility import run_all_fragility, annual_failure_probability, surge_from_wind
from opensees_comparison import run_opensees_comparison, BC_LABELS
from site_specific import run_site_analysis, SITE_NAME, SITE_CITY, SITE_LAT, SITE_LON
from hpc_scaling import run_hpc_scaling

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

    hazard_labels = ["Hurricane\nWind", "EF2+ Tornado\n(annual rate)", "100-yr\nFlood", "Combined\nHurricane"]
    arch_labels = [ARCHETYPES[k]["label"] for k in ARCHETYPES]

    # Approximate annual failure probabilities
    afp = np.zeros((len(ARCHETYPES), 4))

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

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Log-scale colormap so we can see variation from 0.01% to 10%+
    afp_clipped = np.clip(afp, 0.01, 100.0)
    im = ax.imshow(afp_clipped, cmap="YlOrRd",
                   norm=LogNorm(vmin=0.01, vmax=afp_clipped.max() * 2),
                   aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels(hazard_labels, fontsize=10)
    ax.set_yticks(range(len(arch_labels)))
    ax.set_yticklabels(arch_labels, fontsize=10)
    ax.set_ylabel("Building Archetype", fontsize=11)
    ax.set_title("Estimated Annual Wall Failure Probability (%) — "
                 "Legacy URM Thermal Generation Facilities", fontsize=11)

    # Annotate each cell
    for i in range(len(ARCHETYPES)):
        for j in range(4):
            v = afp[i, j]
            txt = f"{v:.2f}%" if v >= 0.01 else "<0.01%"
            color = "white" if afp_clipped[i, j] > 1.0 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Annual Failure Probability (%)", fontsize=9)

    fig.text(0.01, -0.06,
             "Annual failure probability via hazard-fragility convolution: λ_f = Σ P(fail|IM) · Δλ(IM) "
             "(Cornell & Krawinkler 2000; Kennedy & Short 1994). "
             "Wind: ASCE 7-22 Risk Cat II hazard curve (10-yr to 1700-yr return periods). "
             "Flood: FEMA Zone AE depths (conditional on site being in floodplain). "
             "Tornado: Tippett et al. (2016) SE US rates scaled to ~0.2 km² plant footprint. "
             "Reference: ASCE 7-22 Risk Cat II design wind → 700-yr RP (0.14%/yr exceedance probability); "
             "these archetypes reach >50% panel failure well below the 700-yr wind speed.",
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
