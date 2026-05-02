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
from scipy.interpolate import interp1d

from urm_wall import ARCHETYPES
from hazard_loads import (
    RETURN_PERIOD_WIND, RETURN_PERIOD_FLOOD,
    EF_LABELS, EF_MID_SPEEDS,
)
from fragility import run_all_fragility, annual_failure_probability, surge_from_wind

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
    # Explain line styles via proxy artists
    from matplotlib.lines import Line2D
    ax_right.add_artist(ax_right.legend(frameon=False, loc="upper left"))
    style_legend = ax_right.legend(
        handles=[Line2D([0], [0], color="k", ls="--", lw=1.5, alpha=0.5, label="Wind only"),
                 Line2D([0], [0], color="k", ls="-",  lw=2.0,             label="Wind + surge")],
        frameon=False, loc="center left", fontsize=8)
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
    from hazard_loads import RETURN_PERIOD_WIND, RETURN_PERIOD_FLOOD

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
        # Rates from Tippett et al. (2016) SPC tornado climatology for SE US
        # high-exposure region (roughly AL/MS/TN corridor). Values are per
        # grid cell (~1000 km²) per year; treat as site-scale approximation.
        r_t = results[key]["tornado"]
        ef_annual_rates = np.array([0.010, 0.005, 0.002, 0.0005, 0.0001, 0.00002])
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
    from matplotlib.colors import LogNorm
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
             "PEER PBEE integration: λ_f = Σ P(fail|IM) · Δλ(IM). "
             "Wind: ASCE 7-22 Risk Cat II hazard curve (10-yr to 1700-yr return periods). "
             "Flood: FEMA Zone AE depths (conditional on site being in floodplain). "
             "Tornado: Tippett et al. (2016) SE US rates. "
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
    of construction year (1890, 1910, 1930, 1950) — key argument for why aging
    plants face invisibly elevated risk.
    """
    from hazard_loads import wind_pressure_psf
    from limit_states import governing_dc
    from urm_wall import sample_walls, degradation_factor
    from fragility import V_HURRICANE, N_SAMPLES

    fig, ax = plt.subplots(figsize=(7, 4.5))

    years = [1895, 1910, 1925, 1940, 1950]
    cmap = plt.get_cmap("plasma")
    arch_base = ARCHETYPES["turbine_hall"].copy()

    for idx, yr in enumerate(years):
        arch = arch_base.copy()
        arch["year_built"] = yr
        walls = sample_walls(arch, N_SAMPLES, seed=10 + idx)

        p_net = wind_pressure_psf(V_HURRICANE)
        ph = arch["panel_height_ft_mean"]
        F = p_net * ph * arch["width_ft"]
        arm = np.full(len(V_HURRICANE), ph / 2.0)

        result = governing_dc(walls, F, p_net, arm)
        deg = degradation_factor(yr)
        color = cmap(idx / len(years))
        ax.plot(V_HURRICANE, result["p_fail"],
                color=color, lw=2.0,
                label=f"Built {yr} (deg. factor = {deg:.2f})")

    for p_ref in [0.10, 0.50]:
        ax.axhline(p_ref, color="#888", lw=0.7, ls=":")
        ax.text(202, p_ref + 0.01, f"{int(p_ref*100)}%", fontsize=8, color="#888")

    # Mark Cat 3 hurricane ~111 mph
    ax.axvline(111, color="gray", lw=0.9, ls="--")
    ax.text(113, 0.05, "Cat 3", fontsize=8, color="gray")

    ax.set_xlabel("3-second Gust Wind Speed (mph)")
    ax.set_ylabel("P(Wall Panel Failure)")
    ax.set_title("Effect of Building Age on Turbine Hall Hurricane Fragility")
    ax.set_xlim(60, 200)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.01, -0.03,
             "Same archetype geometry; degradation factor reduces effective material strength with age. "
             "A 1925-built turbine hall fails at ~35% probability under a Cat 3 hurricane; "
             "a 1950-built equivalent is below 10%.",
             fontsize=7.5, color="#555")
    save(fig, "degradation_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Running Monte Carlo fragility analysis...")
    results = run_all_fragility()

    print("Generating figures...")
    plot_hurricane_fragility(results)
    plot_tornado_fragility(results)
    plot_flood_fragility(results)
    plot_combined_fragility(results)
    afp = plot_risk_matrix(results)
    plot_degradation_sensitivity()

    print("\n── Summary: Annual failure probabilities (%) ──")
    cols = ["Hurricane", "Tornado", "100-yr Flood", "Combined Hurricane"]
    header = f"{'Archetype':<22}" + "".join(f"{c:<22}" for c in cols)
    print(header)
    for i, key in enumerate(ARCHETYPES):
        row = f"{key:<22}" + "".join(f"{afp[i, j]:<22.3f}" for j in range(4))
        print(row)

    print(f"\nAll outputs written to outputs/")


if __name__ == "__main__":
    main()
