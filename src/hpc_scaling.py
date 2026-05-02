"""
HPC computational scaling argument for the DOE HPC4EI-2026SP proposal.

Produces a concrete table showing why portfolio-scale URM fragility analysis
at high fidelity requires DOE supercomputing resources. Four fidelity levels
span 7 orders of magnitude in CPU-hours.

References
----------
Cundall & Strack (1979) — original DEM; particle count basis.
Lemos (2007) — URM discrete element modelling; CPU-hr estimates.
Lourenço (1996) — URM FEM strategies; 3D DOF counts.
McKenna et al. (2010) — OpenSeesPy; fiber-section runtime basis.
"""
import numpy as np

# Portfolio size basis (AMMTO manufacturing application):
# EIA-860 2024 shows 62 active fossil plants with pre-1950 generators (248–372 URM
# buildings) and 108 with pre-1960 steam equipment (432–648 URM buildings). Using
# 400 as a conservative mid-range for hurricane-exposed states.
# STEAM PLANT VARIANT (DOE CESER / GMI): same number; reframe as "thermal generation
# buildings" rather than "industrial manufacturing facilities".
PORTFOLIO_SIZE      = 400
STORM_SCENARIOS_FEM = 100     # HURDAT2-representative storm tracks for 3D FEM
STORM_SCENARIOS_DEM = 10      # DEM particle model (expensive — fewer scenarios)

# ── Fidelity level definitions ────────────────────────────────────────────────
# Each dict describes one level of model fidelity.
# cpu_hours_single: estimated wall-clock hours on a ~2.5 GHz single-thread core
#                   for ONE building under ONE representative loading scenario.
# feasibility: "Laptop", "Workstation", or "HPC required"

FIDELITY_LEVELS = [
    {
        "name":              "Analytical\n(current)",
        "short_name":        "Analytical",
        "dof_per_building":  3,           # 1 per limit-state mode × 3 modes
        "mc_samples":        12_000,
        "storm_scenarios":   1,           # one representative hazard table
        "cpu_hours_single":  0.008,       # ~30 sec (measured, this work)
        "reference":         "Measured (this work)",
        "feasibility":       "Laptop",
        "feasibility_color": "#d4efdf",   # green
    },
    {
        "name":              "2D Fiber\nSection FE",
        "short_name":        "2D Fiber FE",
        "dof_per_building":  100,         # 10 elements × 10 fibers per section
        "mc_samples":        200,
        "storm_scenarios":   1,
        "cpu_hours_single":  0.5,         # ~30 min (200 samples × 3 BCs, OpenSeesPy)
        "reference":         "McKenna et al. (2010); this work",
        "feasibility":       "Workstation",
        "feasibility_color": "#d4efdf",   # green
    },
    {
        "name":              "3D Solid FEM\n(single bldg)",
        "short_name":        "3D Solid FEM",
        "dof_per_building":  1_000_000,   # full building solid mesh (Lourenço 1996)
        "mc_samples":        1,           # deterministic per scenario
        "storm_scenarios":   STORM_SCENARIOS_FEM,
        "cpu_hours_single":  500,         # ~500 CPU-hrs per bldg per scenario
        "reference":         "Lourenço (1996); Rots (1997)",
        "feasibility":       "HPC required",
        "feasibility_color": "#fef9e7",   # yellow
    },
    {
        "name":              "Full DEM\nParticle Model",
        "short_name":        "DEM Particle",
        "dof_per_building":  100_000_000, # 10⁸ particles (Cundall & Strack 1979)
        "mc_samples":        1,
        "storm_scenarios":   STORM_SCENARIOS_DEM,
        "cpu_hours_single":  5_000,       # ~5,000 CPU-hrs per run (Lemos 2007)
        "reference":         "Cundall & Strack (1979); Lemos (2007)",
        "feasibility":       "HPC required",
        "feasibility_color": "#fdf2f2",   # red
    },
]


def compute_scaling_table() -> list:
    """
    Compute portfolio CPU-hour estimates for each fidelity level.
    Returns: list of dicts with display-ready values added.
    """
    table = []
    for level in FIDELITY_LEVELS:
        row = dict(level)
        scenarios = level["storm_scenarios"]
        row["cpu_hours_portfolio"] = (
            level["cpu_hours_single"] * PORTFOLIO_SIZE * scenarios
        )
        # Format DOF for display
        dof = level["dof_per_building"]
        if dof >= 1_000_000:
            row["dof_display"] = f"{dof/1e6:.0f}M"
        elif dof >= 1_000:
            row["dof_display"] = f"{dof/1e3:.0f}k"
        else:
            row["dof_display"] = str(dof)

        # Format CPU-hours
        ph = row["cpu_hours_portfolio"]
        if ph >= 1_000_000:
            row["portfolio_display"] = f"{ph/1e6:.1f}M"
        elif ph >= 1_000:
            row["portfolio_display"] = f"{ph/1e3:.0f}k"
        else:
            row["portfolio_display"] = f"{ph:.1f}"

        sh = level["cpu_hours_single"]
        if sh < 1.0:
            row["single_display"] = f"{sh*60:.0f} min"
        elif sh < 100:
            row["single_display"] = f"{sh:.0f} hr"
        else:
            row["single_display"] = f"{sh:.0f} hr"

        table.append(row)
    return table


def run_hpc_scaling() -> dict:
    """Top-level entry point called from run_analysis.py."""
    return {
        "table":            compute_scaling_table(),
        "portfolio_size":   PORTFOLIO_SIZE,
        "storm_scenarios_fem": STORM_SCENARIOS_FEM,
        "storm_scenarios_dem": STORM_SCENARIOS_DEM,
    }
