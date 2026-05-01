# HPC-Enabled Digital Twin for Disaster-Driven Energy Plant Outage Risk

Preliminary analytical results supporting the DOE HPC4EI-2026SP concept paper:
**"HPC-Enabled Digital Twin Framework for Predicting and Preventing Disaster-Driven Energy Plant Outages"**

PI: Seth Blumsack (EME, Penn State) · Co-PI: Rebecca Napolitano (AE, Penn State) · Co-PI: Jessica Menold (ICDS, Penn State)

---

## What this code does

This repository contains a simplified analytical fragility analysis of **unreinforced masonry (URM) thermal power plant building wall panels** under multi-hazard loading (hurricane wind, tornado, riverine/coastal flood, combined hurricane landfall). It uses Monte Carlo simulation (n=12,000 samples per archetype) to produce fragility curves and annual failure probability estimates for three representative building archetypes from the U.S. legacy thermal generation fleet.

**This is not the HPC simulation.** This analytical model demonstrates that non-negligible failure probabilities exist for aging URM plant buildings and quantifies the compound-hazard compounding effect. The high-fidelity multi-hazard simulation of complete plant buildings at the fidelity required for outage prediction — including 3D DEM/FEA of large masonry assemblies, soil-structure interaction, and coupled energy system response — requires DOE national laboratory HPC resources. The analysis here motivates the need for that simulation; it does not replace it.

---

## Building archetypes

| Archetype | Era | Wall thickness | Total height | Effective panel span | f_r (mean) |
|-----------|-----|---------------|--------------|---------------------|------------|
| Boiler house | 1895–1915 | 24 in (6-wythe) | 30 ft | 12 ft | 25 psi |
| Turbine hall | 1920–1940 | 16 in (4-wythe) | 45 ft | 20 ft | 45 psi |
| Powerhouse | 1910–1930 | 20 in (5-wythe) | 36 ft | 16 ft | 38 psi |

Material properties reflect early 20th century lime/portland-lime mortar construction. Degradation factor reduces effective material strength at 0.5%/year beyond 50 years of service (floored at 0.5×), following Napolitano et al. (2019).

---

## Hazard models

| Hazard | Intensity measure | Code reference |
|--------|------------------|----------------|
| Hurricane wind | 3-s gust wind speed, 60–200 mph | ASCE 7-22 Ch. 26-27 |
| Tornado | EF-scale (EF0–EF5); 1.5× internal pressure amplification | ASCE 7-22 App. CC |
| Flood | Inundation depth, 0–16 ft; hydrostatic + hydrodynamic | ASCE 7-22 Ch. 5 / FEMA P-55 |
| Combined hurricane | Wind + correlated storm surge (Irish et al. 2008) | — |

**Note on seismic:** Seismic hazard is included in the full proposal scope and will be addressed in the HPC simulation phase. It is not included in this MVP because it requires a different structural model (response spectrum / time-history analysis) beyond the scope of the analytical wall-panel framework.

---

## Limit states checked

All checks are per-panel, fully vectorized over Monte Carlo samples:

1. **Out-of-plane flexure** — simply supported over effective panel span; M_demand = q·h²/8; M_cap = f_r·S (TMS 402)
2. **Base sliding** — Mohr-Coulomb: V_cap = f_v_eff·A·144 + W·tan(φ)
3. **Overturning** — M_ot = F·arm vs. M_stab = W·t/2; arm from resultant of applied forces

Governing D/C ratio across the three checks determines failure (D/C ≥ 1.0).

---

## Key preliminary results

| Archetype | Hurricane AFP | Tornado AFP | 100-yr Flood AFP\* | Combined Hurricane AFP |
|-----------|--------------|-------------|-------------------|----------------------|
| Boiler house | **0.4%** | 0.1% | ~9% | **7.9%** |
| Turbine hall | **6.1%** | 1.1% | ~10% | **9.7%** |
| Powerhouse | **1.5%** | 0.4% | ~10% | **8.7%** |

\*Conditional on FEMA Zone AE location; not all thermal plants are in floodplains.

**The compound hazard finding is the key result.** Wind-only AFP for the boiler house is 0.4% — an apparently acceptable risk level. When storm surge accompanies the hurricane (correlated landfall scenario), AFP rises to 7.9% — a 20× underestimation of risk if compound loading is ignored. Current energy operator risk practice does not account for this compounding, and no computational framework exists to evaluate it at portfolio scale.

---

## Outputs

Six figures are generated in `outputs/`:

| File | Description |
|------|-------------|
| `fragility_wind_hurricane.png` | P(wall failure \| V_hurricane) for all archetypes |
| `fragility_wind_tornado.png` | P(wall failure \| EF category) bar chart |
| `fragility_flood.png` | P(wall failure \| flood depth) with FEMA return-period markers |
| `fragility_combined_hurricane.png` | Wind-only vs. combined landfall; compound hazard gap |
| `risk_matrix.png` | Annual failure probability matrix (log-scale colormap) |
| `degradation_sensitivity.png` | Turbine hall hurricane fragility shift by construction year |

---

## Usage

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python src/run_analysis.py
```

Figures saved to `outputs/`. Runtime: ~30 seconds on a laptop.

---

## Structure

```
src/
  urm_wall.py        — archetype definitions, material property sampler
  hazard_loads.py    — ASCE 7-22 wind/tornado/flood load equations
  limit_states.py    — vectorized D/C ratio calculations
  fragility.py       — Monte Carlo fragility curves, annual failure probability
  run_analysis.py    — main runner + all figure generation
outputs/             — generated PNG figures
requirements.txt     — numpy, scipy, matplotlib
```

---

## Limitations and path to HPC

This analytical model:
- Treats each wall panel as an isolated simply-supported element (no 3D building response, no pilaster-to-wall interaction, no floor/roof diaphragm stiffness)
- Uses idealized material property distributions (lognormal; no spatial correlation of degradation)
- Does not model progressive failure, crack propagation, or post-peak behavior
- Does not couple structural response to energy system downtime (that is the digital twin integration)

Overcoming these limitations at the fidelity required for actionable outage prediction — particularly for the large multi-bay masonry buildings typical of thermal generation plants — requires HPC-scale 3D DEM/FEA. The present analysis establishes the structural fragility problem and demonstrates non-negligible failure probabilities; it does not produce the full-building, multi-hazard simulation that would justify retrofit investment decisions.

---

## References

- ASCE 7-22: *Minimum Design Loads and Associated Criteria for Buildings and Other Structures*
- FEMA P-55: *Coastal Construction Manual*
- TMS 402-22: *Building Code Requirements for Masonry Structures*
- Irish, J.L., Resio, D.T., Ratcliff, J.L. (2008). The influence of storm size on hurricane surge. *Journal of Physical Oceanography*, 38(9), 2003–2013.
- Napolitano, R., Reinhart, C., Glock, C. (2019). Digital twins for historic masonry. *Journal of Structural Engineering*.
