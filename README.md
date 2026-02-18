# PL-FEM Vectoriel

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-v18.10-orange)
![scikit-fem](https://img.shields.io/badge/FEM-scikit--fem-purple)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![DOI](![DOI](https://img.shields.io/badge/DOI-will%20be%20assigned-lightgrey))

**Modular Python Framework for Fully Vectorial FEM Simulation and Dataset Generation  
of Polymer Photonic Lantern Multiplexers/Demultiplexers**

[ Paper](#citation) • [ Dataset ](#dataset) • [ Quick Start](#quick-start) • [ Documentation](#documentation) • [ Contributing](#contributing)

</div>

---

## Overview

**PL-FEM Vectoriel** is an open-source Python framework for large-scale parametric simulation and dataset generation of **polymer photonic lanterns (PLs)** targeting space-division multiplexed (SDM) intra-data-center optical networks.

The framework implements a **fully vectorial H-field finite-element method (FEM)** formulation using quadratic P2 Lagrange elements, combined with divergence-penalty spurious-mode suppression and analytic perfectly matched layers (PML). It generates physically validated datasets of PL configurations described by 86 scalar performance metrics, directly enabling machine learning surrogate modeling, multi-objective optimization, and inverse design.

### Why PL-FEM Vectoriel?

For high-index-contrast polymer/air photonic lanterns (Δn ≈ 0.53), **scalar approximations fundamentally fail**:

| Issue | Scalar Solver | PL-FEM Vectoriel |
|---|---|---|
| Hybrid HE/EH modes | ❌ Not resolved | ✅ Fully captured |
| PDL accuracy | ❌ Overestimated by 30–60% | ✅ Exact via (Px, Py) |
| Field discontinuities at polymer/air | ❌ Ignored | ✅ Naturally enforced |
| Spurious modes | ❌ Present | ✅ >98.5% eliminated |
| Multi-band dispersion (S/C/L/U) | ❌ Fixed index | ✅ Cauchy model (v18) |

---

## Key Features

- **Vectorial H-field FEM** with P2 quadratic elements (scikit-fem), divergence-penalty (αp=1), analytic PML
- **Eigenvalue accuracy** < 5×10⁻⁵ in relative effective index
- **12 experimentally validated MCF layouts** covering N = 2 to 19 cores
- **Multi-band support** — S, C, L, U telecom bands (1460–1675 nm) with Cauchy dispersion model for IP-Dip
- **Intelligent sampling** — Stratified Latin Hypercube Sampling + greedy diversity filtering (τq = 0.35)
- **Adaptive Delaunay meshing** with hierarchical refinement and LRU cache (500 MB)
- **2,000-sample dataset** — 86 metrics per design: IL, MDL, PDL, XT, confinement, effective indices, per-mode polarization states
- **Experimentally validated** against Dana et al. (2024) 7-core hexagonal PL to within 0.3 dB on insertion loss


---

## Architecture

```
pl-fem-vectoriel/
│
├── pl_fem/                         # Main package (17 modules, 7,647 lines)
│   │
│   ├── Core Simulation
│   │   ├── solver_fem.py           # TrueVectorialMaxwellSolver (H-field P2 FEM)
│   │   ├── solver_fem_modes.py     # neff, confinement, polarization analysis
│   │   ├── cmt.py                  # Coupled-mode theory propagation (RK45)
│   │   └── losses.py               # IL, MDL, PDL, XT evaluation
│   │
│   ├── Geometry & Meshing
│   │   ├── geometry.py             # PhotonicLanternGeometry, 3-section taper
│   │   ├── geometry_mcf.py         # 12 validated MCF core layouts
│   │   └── mesh.py                 # Adaptive Delaunay mesh + LRU cache
│   │
│   ├── Dataset Generation
│   │   ├── sampling.py             # SmartSampler: LHS + greedy diversity filter
│   │   ├── parametric_space.py     # 5 continuous + 4 discrete parameters
│   │   ├── dataset_generator.py    # Full pipeline orchestration
│   │   └── materials.py            # IP-Dip Cauchy dispersion, Silica, Air
│   │
│   └── config.py                   # Global constants, JSON/YAML config
│
├── examples/                       # Jupyter notebooks
│   ├── 01_single_simulation.ipynb
│   ├── 02_dataset_generation.ipynb
│   └── 03_ml_surrogate.ipynb
│
├── data/
│   └── sample_100configs.csv       # 100-sample subset for quick testing
│
├── tests/
│   ├── test_solver.py
│   ├── test_geometry.py
│   └── test_sampling.py
│
├── docs/
│   └── paper_preprint.pdf
│
├── requirements.txt
├── setup.py
└── main.py                         # CLI entry point
```

---

## Installation

### Requirements

- Python 3.9+
- pip or conda

### Install from source

```bash
git clone https://github.com/YOUR_USERNAME/pl-fem-vectoriel.git
cd pl-fem-vectoriel
pip install -r requirements.txt
pip install -e .
```

### Dependencies

```txt
scikit-fem>=6.0
scipy>=1.9
numpy>=1.23
matplotlib>=3.6
pandas>=1.5
meshio>=5.3
```

---

## Quick Start

### 1. Single simulation — 7-core hexagonal PL

```python
from pl_fem import PhotonicLanternGeometry, TrueVectorialMaxwellSolver

# Define geometry: 7-core hexagonal, IP-Dip/air, C-band
geom = PhotonicLanternGeometry(
    arrangement="hexagonal_1plus6_7",
    core_radius_um=1.5,
    pitch_um=8.0,
    n_core=1.535,
    n_clad=1.0,
    wavelength_nm=1550
)

# Run vectorial FEM eigenmode solver
solver = TrueVectorialMaxwellSolver(geom, n_modes=10)
modes = solver.solve()

# Print results
for i, mode in enumerate(modes):
    print(f"Mode {i}: neff={mode.n_eff:.6f}, "
          f"Γ={mode.confinement:.3f}, "
          f"PDL={mode.PDL_dB:.2f} dB, "
          f"pol={mode.polarization_state}")
```

### 2. Multi-band simulation with Cauchy dispersion

```python
from pl_fem import PhotonicLanternGeometry, TrueVectorialMaxwellSolver
from pl_fem.materials import IPDipCauchy

# Evaluate dispersive index across S, C, L, U bands
bands = {"S": 1490, "C": 1550, "L": 1600, "U": 1650}

for band, wavelength in bands.items():
    n_core = IPDipCauchy.n(wavelength)   # Cauchy model: A + B/λ² + C/λ⁴
    geom = PhotonicLanternGeometry(
        arrangement="hexagonal_1plus6_7",
        core_radius_um=1.5,
        pitch_um=8.0,
        n_core=n_core,
        n_clad=1.0,
        wavelength_nm=wavelength
    )
    solver = TrueVectorialMaxwellSolver(geom, n_modes=10)
    modes = solver.solve()
    print(f"Band {band} ({wavelength} nm): n_core={n_core:.4f}, "
          f"n_modes={len(modes)}, "
          f"PDL_mean={sum(m.PDL_dB for m in modes)/len(modes):.2f} dB")
```

### 3. Generate a parametric dataset

```python
from pl_fem import DatasetGenerator

generator = DatasetGenerator(
    n_samples=100,
    n_cores_list=[2, 3, 7, 12, 19],
    wavelengths_nm=[1490, 1550, 1600, 1650],
    output_path="my_dataset.csv",
    use_cauchy_dispersion=True,    # Cauchy model for IP-Dip (v18)
    quality_threshold=0.35,
    diversity_filter=True,
    n_jobs=4                       # Parallel CPU cores
)

dataset = generator.run()
print(f"Generated {len(dataset)} valid configurations")
print(f"Columns: {list(dataset.columns)}")
```

### 4. Run via CLI

```bash
# Generate 500 samples, 7-core only, C-band
python main.py --n_samples 500 --n_cores 7 --wavelength 1550 --output dataset_7core.csv

# Full multi-band dataset generation
python main.py --config configs/full_dataset.yaml
```

---

## Dataset
Dataset will be publicly released upon article acceptance

### Dataset Structure

The 86 parameters are organized across 7 hierarchical categories:

| Category | # Params | Key Parameters | Role in ML |
|---|---|---|---|
| Metadata | 4 | sample_id, success, solver_type, solver_time_s | Quality control |
| Geometry | 13 | n_cores, core_radius_um, pitch_um, arrangement | Primary features |
| Materials/Optics | 6 | V_number, n_core_lambda, wavelength_nm, NA | Complementary features |
| Global Modal | 10 | n_modes_found, n_eff_mean, confinement_mean | Intermediate targets |
| Polarization | 5 | PDL_mean_dB, PDL_max_dB, n_hybrid_modes | Vectorial-only targets |
| MUX/DEMUX Losses | 12 | IL_mux_dB, MDL_mux_dB, PDL_mux_dB, XT_mux_dB | Main ML targets |
| Per-mode (0–6) | 35 | n_eff_mode_k, conf_mode_k, PDL_mode_k, pol_mode_k | Fine-grained analysis |

### Observed Ranges

```
n_cores      :  2, 3, 6, 7, 12, 19
core_radius  :  0.5 – 3.0 µm
pitch        :  3.0 – 15.0 µm
V_number     :  2.0 – 12.0
wavelength   :  1490, 1550, 1590, 1610, 1650 nm
IL_mux       :  2.135 – 2.513 dB
MDL_mux      :  0.417 – 0.937 dB
PDL_mean     :  0.06 – 4.887 dB
n_modes      :  6 – 39
```

---

## Physical Formulation

### Vectorial H-field FEM

The transverse magnetic field **Ht = (Hx, Hy)** is governed by:

```
∇t × (εr⁻¹ ∇t × Ht) − k₀² Ht = (β²/k₀²) εr⁻¹ Ht
```

Discretized as the generalized eigenvalue problem:

```
[K + αp·D] {H} = β² [Mε⁻¹] {H}
```

where **K** is the stiffness matrix, **D** the divergence-penalty matrix (αp=1), and **Mε⁻¹** the weighted mass matrix, all assembled via scikit-fem with P2 quadratic elements.

### Cauchy Dispersion Model for IP-Dip (v18)

```
n(λ) = 1.5259 + 0.00860/λ² + 0.000210/λ⁴     [λ in µm]
```

Fitted to ellipsometric measurements (Schmid et al., 2019).  
Residual |Δn| < 3×10⁻⁴ across 1460–1675 nm.

### Performance Metrics

```python
# Insertion Loss
IL = -10 · log10(P_out / P_in)                          [dB]

# Mode-Dependent Loss
MDL = std({IL_m}, m = 1...M)                            [dB]

# Polarization-Dependent Loss (per mode)
PDL_m = 10 · log10(max(Px_m, Py_m) / min(Px_m, Py_m)) [dB]

# Polarization state classifier
η_pol = (||Hx||² - ||Hy||²) / (||Hx||² + ||Hy||²)
# |η_pol| > 0.9 → quasi-TE/TM  |  |η_pol| ≤ 0.9 → Hybrid HE/EH
```

---

## Experimental Validation

Comparison against Dana et al. (2024) — 7-core hexagonal PL fabricated by direct laser lithography:

| Method | IL (dB) | Δ vs Experiment |
|---|---|---|
| **FEM + CMT (this work)** | **2.383** | **+0.3 dB ✅** |
| 3D FDTD (reference) | 0.800 | −1.87 dB |
| Experiment (Dana 2024) | 2.670 | — |

> The FEM+CMT discrepancy with 3D FDTD is expected: 3D FDTD captures Fabry-Pérot reflections and facet coupling effects absent from the 2D cross-sectional FEM + adiabatic CMT model.

**Key result:** 100% of guided modes are hybrid (|η_pol| < 0.9) across all 2,000 configurations, confirming that scalar solvers are physically inadequate for IP-Dip/air PLs at Δn ≈ 0.53.

---

## Machine Learning Applications

The dataset is directly applicable to:

- **Forward surrogate modeling** — predict IL, MDL, PDL from geometric inputs  
  → Gradient-boosted regression: IL error < 0.08 dB, MDL error < 0.15 dB, ~10⁴× speedup

- **Multi-objective optimization** — Pareto front exploration (IL vs MDL vs PDL vs XT)  
  → Enables 10⁵–10⁶ function evaluations via surrogate

- **Inverse design** — map target performance to geometry parameters  
  → Conditional normalizing flows or tandem networks

### Quick ML example

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("pl_fem_dataset_v18.csv")
df = df[df["success"] == True]

features = ["n_cores", "core_radius_um", "pitch_um",
            "V_number", "delta_n", "wavelength_nm",
            "packing_efficiency", "pitch_ratio"]
target = "IL_mux_dB"

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(n_estimators=500, max_depth=5)
model.fit(X_train, y_train)

mae = abs(model.predict(X_test) - y_test).mean()
print(f"IL prediction MAE: {mae:.3f} dB")
```

---

## Supported Core Configurations

| N | Layout | Config type | Primary reference |
|---|---|---|---|
| 2 | Linear | linear_2 | Kokubun & Koshiba, 2009 |
| 3 | Triangular | triangular_3 | Fontaine et al., 2012 |
| 4 | Square 2×2 | square_2x2_4 | Hayashi et al., 2011 |
| 5 | Pentagonal ring | pentagonal_ring_5 | Jinno et al., 2020 |
| 6 | Hex ring | hexagonal_ring_6 | Zhu et al., 2011 |
| 6 | Pentagon + centre | pentagon_center_6 | Stern et al., 2021 |
| 7 | Hex 1+6 | hexagonal_1plus6_7 | Carpenter et al., 2015 |
| 8 | Hex 1+7 | heptagonal_center_8 | Hayashi et al., 2015 |
| 9 | Square 3×3 | square_3x3_9 | Igarashi et al., 2014 |
| 12 | Hex double ring | hex_double_ring_12 | Takenaga et al., 2014 |
| 13 | Hex 1+6+6 | hex_1plus6plus6_13 | Takenaga et al., 2011 |
| 19 | Hex 1+6+12 | hex_1plus6plus12_19 | Mizuno et al., 2016 |

---

## Limitations

Three known limitations are documented for transparency:

**1. Spurious-mode sensitivity at very high contrast (Δn > 0.6)**  
The nodal P2 discretization does not enforce ∇·H = 0 exactly. For Δn > 0.6, increasing αp to [2, 4] may be necessary. A Nédélec edge-element backend is planned for v19.

**2. Chromatic dispersion model**  
v18 implements a Cauchy approximation for IP-Dip (residual |Δn| < 3×10⁻⁴). Full Sellmeier integration for broadband SDWDM simulation is planned for v19.

**3. Deterministic geometry**  
Fabrication imperfections (core displacement disorder, surface roughness, index inhomogeneity) are not currently modeled. A stochastic perturbation layer is under development.

---

## Roadmap

- [x] **v18.10** — Vectorial H-field FEM, P2 elements, Cauchy dispersion, 2,000-sample dataset
- [ ] **v18.11** — Recalculate all 2,000 configs with Cauchy model across all 5 wavelengths
- [ ] **v19.0** — Nédélec edge-element backend, full IP-Dip Sellmeier equation
- [ ] **v19.1** — Stochastic fabrication perturbation layer (DLL process model)
- [ ] **v20.0** — GPU acceleration (CuPy sparse eigensolvers)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: description of your change"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request with a clear description

### Code Style

- Follow PEP 8
- Add docstrings to all public functions (NumPy style)
- Add a test in `tests/` for any new module
- Run `pytest tests/` before submitting

### Reporting Issues

Please use the GitHub Issues tab. Include:
- Python version and OS
- Minimal reproducible example
- Error message and traceback

---

@dataset{aguech2025dataset,
  title     = {PL-FEM Vectoriel Dataset v18.10: 2000 Photonic Lantern Configurations},
  author    = {Aguech, Khaoula and Ben Salem, Amine and Citrin, David S. and Menif, Mourad},
  publisher = {Zenodo},
  year      = {2025},
  note      = {Dataset will be publicly released upon article acceptance}
}

```



## License

This project is released under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [scikit-fem](https://github.com/kinnala/scikit-fem) — Gustafsson et al., finite element assembly library
- [SciPy](https://scipy.org) — ARPACK shift-invert eigensolvers, LHS sampling
- [Dana et al. (2024)](https://doi.org/10.1038/s41377-024-01464-6) — Experimental validation reference
- GRESCOM Research Group, Sup'Com, University of Carthage
- Georgia Tech–CNRS IRL, Georgia Institute of Technology

---

<div align="center">
Made with for the photonic engineering and ML communities  
<br>
<b>GRESCOM — Sup'Com, University of Carthage, Tunisia</b>
</div>
