# UPXO

[![PyPI version](https://img.shields.io/pypi/v/upxo.svg)](https://pypi.org/project/upxo/)
[![Python versions](https://img.shields.io/pypi/pyversions/upxo.svg)](https://pypi.org/project/upxo/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://design-by-fundamentals-ukaea.github.io/UPXO/)

![UPXO logo and banner](https://raw.githubusercontent.com/Design-By-Fundamentals-UKAEA/UPXO/dev/wikidocs/assets/logo/upxo_logo_and_banner.png)

**UPXO (UKAEA Poly-XTAL Operations)** is an open-source Pythonic computational framework for generating, analysing, manipulating, meshing, visualising, and exporting representative polycrystalline grain structures for materials science. Although primarily developed for applications pertaining to multi-scale computational studies of nuclear structural materials, it can also solve a wide range of such problems in the Aerospace and Automobile sectors.

UPXO can enable you to create complex **2D and 3D poly-crystalline grain-structures** suitable for Finite Element (FE) simulations, microstructure characterisation, and data-driven materials research involving such computational domains.

Funding: This work has been funded by STEP, a major technology and infrastructure programme led by UK Industrial Fusion Solutions Ltd (UKIFS), which aims to deliver the UK's prototype fusion power plant and a path to the commercial viability of fusion.

A dedicated wiki has been created to help users. Please find it [here](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki)

---

## Installation

```bash
pip install upxo
```

Install with all optional extras (interactive plots, FE meshing, raster I/O, EBSD import):

```bash
pip install upxo[all]
```

Or select specific extras:

| Extra | Adds | Command |
|---|---|---|
| `viz` | Interactive plots (Plotly) | `pip install upxo[viz]` |
| `mesh` | FE meshing (pyvoro, tetgen) | `pip install upxo[mesh]` |
| `io` | Raster I/O (rasterio) | `pip install upxo[io]` |
| `ebsd` | EBSD data import (DefDAP) | `pip install upxo[ebsd]` |

Requires **Python >= 3.13**. See the [Getting Started wiki page](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Getting-started) for environment setup guides.

Canonical package metadata is in **`pyproject.toml`** (`pip install .` / `python -m build`). Optional `requirements.txt` mirrors **core** deps only; use extras (`upxo[all]`) for DefDAP, meshing backends, etc.

---

## Quick Start

```python
from upxo.ggrowth.mcgs import mcgs

# Run a 2D Monte Carlo grain growth simulation
pxt = mcgs(input_dashboard='path/to/input_dashboard.xls')
pxt.simulate()
pxt.detect_grains()

# Access the grain structure at the final saved time slice
gs = pxt.gs[pxt.m[-1]]
print(f"Number of grains: {gs.n}")
```

More end-to-end examples (2D/3D generation, hierarchical and twinned microstructures, meshing, visualisation) are in the [Workflows documentation](https://design-by-fundamentals-ukaea.github.io/UPXO/workflows.html).

---

## GUI applications (Windows)

After cloning this repository and installing UPXO into a Python ≥ 3.13 environment, you can start the material wizards by **double-clicking** in **`guiLaunchers/`**:

| File | Application |
|---|---|
| `guiLaunchers/Launch_FM_Steel_GUI.bat` | Ferritic–Martensitic steel 3D GUI |
| `guiLaunchers/Launch_Twinned_FCC_GUI.bat` | Twinned FCC 3D GUI (Cu / CuCrZr / OFHC-Cu) |

The scripts auto-detect a Python that can `import upxo` (or set `PYTHON_EXE` at the top of the `.bat`).

Alternatively, from Python:

```python
from upxo.pxtal.fm_steel_3d.gui_launcher import launch_gui
launch_gui()

from upxo.pxtal.twinned_simple_3d.gui_launcher import launch_gui
launch_gui()
```

Details: [GUI Applications wiki](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/GUI-Applications).

Demo notebooks are **not** in the pip wheel. Clone this repo and see `src/upxo/demos/` (a curated set is tracked in git; see the [Demo Notebooks wiki](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Demo-Notebooks)).

---

## Core capabilities

- **Generation of grain structures** — Voronoi tessellation and Monte Carlo simulation (Potts model); hierarchical lath microstructures; twinned FCC microstructures.
- **Characterisation and Analysis** — morphology, texture, and topological properties.
- **Manipulation Tools** — feature removal, introduction, modification, and replacement.
- **Representativeness Assessment** — morphological, textural, and topological assessment.
- **GUI Applications** — interactive wizards for FM Steel hierarchical generation and FCC twinned microstructure design.
- **FE Meshing** — grain boundary geometry conformant and non-conformant Finite Element meshes.
- **Data Interface** — import, export, and management of grain structure data; EBSD integration.
- **Visualisation** — tools for plotting 2D and 3D microstructures.

---

## Microstructures Supported

UPXO can produce a broad range of synthetic grain structures, including:

- Equiaxed polycrystals generated from tessellation methods
- Elongated and directionally structured grains
- **Hierarchical microstructures** — lath-based morphologies with packets, blocks, and sub-blocks (Ferritic-Martensitic steels)
- **Twinned microstructures** — Sigma-3 annealing and deformation twins in FCC materials (Cu, CuCrZr, OFHC-Cu)
- Texture-guided crystal orientation assignment
- EBSD-derived microstructure input and validation
- Multi-scale grain arrangements  

---

## Applications

UPXO is intended to cater to the needs of research involving microstructure-based modelling of structural materials.

Typical applications include:

### Nuclear materials research
> Synthetic microstructures can be generated for nuclear fusion relevant structural materials, enabling computational investigations of irradiation-induced degradation and microstructural evolution.

### Aerospace and automotive materials
> Non-equiaxed, gradient grain morphologies representative of manufacturing processes such as rolling, extrusion, forging, additive manufacturing, and welding may be produced in UPXO.

### Data-driven materials modelling
Large ensembles of statistically representative microstructures can be generated and analysed, supporting machine-learning approaches and surrogate modelling.

### Research in grain growth kinetics
> Researchers can take advantage of the easy to use pipelines and templates to run existing or custom Pott's model Monte-Carlo simulation algorithms. The frameworks provide multiple entry points to study the grain growth kinetics, such as (a) Energetics (b) Ensemble properties of space partitioning (statistical - morphological, topological and spatial)

---

## Contributors

- **Dr. Sunil Anandatheertha** - UK Atomic Energy Authority (UKAEA), Culham, Oxfordshire, OX14 3DB, UK
- **Dr. Vikram Phalke** - UK Atomic Energy Authority (UKAEA), Culham, Oxfordshire, OX14 3DB, UK
- **Dr. Chris Hardie** - UK Atomic Energy Authority (UKAEA), Culham, Oxfordshire, OX14 3DB, UK
- **Dr. Eralp Demir** - University of Oxford, Parks Road, Oxford, OX1 3PJ, UK

---

## Cite As

If you use UPXO in your research, please cite:

> Sunil Anandatheertha, Vikram Phalke, Eralp Demir, Chris Hardie, UKAEA Poly-XTAL operations (UPXO V1.0): An open-source Python package for generating, assessment and meshing of poly-crystalline grain structures, *SoftwareX*, Volume 34, 2026, 102736, ISSN 2352-7110, https://doi.org/10.1016/j.softx.2026.102736

```bibtex
@article{anandatheertha2026upxo1pnt1,
  title   = {UKAEA Poly-XTAL operations (UPXO V1.0): An open-source Python package for
             generating, assessment and meshing of poly-crystalline grain structures},
  author  = {Anandatheertha, Sunil and Phalke, Vikram and Demir, Eralp and Hardie, Chris},
  journal = {SoftwareX},
  volume  = {34},
  pages   = {102736},
  year    = {2026},
  issn    = {2352-7110},
  doi     = {10.1016/j.softx.2026.102736}
}
```

---

## Need Help Getting Started?

If you are from an academic institution and feel UPXO could be useful to your project or use case, and need personal help, don't hesitate to contact Dr. Sunil Anandatheertha — I will be happy to help you set up and offer help in using UPXO.

📧 vaasu.anandatheertha@ukaea.uk

---

## License

UPXO is distributed under the **GNU General Public License v3.0 (GPL-3.0)** for open-source and academic use.

Companies, industrial users, and other organisations wishing to use UPXO in **commercial or proprietary applications** may obtain a separate commercial license.

For commercial licensing enquiries, please contact:
- **Dr. Sunil Anandatheertha** (Email: *vaasu.anandatheertha@ukaea.uk*) and
- **Dr. Chris Hardie** (Email: *chris.hardie@ukaea.uk*)

Additional licensing information is provided in `COMMERCIAL.md`.