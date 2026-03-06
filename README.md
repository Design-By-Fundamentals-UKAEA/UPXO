# UPXO

**UPXO (UKAEA Poly-XTAL Operations)** is an open-source Python framework for generating, analysing, manipulating, meshing, visualising, and exporting representative polycrystalline grain structures for computational materials science.

The framework enables the creation of complex **2D and 3D microstructures** suitable for finite-element simulations, microstructure characterisation, and data-driven materials research.

---

## Architecture

UPXO is organised as a modular environment supporting the full workflow of synthetic microstructure modelling.

Core capabilities include:

- **Generation** — creation of grain structures using approaches such as Monte Carlo methods and Voronoi tessellations  
- **Characterisation and Analysis** — extraction of morphology, texture, and structural descriptors  
- **Manipulation Tools** — controlled introduction, removal, or modification of microstructural features  
- **Representativeness Assessment** — statistical evaluation of generated microstructures  
- **FE Meshing** — preparation of conformal and non-conformal meshes for simulation  
- **Data Interface** — import, export, and management of grain structure data  
- **Visualisation** — tools for inspecting 2D and 3D microstructures  
- **Structure-Specific Utilities** — specialised tools tailored to different microstructure classes

---

## Microstructures Supported
UPXO can produce a broad range of synthetic grain structures, including:

- Equiaxed polycrystals generated from tessellation methods  
- Elongated and directionally structured grains  
- Hierarchical microstructures such as lath-based morphologies  
- Multi-scale grain arrangements  
- Large three-dimensional polycrystals suitable for simulation studies

---

## Applications

UPXO is intended for research involving microstructure-based modelling of structural materials.

Typical applications include:

### Nuclear materials research
Synthetic microstructures can be generated for materials relevant to nuclear systems, enabling computational investigations of irradiation-induced degradation and microstructural evolution.

### Aerospace and automotive materials
The framework can produce non-equiaxed or gradient grain morphologies representative of manufacturing processes such as rolling, extrusion, forging, additive manufacturing, and welding.

### Data-driven materials modelling
Large ensembles of statistically representative microstructures can be generated and analysed, supporting machine-learning approaches and surrogate modelling.

---

## Contributors

- **Dr. Sunil Anandatheertha** — UK Atomic Energy Authority (UKAEA), United Kingdom
- **Dr. Vikram Phalke** — UK Atomic Energy Authority (UKAEA), United Kingdom
- **Dr. Chris Hardie** — UK Atomic Energy Authority (UKAEA), United Kingdom
- **Dr. Eralp Demir** — University of Oxford, United Kingdom

---

## License

UPXO is distributed under the **GNU General Public License v3.0 (GPL-3.0)** for open-source and academic use.

Companies, industrial users, and other organisations wishing to use UPXO in **commercial or proprietary applications** may obtain a separate commercial license.

For commercial licensing enquiries, please contact:

**Dr. Sunil Anandatheertha**
United Kingdom Atomic Energy Authority (UKAEA)  
Email: *vaasu.anandatheertha@ukaea.uk*

**Dr. Chris Hardie**
United Kingdom Atomic Energy Authority (UKAEA)  
Email: *chris.hardie@ukaea.uk*

Additional licensing information will be provided in `COMMERCIAL.md`.
