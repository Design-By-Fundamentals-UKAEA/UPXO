# UPXO

**UPXO (UKAEA Poly-XTAL Operations)** is an open-source Pythonic computational framework for generating, analysing, manipulating, meshing, visualising, and exporting representative polycrystalline grain structures for materials science. Although primarily developed for applciations pertaining multi-scale computational studies of nuclear structural materials, it can also solve a wide range of such problems in the Aerospace and Automobile sectors.

UPXO can enable you to create complex **2D and 3D poly-crystalline grain-structures** suitable for Finite Element (FE) simulations, microstructure characterisation, and data-driven materials research involving such computational domains.

Funding: This work has been funded by STEP, a major tehnology and infrastructure programme led by UK Industrial Fusion Solutions Ltd (UKIFS), which aims to deliver the UK's prototype fusion powerpoint and a path to the commercial visibility of fusion.

A dedicated wiki has been created to help users. Please find it [here](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki)

---

## Core capabilities

- **Generation of grain structures** - Voronoi type anbd Monte-Carlo simulation type.
- **Characterisation and Analysis** - morphology, texture, and topological.
- **Manipulation Tools** - feature removal, introduction, modification and replacement.
- **Representativeness Assessment** - morphology, texture, and topological.
- **FE Meshing** - grain boundary geometry conformant and non-conformant Finite Element meshes.
- **Data Interface** - import, export, and management of grain structure data.
- **Visualisation** - tools for plotting 2D and 3D microstructures.

---

## Microstructures Supported
UPXO can produce a broad range of synthetic grain structures, including:

- Equiaxed polycrystals generated from tessellation methods  
- Elongated and directionally structured grains  
- Hierarchical microstructures such as lath-based morphologies  
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
> Researchers can take advantage of the easy to use pipelines and templates to run existing or custom Pott's model Monte-Carlo simulation algorithms. The frameworks provide multiple entry points to study the grain growth kinetics, such as (a) Energetics (b) Ensemble propertie of space partitioning (statistical - morphological, topological and spatial)
---

## Contributors

- **Dr. Sunil Anandatheertha** - UK Atomic Energy Authority (UKAEA), Culham, Oxfordshire, OX14 3DB, UK
- **Dr. Vikram Phalke** - UK Atomic Energy Authority (UKAEA), Culham, Oxfordshire, OX14 3DB, UK
- **Dr. Chris Hardie** - UK Atomic Energy Authority (UKAEA), Culham, Oxfordshire, OX14 3DB, UK
- **Dr. Eralp Demir** - University of Oxford, Parks Road, Oxford, OX1 3PJ, UK

---

## License

UPXO is distributed under the **GNU General Public License v3.0 (GPL-3.0)** for open-source and academic use.

Companies, industrial users, and other organisations wishing to use UPXO in **commercial or proprietary applications** may obtain a separate commercial license.

For commercial licensing enquiries, please contact:
- **Dr. Sunil Anandatheertha** (Email: *vaasu.anandatheertha@ukaea.uk*) and
- **Dr. Chris Hardie** (Email: *chris.hardie@ukaea.uk*)

Additional licensing information is provided in `COMMERCIAL.md`.