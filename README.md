# Entropic analysis of solid-state electrolytes (SSEs)
**Currently, this repository is under construction.**

This repository (EA-SSEs) implements the path entropy analysis of SSEs, which quantify the diffusion ability in view of the diversity of the pathways.
It has several main functionalities:
1. Markov state model construction, and analysis of kinetics of lithium diffusion.
2. Analysis of entropy-driven strageties, including both path entropy and configurational entropy
3. Collective behavior of lithium - "softness" of lithium

## Table of Contents
- [Prerequisties](#pre)
- [Data](#data)
- [Tutorial](#tut)

<a name="pre"> </a>
## Prerequisties
To run the EA-SSEs, the python packages (env.yml) are needed. The recommended way to install those prerequisties is via [conda](https://conda.io/docs/index.html).

<a name="data"> </a>
## Data
All the data needed to run the EA-SSEs are directly avaliable on `data/`. Since the original `MD` trajectories have large size, the encoded traj in `hdf5` format are loaded.

<a name="tut"> </a>
## Tutorial
You can find some jupyter-notebooks of EA-SSEs useful on `tut/`.
