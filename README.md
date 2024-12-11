# Entropy analysis of solid-state electrolytes (EA-SSEs)
**Some part of this repository is still under construction.**

This repository (EA-SSEs) implements the path entropy analysis of SSEs, which quantifies the diffusion ability in view of the diversity of the pathways.
It has several main functionalities:
1. Markov state model construction, and analysis of kinetics of lithium diffusion.
2. Analysis of entropy-driven strageties, including both path entropy and configurational entropy.
3. Collective behavior of lithium - "softness" of lithium.

## Table of Contents
- [Prerequisties](#pre)
- [Data](#data)
- [Citation](#cite)

<a name="pre"> </a>
## Prerequisties
To run the EA-SSEs, some python packages (env.txt) are needed. The recommended way to install those prerequisties is via [conda](https://conda.io/docs/index.html).

<a name="data"> </a>
## Data
All the data needed to run the EA-SSEs will be directly avaliable on `data/`. Since the original `MD` trajectories have large size (~several GB), the trajs are not loaded here.

<a name="cite"> </a>
## Citation
Feel free to use this repository; we'd appreciate a citation of our related work.

@misc{guan2024unlockingionmigrationsolidstate,
      title={Unlocking the Ion Migration in Solid-State Electrolytes via Path Entropy}, 
      author={Qiye Guan and Kaiyang Wang and Jingjie Yeo and Yongqing Cai},
      year={2024},
      eprint={2412.07115},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2412.07115}, 
}
