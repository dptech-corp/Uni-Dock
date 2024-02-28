# Uni-Dock

<picture><source media="(prefers-color-scheme: dark)" srcset="./unidock/assets/logo-dark.svg"><source media="(prefers-color-scheme: light)" srcset="./unidock/assets/logo.svg"><img alt="Uni-Dock logo" src="./unidock/assets/logo.svg"></picture>

[![DeepModeling](https://img.shields.io/badge/DeepModeling-Incubating_Project-blue)](https://github.com/deepmodeling)

**Uni-Dock** is a GPU-accelerated molecular docking program developed by DP Technology.
It supports various scoring functions including vina, vinardo, and ad4. Uni-Dock achieves more than 1000-fold speed-up on V100 GPU with high-accuracy compared with the AutoDock Vina running in single CPU core.
The [paper](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145) has been accepted by JCTC (doi: 10.1021/acs.jctc.2c01145).

**Uni-Dock** joins the DeepModeling community, a community devoted of AI for science, as an incubating level project. [Learn more about DeepModeling](https://github.com/deepmodeling/community)

![Runtime performance of Uni-Dock on different GPUs in three modes](./unidock/assets/gpu_speeds.png)

Please check [`unidock` folder](./unidock/) for installing instructions, source codes, and usage.

**Uni-Dock Tools** is a Python package developed to handle the inputs and outputs of Uni-Dock.
It is committed to support more input formats and scoring functions. We hope it could be an easy-to-use virtual screening workflow for users with diversed backgrounds.

Please check [`unidock_tools` folder](./unidock_tools/) for installing instructions, source codes, and usage.

## Changelog

- 2023-08-21: Upload source codes of Uni-Dock.
- 2023-08-14: Add Uni-Dock Tools to support SDF format input for vina and vinardo scoring functions.

## Citation

If you used Uni-Dock in your work, please cite:

Yu, Y., Cai, C., Wang, J., Bo, Z., Zhu, Z., & Zheng, H. (2023).
Uni-Dock: GPU-Accelerated Docking Enables Ultralarge Virtual Screening.
Journal of Chemical Theory and Computation.
https://doi.org/10.1021/acs.jctc.2c01145
