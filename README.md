# Uni-Dock
[![DeepModeling](https://img.shields.io/badge/DeepModeling-Incubating_Project-blue)](https://github.com/deepmodeling)
## Introduction

**Uni-Dock** is a GPU-accelerated molecular docking program developed by DP Technology.
It supports various scoring functions including vina, vinardo, and ad4. Uni-Dock achieves more than 1000-fold speed-up on V100 GPU with high-accuracy compared with the AutoDock Vina running in single CPU core.
The [paper](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145) has been accepted by JCTC (doi: 10.1021/acs.jctc.2c01145).

![Runtime performance of Uni-Dock on different GPUs in three modes](./unidock/assets/gpu_speeds.png)

Check this [subfolder](./unidock/) for instructions on installing Uni-Dock.

**Uni-Dock** joins the DeepModeling community, a community devoted of AI for science, as an incubating level project. To learn more about the DeepModeling community, see the [introduction of community](https://github.com/deepmodeling/community).
## UniDockTools

**UniDockTools** is a python package developed to handle the inputs and outputs of Uni-Dock.
In the future, UniDockTools will support more input formats and scoring functions. We hope it could be an easy-to-use virtual screening workflow for all users.

Check this [subfolder](./unidock_tools/) for more details.

## Changelog

- 2023-08-21: Upload source codes of Uni-Dock.
- 2023-08-14: Add unidock_tools to support SDF format input for vina and vinardo scoring functions.

## Citation

If you used Uni-Dock in your work, please cite:

Yu, Y., Cai, C., Wang, J., Bo, Z., Zhu, Z., & Zheng, H. (2023).
Uni-Dock: GPU-Accelerated Docking Enables Ultralarge Virtual Screening.
Journal of Chemical Theory and Computation.
https://doi.org/10.1021/acs.jctc.2c01145

Tang, S., Chen, R., Lin, M., Lin, Q., Zhu, Y., Ding, J., ... & Wu, J. (2022).
Accelerating autodock vina with gpus. Molecules, 27(9), 3041.
DOI 10.3390/molecules27093041

J. Eberhardt, D. Santos-Martins, A. F. Tillack, and S. Forli
AutoDock Vina 1.2.0: New Docking Methods, Expanded Force
Field, and Python Bindings, J. Chem. Inf. Model. (2021)
DOI 10.1021/acs.jcim.1c00203

O. Trott, A. J. Olson,
AutoDock Vina: improving the speed and accuracy of docking
with a new scoring function, efficient optimization and
multithreading, J. Comp. Chem. (2010)
DOI 10.1002/jcc.21334
