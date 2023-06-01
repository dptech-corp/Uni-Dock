# Uni-Dock

Uni-Dock is a GPU-accelerated molecular docking program developed by DP Technology.
It supports various scoring functions including vina, vinardo, and ad4.
Uni-Dock achieves more than 1000-fold speed-up with high-accuracy compared with the AutoDock Vina running in single CPU core.
The [paper](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145) has been accepted by JCTC (doi: 10.1021/acs.jctc.2c01145).

![Runtime performance of Uni-Dock on different GPUs in three modes](assets/gpu_speeds.png)

## Usage Guideline

We offer the software **for academic purposes only**. By downloading and using Uni-Dock, you are agreeing to the usage guideline under our GitHub repository.

Developed by [DP Technology](https://dp.tech/en), [Hermite®](https://dp.tech/en/product/hermite) is a new-generation drug computing design platform which integrates artificial intelligence, physical modeling and high-performance computing to provide a one-stop computing solution for preclinical drug research and development. It integrates the features of Uni-Dock, along with virtual screening workflow for an efficient drug discovery process.

For commercial usage and further cooperations, please contact us at bd@dp.tech .

## Online Access

We provides an [online Uni-Dock service](https://labs.dp.tech/projects/uni-dock-serving/).

## Installation

Uni-Dock supports NVIDIA GPUs on Linux platform.
CUDA runtime environment is required.
Please download the latest binary at the assets tab of [the Release page](https://github.com/dptech-corp/Uni-Dock/releases).
Executable `unidock` supports vina and vinardo scoring functions, and `unidock_ad4` supports ad4 scoring function.

## Examples

We have provided a target from DUD-E dataset for screening test. Python version `>=3.6` is recommended.
Please make sure that `unidock` is in your `PATH` environment variable.

```bash
cd example/screening_test
python run_dock.py
```

If you want to use search mode presets, specify the parameter `search_mode` in `config.json` and delete `nt` and `ns` in `config.json`.

## Bug Report

Please report bugs to [Issues](https://github.com/dptech-corp/Uni-Dock/issues) page.

## Ackowledgement

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

## FAQ

1. The GPU encounters out-of-memory error.
Uni-Dock estimates the number of ligands put into GPU memory in one pass based on the available GPU memory size. If it fails, please use `--max_gpu_memory` to limit the usage of GPU memory size by Uni-Dock.
2. I want to put all my ligands in `--gpu_batch`, but it exceeds the maximum command line length that linux can accept.
    - You can save your command in a shell script like `run.sh`, and run the command by `bash run.sh`.
    - You can save your ligands path in a file (separated by spaces) by `ls *.pdbqt > index.txt`, and use `--ligand_index index.txt` in place of `--gpu_batch`.
