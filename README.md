*This repo is under development and subject to change.*

# Introduction

[AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina) is one of the fastest and most widely used open-source docking engines. It is a turnkey computational docking program that is based on a simple scoring function and rapid gradient-optimization conformational search.

**Uni-Dock**, developed by DP Technology, carries out extreme performance optimization based on GPU acceleration, increasing AutoDock Vina's computing speed by more than **1000 times** on one Nvidia V100 32G GPU compared with one CPU core.

# Installation

Uni-Dock requires an NVIDIA GPU. It works best on V100 32G machines (no matter how many CPU cores there are), and we recommend running on it.

## Docker Image

We recommend that you use Docker to run the Uni-Dock, with all the required environments and dependencies configured in Docker image.

1. Install nvidia-container (a GPU supported version of docker): Refer to the [installation tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) provided by nvidia
2. Pull image of Uni-Dock from DP image registry

```bash
docker pull dp-harbor-registry.cn-zhangjiakou.cr.aliyuncs.com/dplc/vina_gpu:latest
```

## Binary

Refer to the releases page of this repo to download binary of Uni-Dock: <https://github.com/dptech-corp/Uni-Dock/releases>

## Build from source

1. Install dependencies

- Boost 1.77.0

```bash
mkdir /opt/packages
cd /opt/packages
wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz
tar -xzvf boost_1_77_0.tar.gz
rm boost_1_77_0.tar.gz
cd boost_1_77_0/
./bootstrap.sh
./b2
./b2 install --prefix=/opt/lib/packages/boost1_77
export LD_LIBRARY_PATH=/opt/lib/packages/boost1_77/lib/:$LD_LIBRARY_PATH
```

Alternatively, install from a package management system:

```bash
sudo apt install libboost-system-dev libboost-thread-dev libboost-serialization-dev libboost-filesystem-dev libboost-program-options-dev
```

- CUDA toolkit

Please refer to the [installation tutorial](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) provided by nvidia.
2. Clone GitHub repo or retrieve source code from release page
3. Build Uni-Dock

```bash
cd ./build/linux/release
make clean
make -j 4
```

Or build with CMake:

```bash
cmake -B build
cmake --build build -j4
```

## Build from source (DCU)

1. Install dependencies (same as CUDA)

Boost 1.77.0

2. Clone GitHub repo or retrieve source code from release page (branch: dcu_compile)
3. Build Uni-Dock

```bash
cd ./build/dcu/release
make clean
make -j 4
```

# Usage

## Example

To launch a Uni-Dock job, the most important parameters are as follows:

- `--receptor`: filepath of the receptor (PDBQT)
- `--gpu_batch`: filepath of the ligands to dock with GPU (PDBQT), enter multiple at a time, separated by spaces (" ")
- `--search_mode`: computational complexity, choice in [*`fast`*, *`balance`*, and *`detail`*].

***Advanced options***
`--search_mode` is the recommended setting of `--exhaustiveness` and `--max_step`, with three combinations called `fast`, `balance`, and `detail`.

- `fast` mode: `--exhaustiveness 128` & `--max_step 20`
- `balance` mode: `--exhaustiveness 384` & `--max_step 40`
- `detail` mode: `--exhaustiveness 512` & `--max_step 40`

The larger `--exhaustiveness` and `--max_step`, the higher the computational complexity, the higher the accuracy, but the larger the computational cost.

```bash
unidock --receptor <receptor.pdbqt> \
     --gpu_batch <lig1.pdbqt> <lig2.pdbqt> ... <ligN.pdbqt> \
     --search_mode balance \
     --scoring vina \
     --center_x <center_x> \
     --center_y <center_y> \
     --center_z <center_z> \
     --size_x <size_x> \
     --size_y <size_y> \
     --size_z <size_z> \
     --num_modes 1 \
     --dir <save dir>
```

## Parameters

```shell
>> unidock --help

Input:
  --receptor arg             rigid part of the receptor (PDBQT)
  --flex arg                 flexible side chains, if any (PDBQT)
  --ligand arg               ligand (PDBQT)
  --ligand_index arg         file containing paths to ligands
  --batch arg                batch ligand (PDBQT)
  --gpu_batch arg            gpu batch ligand (PDBQT)
  --scoring arg (=vina)      scoring function (ad4, vina or vinardo)

Search space (required):
  --maps arg                 affinity maps for the autodock4.2 (ad4) or vina
                             scoring function
  --center_x arg             X coordinate of the center (Angstrom)
  --center_y arg             Y coordinate of the center (Angstrom)
  --center_z arg             Z coordinate of the center (Angstrom)
  --size_x arg               size in the X dimension (Angstrom)
  --size_y arg               size in the Y dimension (Angstrom)
  --size_z arg               size in the Z dimension (Angstrom)
  --autobox                  set maps dimensions based on input ligand(s) (for
                             --score_only and --local_only)

Output (optional):
  --out arg                  output models (PDBQT), the default is chosen based
                             on the ligand file name
  --dir arg                  output directory for batch mode
  --write_maps arg           output filename (directory + prefix name) for
                             maps. Option --force_even_voxels may be needed to
                             comply with .map format

Misc (optional):
  --cpu arg (=0)             the number of CPUs to use (the default is to try
                             to detect the number of CPUs or, failing that, use
                             1)
  --seed arg (=0)            explicit random seed
  --exhaustiveness arg (=8)  exhaustiveness of the global search (roughly
                             proportional to time): 1+
  --max_evals arg (=0)       number of evaluations in each MC run (if zero,
                             which is the default, the number of MC steps is
                             based on heuristics)
  --num_modes arg (=9)       maximum number of binding modes to generate
  --min_rmsd arg (=1)        minimum RMSD between output poses
  --energy_range arg (=3)    maximum energy difference between the best binding
                             mode and the worst one displayed (kcal/mol)
  --spacing arg (=0.375)     grid spacing (Angstrom)
  --verbosity arg (=1)       verbosity (0=no output, 1=normal, 2=verbose)
  --max_step arg (=0)        maximum number of steps in each MC run (if zero,
                             which is the default, the number of MC steps is
                             based on heuristics)
  --max_gpu_memory arg (=0)  maximum gpu memory to use (default=0, use all
                             available GPU memory to optain maximum batch size)
  --search_mode arg          search mode of unidock (fast, balance, detail), using
                             recommended settings of exhaustiveness and search
                             steps; the higher the computational complexity,
                             the higher the accuracy, but the larger the
                             computational cost

Configuration file (optional):
  --config arg               the above options can be put here

Information (optional):
  --help                     display usage summary
  --help_advanced            display usage summary with advanced options
  --version                  display program version
```

# Troubleshooting

1. Some bugs are triggered when I run Uni-Dock in Docker, but no such problems when running directly on the server.
    - We did find that when running Uni-Dock in docker, there will be additional GPU memory usage. When you use docker to run Uni-Dock on a machine with V100 32G, please use `--max_gpu_memory 27000` to limit the usage of GPU memory size by Uni-Dock.
2. I want to put all my ligands in `--gpu_batch`, but it exceeds the maximum command line length that linux can accept.
    - You can save your command in a shell script like run.sh, and run the command by `bash run.sh`.
    - You can save your ligands path in a file (separated by spaces) by `ls . | tee index.txt`, and use `--ligand_index <ligands path file>` in place of `--gpu_batch`.
