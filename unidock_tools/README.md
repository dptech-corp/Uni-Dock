# Introduction

Based on Uni-Dock, we have developed several user-friendly and enhanced applications in **Uni-Dock Tools**.
Main features are:

- Individual receptor and ligand preparation applications

- End-to-End Uni-Dock pipeline starting from common formats receptor and ligands

- Multi-Conformation Docking (mcdock): See [MCDock Introduction](./MCDOCK.md#introduction)

# Installation

## Dependency

- [Uni-Dock](../unidock/README.md#installation)
- Python >= 3.8
- MGLTools, if you want to use AD4 scoring function in Uni-Dock
- MCDock dependencies, see [MCDock Installation](./MCDOCK.md#installation)

## Install

Clone Uni-Dock repo, and under `unidock_tools` subdirectory run ```pip install .```

Or ```pip install git+https://github.com/dptech-corp/Uni-Dock.git#subdirectory=unidock_tools```

## Docker

 We provide docker image with all the dependencies installed. You can pull the docker image from the following link:

```docker pull dptechnology:unidocktools```

And run the docker container using the following command:

```docker run --gpus 0 -it -v $(pwd):/workpsace unidocktools:v1.0.0 cd /workspace &&  <Your command>```

# Applications

## 1. ProteinPrep

Prepare and Convert PDB-format receptor protein into PDBQT format

`unidocktools proteinprep -r <Input PDB File> -o <Output PDBQT File Path>`

## 2. LigandPrep

Prepare ligands to be used in Uni-Dock

`unidocktools ligandprep -l <Input SDF ligand files, use commas to seperate> -sd <output_dir> -bs <batch_size=1200>`

or write a list of ligand files in a text file and use the following command:

`unidocktools ligandprep -i <Txt File> -sd <output_dir>`

## 3. Uni-Dock Pipeline

End-to-End pipeline to run Uni-Dock with common-format receptor and ligands

`unidocktools unidock_pipeline -r <receptor file> -l <ligand files> -sd <output_dir> -cx <center_x> -cy <center_y> -cz <center_z>`

### Parameters

#### IO Parameters
- `-r, --receptor`: Path to the receptor file in PDBQT format.
- `-l, --ligands`: Path to the ligand file in SDF format. For multiple files, separate them by commas.
- `-i, --ligand_index`: A text file containing the path of ligand files in sdf format.
- `-sd, --savedir`: Save directory (default: 'unidock_results').

#### Pocket Parameters
- `-cx, --center_x`: X-coordinate of the docking box center.
- `-cy, --center_y`: Y-coordinate of the docking box center.
- `-cz, --center_z`: Z-coordinate of the docking box center.
- `-sx, --size_x`: Width of the docking box in the X direction (default: 22.5).
- `-sy, --size_y`: Width of the docking box in the Y direction (default: 22.5).
- `-sz, --size_z`: Width of the docking box in the Z direction (default: 22.5).

#### Docking Parameters
- `-sf, --scoring_function`: Scoring function (default: 'vina').
- `-ex, --exhaustiveness`: exhaustiveness (default: 128).
- `-ms, --max_step`: max_step (default: 20)
- `-nm, --num_modes`: Number of poses to output (default: 3).
- `-rs, --refine_step`: Refine step (default: 3).
- `-topn, --topn`: Top N pose results to keep (default: 100).

#### Others
- `-wd, --workdir`: Working directory (default: 'unidock_workdir').
- `-bs, --batch_size`: Batch size (default: 20).

## 4. Multi-Conformation Docking (mcdock)

See [MCDock Usage](./MCDOCK.md#usage)

# License

This project is licensed under the terms of Apache license 2.0. See [LICENSE](../LICENSE) for additional details.