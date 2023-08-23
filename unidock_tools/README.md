# Introduction

To make Uni-Dock more user-friendly and compatible with more ligand input formats and scoring functions, we have introduced **UniDockTools**.
Now, UniDockTools has two new functions:

- support SDF format

  UniDockTools has a built-in molecular preparation function, so users can use the original SDF files as the input for ligand files, provided that the compound structures in the original SDF files are reasonable.

- support gnina CNNscores

  Gnina CNNscores is a scoring function known for its outstanding screening performance. UniDockTools offers a workflow for Uni-Dock to generate ligand conformations and subsequently re-score them using Gnina CNNscores.

# Installation

## 1. Install Uni-Dock and UniDockTools

To install UniDock, please follow [Uni-Dock installation docs](../unidock/README.md).

To install UniDockTools, please execute the following command in `Uni-Dock/unidock_tools` directory:

```python setup.py install```

## 2. Install MGLTools

If you want ro run Uni-Dock with receptor in PDB format, you need to install `mgltools`. Please use the command below:

```conda create -n mgltools mgltools -c bioconda```

## 3. Install gnina

If you want to use gnina CNNscores to rescore docking poses, you should install gnina.

- binary
install gnina by download binary file from [gnina's release page](https://github.com/gnina/gnina/releases)
- source code
install gnina from source code according to [gnina installation document](https://github.com/gnina/gnina#installation)

# Usage

By installing UniDockTools, you have obtained an executable file called **Unidock** (note the capitalized U), which you can use just like running **unidock**.

## Input ligands with origin sdf format

`Unidock --receptor receptor.pdbqt --gpu_batch ligand1.sdf ligand2.sdf --center_x 9 --center_y -5  --center_z -5 --size_x 20  --size_y 20 --size_z 20 --search_mode balance  --dir .`

## Use gnina CNNscores to rescore docking poses

`Unidock --receptor receptor.pdbqt --gpu_batch ligand1.sdf ligand2.sdf  --scoring gnina --center_x 9 --center_y -5  --center_z -5 --size_x 20  --size_y 20 --size_z 20 --search_mode balance  --dir .`

## Use ligands structure as bias

`Unidock --receptor receptor.pdbqt --gpu_batch ligand1.sdf ligand2.sdf  --scoring gnina --center_x 9 --center_y -5  --center_z -5 --size_x 20  --size_y 20 --size_z 20 --search_mode balance  --dir . --ligand_bias`

## Other usage

  To lower users' learning cost, the other usage methods of **Unidock** remain consistent with the usage of **unidock**.

# License

This project is licensed under the terms of Apache license 2.0. See [LICENSE](./LICENSE) for additional details.
