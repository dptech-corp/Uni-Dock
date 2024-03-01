# Multi-Conformation Docking

## Introduction
Traditional molecular docking methods typically involve positioning the ligand molecule within the binding pocket of the target protein and then sampling its rotations, translations, and torsions of rotatable bonds to identify the optimal binding pose. 
However, due to computational resource constraints and the vast search space resulting from 3D continuity, these methods often assess only a subset of possible conformational combinations. 
Consequently, this can lead to suboptimal docking results in some protein-ligand complex systems. 

***Multi-Conformation Docking (mcdock)*** addresses this limitation by advancing the conformational search process into the molecule preparation phase, thereby artificially ensuring a more comprehensive coverage of the search space in an attempt to enhance the reliability of molecular docking. 
It consists of three steps: optional ***Conformation Generation***, ***Rigid Docking***, and ***Local Refinement***. 

***Advantages of MultiConfDock***

1. **Comprehensive Conformation Generation**: `confgen` can rapidly generate a diverse array of low-energy conformations, ensuring that the search space encompasses as many ligand conformations as possible, increasing the likelihood of identifying suitable binding poses.

2. **Efficient Rigid Docking**: The method is capable of swiftly evaluating a vast number of ligand conformations for their relative positions and orientations within the protein's binding pocket, ensuring a thorough coverage of the search space for each ligand conformation.

3. **Refined Local Refinement**: MultiConfDock allows for minor local movements of the ligand to fine-tune the docking pose, ensuring that each binding pose is at a locally optimized structure.

This workflow is not only efficient but also powerful in predicting the optimal protein-ligand binding complexes.

## Description
***Conformation Generation***: This is the optional zeroth step in the MultiConfDock process. In this phase, MultiConfDock generates multiple conformations of the ligand. This is achieved using a conformation generation algorithm known as ConfGen. ConfGen employs `CDPKit` that can efficiently generate a large number of ligand conformations. This step is optional, and if the user already has the conformations of the ligand, he can skip this step.

***Rigid Docking***: The first step in the process is RigidDock. In this phase, MultiConfDock performs a rigid docking of each ligand conformation against the target protein. This means that the ligand and the protein are treated as rigid bodies, and only their relative positions and orientations change. This step is computationally efficient and allows MultiConfDock to quickly evaluate a large number of ligand conformations.

***Local Refinement***: After the RigidDock phase, MultiConfDock proceeds to the LocalRefine step. In this phase, the top scoring conformations from the RigidDock step are selected, and a local refinement is performed. This involves allowing small, local movements of the ligand and the protein to fine-tune the docking pose. This step is more computationally intensive, but it is applied only to a subset of the initial conformations, making the process efficient.

By combining these steps, MultiConfDock can efficiently identify the optimal protein-ligand binding complexes. This makes it a powerful tool for researchers in the field of drug design.

## Installation

### 1. Install from Source

To use mcdock, you need to install the following dependencies:

- [Uni-Dock Tools Dependencies](./README.md#dependency)
- [CDPKit](https://github.com/molinfo-vienna/CDPKit) or [OpenBabel](https://github.com/openbabel/openbabel)

Once you have installed the dependencies, install Uni-Dock Tools by

`pip install .`

Then, you can run mcdock by the following command:

`unidocktools mcdock <arguments>...`

### 2. Use by Docker
We also provide docker image to run MultiConfDock

1. **Pull from DockerHub**: You can pull the docker image from docker hub using the following command:

    ```shellscript
    docker pull dptechnology/unidock_tools
    ```

3. **Run mcdock by docker image**:

    ```shellscript
    docker run --gpus 0 -it -v $(pwd):/workspace dptechnology:unidock_tools cd /workspace && unidocktools mcdock <arguments>...
    ```


## Usage
Once you have MultiConfDock installed, you can use it to perform multi-conformation ligand docking. Here's a basic example of how to use MultiConfDock:

      unidocktools mcdock -r <pro.pdb> -l <lig1.sdf,lig2.sdf> -sd savedir -cx <center_x> -cy <center_y> -cz <center_z> --gen_conf
    


## Parameters
MultiConfDock is controlled via several command-line parameters. 
```shell
mcdock --help
```
Here's a brief overview:

### Required Arguments

- `-r, --receptor`: Path to the receptor file in PDBQT format.
- `-l, --ligands`: Path to the ligand file in SDF format. For multiple files, separate them by commas.
- `-i, --ligand_index`: A text file containing the path of ligand files in sdf format.

### ConfGen Arguments
- `-g, --gen_conf`: Whether to generate conformers for the ligands (default: False).
- `-n, --max_num_confs_per_ligand`: Maximum number of conformers to generate for each ligand (default: 1000).
- `-m, --min_rmsd`: Minimum RMSD for output conformer selection (default: 0.5000, must be >= 0, 0 disables RMSD checking).

### Docking Box Parameters

- `-cx, --center_x`: X-coordinate of the docking box center.
- `-cy, --center_y`: Y-coordinate of the docking box center.
- `-cz, --center_z`: Z-coordinate of the docking box center.
- `-sx, --size_x`: Width of the docking box in the X direction (default: 22.5).
- `-sy, --size_y`: Width of the docking box in the Y direction (default: 22.5).
- `-sz, --size_z`: Width of the docking box in the Z direction (default: 22.5).

### Directory

- `-wd, --workdir`: Working directory (default: 'MultiConfDock').
- `-sd, --savedir`: Save directory (default: 'MultiConfDock-Result').
- `-bs, --batch_size`: Batch size for mcdock (default: 20). 

### Rigid Docking Parameters

- `-sf_rd, --scoring_function_rigid_docking`: Scoring function used in rigid docking (default: 'vina').
- `-ex_rd, --exhaustiveness_rigid_docking`: exhaustiveness used in rigid docking (default: 128).
- `-ms_rd, --max_step_rigid_docking`: maxstep used in rigid docking (default: 20)
- `-nm_rd, --num_modes_rigid_docking`: Number of modes used in rigid docking (default: 3).
- `-rs_rd, --refine_step_rigid_docking`: Refine step used in rigid docking (default: 3).
- `-topn_rd, --topn_rigid_docking`: Top N results used in rigid docking (default: 100).

### Local Refine Parameters

- `-sf_lr, --scoring_function_local_refine`: Scoring function used in local refine (default: 'vina').
- `-ex_lr, --exhaustiveness_local_refine`: exhaustiveness used in local refine (default: 32)
- `-ms_lr, --max_step_local_refine`: maxstep used in local refine (default: 40)
- `-nm_lr, --num_modes_local_refine`: Number of modes used in local refine (default: 1).
- `-rs_lr, --refine_step_local_refine`: Refine step used in local refine (default: 5).
- `-topn_lr, --topn_local_refine`: Top N results used in local refine (default: 1).
  
These parameters allow you to control the behavior of MultiConfDock and customize it to suit your specific needs.