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
MultiConfDock can be installed and run using Docker, which ensures that it operates in a consistent environment across different systems. Here are the steps to install and run MultiConfDock using Docker:

1. **Build the Docker image**: In the root directory of the project, run the following command to build the Docker image:

    ```shellscript
    docker build . -f contrainers/mcdock.dockerfile -t mcdock:v1.0.0
    ```

    This command builds a Docker image named `mcdock:v1.0.0` using the Dockerfile located at `contrainers/mcdock.dockerfile`.
    
2. **Pull from DockerHub**: You can also pull the docker image from docker hub using the following command:

    ```shellscript
    docker pull dpzhengh/mcdock:v1.0.0

    docker tag dpzhengh/mcdock:v1.0.0 mcdock:v1.0.0
    ```

3. **Run the Docker container**: After the image is built, you can run a Docker container using the following command:

    ```shellscript
    docker run --gpus 0 -dit --name mcdock_container mcdock:v1.0.0
    ```

    This command runs a Docker container named `mcdock_container` in the background, using the `mcdock:v1.0.0` image. The `--gpus 0` option enables GPU support.

4. **Attach to the Docker container**: You can attach to the running Docker container using the following command:

    ```shellscript
    docker attach mcdock_container
    ```

    This command opens an interactive shell in the `mcdock_container` Docker container.

Once you're inside the Docker container, you can navigate to the `examples/1G9V/` directory, run the `run.sh` script, and view the results in the `result/final_result/1G9V_multiconf.sdf` file. 

You can also navigate to the `examples/1G9V/` directory, run the `run_confgen.sh` script, and view the results in the `result/final_result/1G9V_ligand.sdf` file. 


## Usage
Once you have MultiConfDock installed and running in a Docker container, you can use it to perform multi-conformation ligand docking. Here's a basic example of how to use MultiConfDock:

1. **Navigate to the example directory**: In the Docker container, navigate to the example directory using the following command:

    ```bash
    cd /opt/mcdock/examples/1G9V/
    ```

2. **Run the example script**: In the example directory, there is a script named `run.sh` that performs docking using MultiConfDock. Run this script using the following command:

    ```bash
    bash run.sh
    # bash run_confgen.sh
    ```

    This script runs MultiConfDock with the provided receptor and ligand files, and saves the results in the `result/final_result/1G9V_multiconf.sdf` file.

3. **View the results**: After the script finishes running, you can view the results using the following command:

    ```bash
    cat result/final_result/1G9V_multiconf.sdf
    # cat result/final_result/1G9V_ligand.sdf
    ```

    This command prints the contents of the `result/final_result/1G9V_multiconf.sdf` file, which contains the docking results.

This is a basic example of how to use MultiConfDock. Depending on your specific needs, you may need to modify the `run.sh` script or provide different receptor and ligand files.

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
- `-ms_rd, --maxstep_rigid_docking`: maxstep used in rigid docking (default: 20)
- `-nm_rd, --num_modes_rigid_docking`: Number of modes used in rigid docking (default: 3).
- `-rs_rd, --refine_step_rigid_docking`: Refine step used in rigid docking (default: 3).
- `-topn_rd, --topn_rigid_docking`: Top N results used in rigid docking (default: 100).

### Local Refine Parameters

- `-sf_lr, --scoring_function_local_refine`: Scoring function used in local refine (default: 'vina').
- `-ex_lr, --exhaustiveness_local_refine`: exhaustiveness used in local refine (default: 32)
- `-ms_lr, --maxstep_local_refine`: maxstep used in local refine (default: 40)
- `-nm_lr, --num_modes_local_refine`: Number of modes used in local refine (default: 1).
- `-rs_lr, --refine_step_local_refine`: Refine step used in local refine (default: 5).
- `-topn_lr, --topn_local_refine`: Top N results used in local refine (default: 1).
  
These parameters allow you to control the behavior of MultiConfDock and customize it to suit your specific needs.