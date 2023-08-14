# Introduction
To make Uni-Dock more user-friendly and compatible with more ligand input formats and scoring functions, we have introduced **UniDockTools**.    
Now, UniDockTools has two new functions:
- support SDF format
  
  UniDockTools has a built-in molecular preparation function, so users can use the original SDF files as input for ligand files, provided that the compound structures in the original SDF files are reasonable.  
  
- support gnina CNNscores 
  
  Gnina CNNscores is a scoring function known for its outstanding screening performance, UniDockTools offers a workflow for Uni-Dock to generate ligand conformations and subsequently re-score them using Gnina CNNscores.  

# Installation

## 1. install Uni-Dock and UniDockTools
UniDock and UniDockTools have been uploaded to the Anaconda repository. To install them, please use the following command:

    conda create -n unidock -c https://conda.mlops.dp.tech:443/caic unidock

If you have already installed Uni-Dock and want to install UniDockTools separately, you can use the command:

    pip install unidock_tools

## 2. install gnina
If you want to use gnina CNNscores to rescore docking poses, you should install gnina.
- binary   
install gnina by download binary file from [gnina's release website](https://github.com/gnina/gnina/releases)
- source code  
install gnina from source code according to [gnina installation document](https://github.com/gnina/gnina#installation)

# Usage   
By installing UniDockTools, you have obtained an executable file called **Unidock**, which you can use just like running **unidock**.

## 1. input ligands with origin sdf format

    Unidock --receptor receptor.pdbqt --gpu_batch ligand1.sdf ligand2.sdf --center_x 9 --center_y -5  --center_z -5 --size_x 20  --size_y 20 --size_z 20 --search_mode banlance --num_modes 9 --seed 123 --dir .


## 2. use gnina CNNscores to rescore docking poses

    Unidock --receptor receptor.pdbqt --gpu_batch ligand1.sdf ligand2.sdf  --scoring gnina --center_x 9 --center_y -5  --center_z -5 --size_x 20  --size_y 20 --size_z 20 --search_mode banlance --num_modes 9 --seed 123 --dir .
## 3. other usage

To lower users' learning cost, the other usage methods of **Unidock** remain consistent with the usage of **unidock**.


# License

This project is licensed under the terms of Apache license 2.0. See [LICENSE](./LICENSE) for additional details.
