# Introduction
Comparing with the origin Uni-Dock software, here are two new features:
- ligands prepare  
  
  support origin SDF formats  
  
- gnina CNNscores
  
  support using gnina CNNscores to rescore docking posing which sampled by vina scoring function

# Installation

## 1. install UniDockTools 
UniDockTools has been uploaded to PyPi, and you can install it using the following command:

    pip install UniDockTools

## 2. install unidock
   
We recommend users to install unidock in a new conda virtual environment to avoid potential configuration conflict issues.

    conda create -n unidock -c https://conda.mlops.dp.tech:443/caic unidock
  
## 3. install gnina
If you also want to use gnina CNNscores to rescore docking pose, you should install gnina.
- binary   
install gnina by download binary file from [gnina's release website](https://github.com/gnina/gnina/releases)
- source code  
install gnina from source code according to [gnina installation document](https://github.com/gnina/gnina#installation)

# Usage   
By installing UniDockTools, you have obtained an executable file called **Unidock**, which you can use just like running **unidock** before.

## 1. input ligands with origin sdf format

    Unidock --receptor receptor.pdbqt --gpu_batch ligand1.sdf ligand2.sdf ...


## 2. use gnina CNNscores to rescore docking poses

    Unidock --receptor receptor.pdbqt --gpu_batch ligand1.sdf ligand2.sdf  --scoring gnina ...
