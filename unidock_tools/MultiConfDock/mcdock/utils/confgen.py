from .utils import makedirs
import multiprocessing as mlp
import subprocess as sp
from rdkit import Chem
import os
import rdkit
import logging
from pathlib import Path
from typing import List
from copy import deepcopy as dc
from rdkit.Geometry.rdGeometry import Point3D
import shutil

def confgen(
    mol:rdkit.Chem.rdchem.Mol,
    name:str="",
    max_num_confs_per_ligand:int=1000,
    min_rmsd:float=0.5,
    time_limit:float=300,
) -> List[rdkit.Chem.rdchem.Mol]:
    
    workdir = makedirs("confgen")
    if name == "": name = "ligand"
        
    smi = "{}\t{}".format(Chem.MolToSmiles(mol, isomericSmiles=True, 
        allHsExplicit=True), name)
    with open(f"{workdir}/{name}.smi", "w") as f:
        f.write(smi)
        
    cmd  = ["confgen"]
    cmd += ["-i", f"{workdir}/{name}.smi"]
    cmd += ["-o", f"{workdir}/{name}.sdf"]
    cmd += ["-t", str(mlp.cpu_count())]
    cmd += ["-n", f"{max_num_confs_per_ligand}"]
    cmd += ["-C", "LARGE_SET_DIVERSE"]
    cmd += ["-T", f"{time_limit}"]
    cmd += ["-v", "ERROR"]
    cmd += ["-p", "False"]
    cmd += ["-r", f"{min_rmsd}"]
    status = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    if status.returncode != 0:
        logging.error(f"ConfGen failed: {name}")
        logging.error(status.stderr.decode())
        
        logging.info("Try to generate conformers with openbabel instead")
        cmd = ["obabel"]
        cmd += [f"{workdir}/{name}.smi"]
        cmd += ["-O", f"{workdir}/{name}_obabel_3D.sdf"]
        cmd += ["-d"]
        cmd += ["-h"]
        cmd += ["--gen3D"]
        status1 = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        
        cmd = ["obabel"]
        cmd += [f"{workdir}/{name}_obabel_3D.sdf"]
        cmd += ["-O", f"{workdir}/{name}.sdf"]
        cmd += ["--conformer"]
        cmd += ["-nconf", f"{max_num_confs_per_ligand}"]
        cmd += ["--score", "rmsd"]
        cmd += ["--writeconformers"]
        status2 = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        
        if status1.returncode != 0 or status2.returncode != 0:
            logging.error(f"OpenBabel failed: {name}")
            logging.error(status1.stderr.decode())
            logging.error(status2.stderr.decode())
            return None
    
    mollist = [mol for mol in Chem.SDMolSupplier(f"{workdir}/{name}.sdf", removeHs=False) \
        if mol is not None]
    logging.info(f"ConfGen generated {len(mollist)} conformers: {name}")
        
    shutil.rmtree(workdir)
    
    return mollist

if __name__ == "__main__":
    pass