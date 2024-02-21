from typing import List
import shutil
import logging
import multiprocessing as mlp
import subprocess as sp
from rdkit import Chem

from unidock_tools.utils import make_tmp_dir
from .base import ConfGeneratorBase


class OBabelConfGenerator(ConfGeneratorBase):
    def __init__(self):
        pass

    @staticmethod
    def check_env():
        return shutil.which("obabel") is not None

    def generate_conformation(self,
                              mol: Chem.Mol,
                              name: str = "",
                              max_num_confs_per_ligand: int = 1000,
                              *args, **kwargs) -> List[Chem.Mol]:
        workdir = make_tmp_dir("obabel")
        if not name:
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
            else:
                name = "ligand"
        smi = f"{Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)}\t{name}"
        with open(f"{workdir}/{name}.smi", "w") as f:
            f.write(smi)

        cmd = ["obabel"]
        cmd += [f"{workdir}/{name}.smi"]
        cmd += ["-O", f"{workdir}/{name}_obabel_3D.sdf"]
        cmd += ["-d"]
        cmd += ["-h"]
        cmd += ["--gen3D"]
        resp = sp.run(cmd, capture_output=True, encoding="utf-8")
        logging.debug(resp.stdout)
        if resp.returncode != 0:
            logging.error(resp.stderr)
            return []

        cmd = ["obabel"]
        cmd += [f"{workdir}/{name}_obabel_3D.sdf"]
        cmd += ["-O", f"{workdir}/{name}.sdf"]
        cmd += ["--conformer"]
        cmd += ["-nconf", str(max_num_confs_per_ligand)]
        cmd += ["--score", "rmsd"]
        cmd += ["--writeconformers"]
        resp = sp.run(cmd, capture_output=True, encoding="utf-8")
        logging.debug(resp.stdout)
        if resp.returncode != 0:
            logging.error(resp.stderr)
            return []

        mol_list = [mol for mol in Chem.SDMolSupplier(f"{workdir}/{name}.sdf",
                                                      removeHs=False) if mol is not None]
        logging.info(f"OBabel generated {len(mol_list)} conformers")
        if len(mol_list) > max_num_confs_per_ligand:
            mol_list = mol_list[:max_num_confs_per_ligand]

        shutil.rmtree(workdir, ignore_errors=True)
        return mol_list
