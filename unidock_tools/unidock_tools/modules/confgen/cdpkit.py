from typing import List
import shutil
import logging
import multiprocessing as mlp
import subprocess as sp
from rdkit import Chem

from unidock_tools.utils import make_tmp_dir
from .base import ConfGeneratorBase


class CDPKitConfGenerator(ConfGeneratorBase):
    def __init__(self):
        pass

    @staticmethod
    def check_env():
        return shutil.which("confgen") is not None

    def generate_conformation(self,
                              mol: Chem.Mol,
                              name: str = "",
                              max_num_confs_per_ligand: int = 1000,
                              min_rmsd: float = 0.5,
                              time_limit: float = 300,
                              *args, **kwargs) -> List[Chem.Mol]:
        workdir = make_tmp_dir("confgen")
        if not name:
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
            else:
                name = "ligand"
        smi = f"{Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)}\t{name}"
        with open(f"{workdir}/{name}.smi", "w") as f:
            f.write(smi)

        cmd = ["confgen"]
        cmd += ["-i", f"{workdir}/{name}.smi"]
        cmd += ["-o", f"{workdir}/{name}.sdf"]
        cmd += ["-t", str(mlp.cpu_count())]
        cmd += ["-n", str(max_num_confs_per_ligand)]
        cmd += ["-C", "LARGE_SET_DIVERSE"]
        cmd += ["-T", str(time_limit)]
        cmd += ["-v", "ERROR"]
        cmd += ["--progress", "False"]
        cmd += ["-r", str(min_rmsd)]
        resp = sp.run(cmd, capture_output=True, encoding="utf-8")
        logging.debug(resp.stdout)
        if resp.returncode != 0:
            logging.error(resp.stderr)
            return []

        mol_list = [mol for mol in Chem.SDMolSupplier(f"{workdir}/{name}.sdf", removeHs=False) \
                    if mol is not None]
        logging.info(f"ConfGen generated {len(mol_list)} conformers")

        shutil.rmtree(workdir, ignore_errors=True)

        return mol_list
