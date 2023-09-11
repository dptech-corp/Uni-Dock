from typing import List, Union
from pathlib import Path
import os
import shutil
import subprocess
from io import BytesIO
from functools import partial
import multiprocessing
from rdkit import Chem
from rdkit.Chem import AllChem
from unidock_tools.ligand_prepare.topology_builder import TopologyBuilder


class LigandPrepareRunner:
    def __init__(self, ligand_files:List[str], workdir:str='./prepared_ligands', standardize:bool=False) -> None:
        self.ligand_files = ligand_files
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self.standardize = standardize

    @staticmethod
    def set_properties(mol:Chem.rdchem.Mol, props_dict:dict):
        for key, value in props_dict.items():
            if isinstance(value, int):
                mol.SetIntProp(key, value)
            elif isinstance(value, float):
                mol.SetDoubleProp(key, value)
            elif isinstance(value, str):
                mol.SetProp(key, value)

    @staticmethod
    def check_no_3d(ligand_file:str) -> bool:
        mol = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=False)[0]
        return all([c == 0 for c in mol.GetConformer().GetPositions()[:, 2]])

    @staticmethod
    def add_hydrogen(ligand_file:str, output_file:str, use_obabel:bool=False):
        mol = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=True)[0]
        props_dict = mol.GetPropsAsDict()

        if use_obabel and shutil.which("obabel"):
            mol_block = Chem.MolToMolBlock(mol, kekulize=True)
            mol_str = subprocess.check_output(["obabel", "-imol", "-osdf", "-h", "--gen3d"],
                                            text=True, input=mol_block, stderr=subprocess.DEVNULL)
            bstr = BytesIO(bytes(mol_str, encoding='utf-8'))
            addH_mol = next(Chem.ForwardSDMolSupplier(bstr, removeHs=False, sanitize=True))
            __class__.set_properties(mol=addH_mol, props_dict=props_dict)
        else:
            addH_mol = Chem.AddHs(mol, addCoords=True)
        with Chem.SDWriter(output_file) as writer:
            writer.write(addH_mol)
    
    @staticmethod
    def gen_3d(ligand_file:str, out_file:str):
        try:
            mol = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=True)[0]
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            with Chem.SDWriter(out_file) as writer:
                writer.write(mol)
        except:
            print(f"ligand {os.path.splitext(os.path.basename(ligand_file))[0]} gen 3d failed")

    @staticmethod
    def prepare_one_ligand(ligand_file:str, workdir:str, standardize:bool=False) -> Union[str, None]:
        filename = os.path.splitext(os.path.basename(ligand_file))[0]
        try:
            out_path = os.path.join(workdir, filename + ".sdf")
            tmp_file = os.path.join(workdir, filename + "_tmp.sdf")
            if standardize:
                if __class__.check_no_3d(ligand_file):
                    __class__.gen_3d(ligand_file, tmp_file)
                    ligand_file = tmp_file
                __class__.add_hydrogen(ligand_file, tmp_file)
                ligand_file = tmp_file
            topo=TopologyBuilder(ligand_file)
            topo.build_molecular_graph()
            topo.write_torsion_tree_sdf_file(out_path)
            print(f"ligand {filename} preperation successful")
            Path(tmp_file).unlink(missing_ok=True)
            return out_path
        except Exception as e:
            print(f"ligand {filename} preperation failed: {str(e)}")
            return None

    def prepare_ligands(self):
        cpu_count = os.cpu_count()
        if not cpu_count:
            cpu_count = 1
        with multiprocessing.Pool(processes=max(1, min(len(self.ligand_files), int(cpu_count//1.5))), maxtasksperchild=10) as pool:
            result_files = pool.map(partial(__class__.prepare_one_ligand, 
                workdir=self.workdir, standardize=self.standardize), self.ligand_files, chunksize=100)
        result_files = [f for f in result_files if f]

        ligands_num = len(self.ligand_files)
        ligands_prepared_num = len(result_files)
        print(f"Prepare ligands, finish {ligands_prepared_num} / {ligands_num}")

        return result_files
