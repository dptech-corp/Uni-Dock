import os
from shutil import rmtree
from typing import List, Tuple, Optional

import numpy as np

from unidock_tools.modules.protein_prep.receptor_standardization.receptor_pdb_reader import ReceptorPDBReader
from unidock_tools.modules.protein_prep.receptor_topology.docking_grids_generator import DockingGridsGenerator

class ReceptorPreprocessorRunner(object):
    def __init__(self,
                 protein_pdb_file_name,
                 kept_ligand_resname_list=None,
                 prepared_hydrogen=True,
                 preserve_original_resname=True,
                 target_center=(0.0, 0.0, 0.0),
                 box_size=(22.5, 22.5, 22.5),
                 covalent_residue_atom_info_list=None,
                 generate_ad4_grids=False,
                 working_dir_name='.'):

        self.protein_pdb_file_name = protein_pdb_file_name
        self.kept_ligand_resname_list = kept_ligand_resname_list
        self.prepared_hydrogen = prepared_hydrogen
        self.preserve_original_resname = preserve_original_resname

        self.target_center = target_center
        self.box_size = box_size
        self.covalent_residue_atom_info_list = covalent_residue_atom_info_list
        self.generate_ad4_grids = generate_ad4_grids

        ####################################################################################################################################
        ## Prepare working directories
        self.root_working_dir_name = os.path.abspath(working_dir_name)
        self.receptor_reader_working_dir_name = os.path.join(self.root_working_dir_name, 'receptor_reader')
        self.receptor_grids_working_dir_name = os.path.join(self.root_working_dir_name, 'receptor_grids')

        if os.path.isdir(self.receptor_reader_working_dir_name):
            rmtree(self.receptor_reader_working_dir_name, ignore_errors=True)
            os.mkdir(self.receptor_reader_working_dir_name)
        else:
            os.mkdir(self.receptor_reader_working_dir_name)

        if os.path.isdir(self.receptor_grids_working_dir_name):
            rmtree(self.receptor_grids_working_dir_name, ignore_errors=True)
            os.mkdir(self.receptor_grids_working_dir_name)
        else:
            os.mkdir(self.receptor_grids_working_dir_name)
        ####################################################################################################################################

    def run(self):
        ####################################################################################################################################
        ## Run receptor pdb reader to clean protein information
        receptor_pdb_reader = ReceptorPDBReader(self.protein_pdb_file_name,
                                                kept_ligand_resname_list=self.kept_ligand_resname_list,
                                                prepared_hydrogen=self.prepared_hydrogen,
                                                preserve_original_resname=self.preserve_original_resname,
                                                working_dir_name=self.receptor_reader_working_dir_name)

        self.receptor_cleaned_pdb_file_name = receptor_pdb_reader.run_receptor_system_cleaning()
        ####################################################################################################################################

        ####################################################################################################################################
        ## Run autogrid runner to generate protein PDBQT files and AD4 grids if necessary
        box_size_array = np.array(self.box_size)
        num_grid_points_array = box_size_array / 0.375
        num_grid_points_array = num_grid_points_array.astype(np.int32)
        self.num_grid_points = tuple(num_grid_points_array)
        self.grid_spacing = (0.375, 0.375, 0.375)

        docking_grids_generator = DockingGridsGenerator(self.receptor_cleaned_pdb_file_name,
                                                        kept_ligand_resname_list=self.kept_ligand_resname_list,
                                                        target_center=self.target_center,
                                                        num_grid_points=self.num_grid_points,
                                                        grid_spacing=self.grid_spacing,
                                                        covalent_residue_atom_info_list=self.covalent_residue_atom_info_list,
                                                        generate_ad4_grids=self.generate_ad4_grids,
                                                        working_dir_name=self.receptor_grids_working_dir_name)

        docking_grids_generator.generate_docking_grids()

        self.protein_pdbqt_file_name = os.path.join(self.receptor_grids_working_dir_name, 'protein.pdbqt')

        if self.generate_ad4_grids:
            self.protein_grid_prefix = os.path.join(self.receptor_grids_working_dir_name, 'protein')
        else:
            self.protein_grid_prefix = ''
        ####################################################################################################################################
###


def receptor_preprocessor(
        protein_pdb_file_name: str,
        kept_ligand_resname_list: Optional[List[str]] = None,
        prepared_hydrogen: bool = True,
        preserve_original_resname: bool = True,
        target_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        box_size: Tuple[float, float, float] = (22.5, 22.5, 22.5),
        covalent_residue_atom_info_list: Optional[List[Tuple[str, str]]] = None,
        generate_ad4_grids: bool = False,
        working_dir_name: str = '.'
        ):
    runner = ReceptorPreprocessorRunner(
        protein_pdb_file_name = protein_pdb_file_name,
        kept_ligand_resname_list=kept_ligand_resname_list,
        prepared_hydrogen=prepared_hydrogen,
        preserve_original_resname=preserve_original_resname,
        target_center=target_center,
        box_size=box_size,
        covalent_residue_atom_info_list=covalent_residue_atom_info_list,
        generate_ad4_grids=generate_ad4_grids,
        working_dir_name=working_dir_name
        )
    runner.run()
    protein_pdbqt_file_name = runner.protein_pdbqt_file_name
    protein_grid_prefix = runner.protein_grid_prefix
    return protein_pdbqt_file_name, protein_grid_prefix


if __name__ == "__main__":
    import argparse
    import shutil
    
    def parse_covalent_residue_atom_info(covalent_residue_atom_info_str: str) -> List[List[Tuple[str, str, int, str]]]:
        residue_info_list = []
        residue_atoms = covalent_residue_atom_info_str.split(',')
        for residue_atom in residue_atoms:
            residue_info = residue_atom.strip().split()
            chain_id, residue_name, residue_number, atom_name = residue_info
            residue_info_list.append((chain_id, residue_name, int(residue_number), atom_name))
        return residue_info_list
    
    parser = argparse.ArgumentParser(description="Receptor Preprocessor")
    parser.add_argument("-r", "--protein_pdb", type=str, required=True,
                        help="protein PDB file name")
    parser.add_argument("-kr", "--kept_ligand_resname_list", type=str, nargs="+", default=None,
                        help="list of ligand residue names to keep")
    parser.add_argument("-ph", "--prepared_hydrogen", action="store_false",
                        help="prepare hydrogen atoms")
    parser.add_argument("-pr", "--preserve_resname", action="store_false",
                        help="preserve original residue names")
    parser.add_argument("-c", "--target_center", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="target center coordinates (x, y, z). default=[0.0, 0.0, 0.0]")
    parser.add_argument("-s", "--box_size", nargs=3, type=float, default=[22.5, 22.5, 22.5],
                        help="box size. default=[22.5, 22.5, 22.5]")
    parser.add_argument("-g", "--generate_grids", action="store_true",
                        help="generate AD4 grids")
    parser.add_argument("-cra", "--covalent_residue_atom_info", type=str, default=None,
                        help="Atom information for covalent residues during receptor preprocessing.To use it like this: -cra 'A VAL 1 CA, A VAL 1 CB, A VAL 1 O'")
    parser.add_argument("-wd", "--working_dir", type=str, default=".",
                        help="working directory")
    parser.add_argument("-o", "--protein_pdbqt", type=str, required=True,
                        help="protein PDBQT file name")

    args = parser.parse_args()

    protein_pdbqt_file_name, protein_grid_prefix = receptor_preprocessor(
        protein_pdb_file_name=args.protein_pdb,
        kept_ligand_resname_list=args.kept_ligand_resname_list,
        prepared_hydrogen=args.prepared_hydrogen,
        preserve_original_resname=args.preserve_resname,
        target_center=tuple(args.target_center),
        box_size=tuple(args.box_size),
        covalent_residue_atom_info_list = parse_covalent_residue_atom_info(args.covalent_residue_atom_info) if args.covalent_residue_atom_info is not None else None,
        generate_ad4_grids=args.generate_grids,
        working_dir_name=args.working_dir
    )

    protein_pdbqt_dst = os.path.join(args.working_dir, args.protein_pdbqt)
    shutil.copy(protein_pdbqt_file_name, protein_pdbqt_dst)
