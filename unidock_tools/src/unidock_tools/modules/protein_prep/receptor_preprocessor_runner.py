
import os
from shutil import rmtree

from typing import List, Tuple, Dict, Optional
from unidock_tools.modules.protein_prep.receptor_standardization.receptor_pdb_reader import ReceptorPDBReader
from unidock_tools.modules.protein_prep.receptor_topology.autogrid_runner import AutoGridRunner

class ReceptorPreprocessorRunner(object):
    def __init__(self,
                 protein_pdb_file_name,
                 protein_conf_name='protein_conf_0',
                 kept_ligand_resname_list=None,
                 prepared_hydrogen=False,
                 preserve_original_resname=True,
                 target_center=(0.0, 0.0, 0.0),
                 box_size=(22.5, 22.5, 22.5),
                 covalent_residue_atom_info_list=None,
                 generate_ad4_grids=False,
                 working_dir_name='.'):

        self.protein_pdb_file_name = protein_pdb_file_name
        self.protein_conf_name = protein_conf_name
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
        autogrid_runner = AutoGridRunner(protein_pdb_file_name_list=[self.receptor_cleaned_pdb_file_name],
                                         protein_conf_name_list=[self.protein_conf_name],
                                         kept_ligand_resname_nested_list=[self.kept_ligand_resname_list],
                                         target_center_list=[self.target_center],
                                         box_size=self.box_size,
                                         covalent_residue_atom_info_nested_list=[self.covalent_residue_atom_info_list],
                                         generate_ad4_grids=self.generate_ad4_grids,
                                         working_dir_name=self.receptor_grids_working_dir_name)

        autogrid_runner.run()

        self.protein_pdbqt_file_name = autogrid_runner.protein_info_df.loc[:, 'protein_pdbqt_file_name'].values.tolist()[0] 
        ####################################################################################################################################
###


def receptor_preprocessor(
        protein_pdb_file_name: str,
        protein_conf_name: str = 'protein_conf_0',
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
        protein_conf_name=protein_conf_name,
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
    return protein_pdbqt_file_name


if __name__ == "__main__":
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description="Receptor Preprocessor")
    parser.add_argument("-r", "--protein_pdb", type=str, required=True,
                        help="protein PDB file name")
    parser.add_argument("-l", "--ligand_resname", nargs="+", default=None,
                        help="list of ligand residue names to keep")
    parser.add_argument("-H", "--prepared_hydrogen", action="store_true",
                        help="prepare hydrogen atoms")
    parser.add_argument("-p", "--preserve_resname", action="store_false",
                        help="preserve original residue names")
    parser.add_argument("-c", "--target_center", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="target center coordinates (x, y, z)")
    parser.add_argument("-s", "--box_size", nargs=3, type=float, default=[22.5, 22.5, 22.5],
                        help="box size")
    parser.add_argument("-g", "--generate_grids", action="store_true",
                        help="generate AD4 grids")
    parser.add_argument("-w", "--working_dir", type=str, default=".",
                        help="working directory")
    parser.add_argument("-o", "--protein_pdbqt", type=str, required=True,
                        help="protein PDBQT file name")

    args = parser.parse_args()

    protein_pdbqt_file_name = receptor_preprocessor(
        protein_pdb_file_name=args.protein_pdb,
        kept_ligand_resname_list=args.ligand_resname,
        prepared_hydrogen=args.prepared_hydrogen,
        preserve_original_resname=args.preserve_resname,
        target_center=tuple(args.target_center),
        box_size=tuple(args.box_size),
        generate_ad4_grids=args.generate_grids,
        working_dir_name=args.working_dir
    )

    protein_pdbqt_dst = os.path.join(args.working_dir, args.protein_pdbqt)
    shutil.copy(protein_pdbqt_file_name, protein_pdbqt_dst)
    
