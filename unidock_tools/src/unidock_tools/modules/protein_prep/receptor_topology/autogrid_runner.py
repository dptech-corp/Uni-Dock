import os
import numpy as np
import pandas as pd

from docking_grids_generator import DockingGridsGenerator

class AutoGridRunner(object):
    def __init__(self,
                 protein_pdb_file_name_list,
                 protein_conf_name_list=None,
                 kept_ligand_resname_nested_list=None,
                 target_center_list=None,
                 box_size=(22.5, 22.5, 22.5),
                 covalent_residue_atom_info_nested_list=None,
                 generate_ad4_grids=False,
                 working_dir_name='.'):

#        os.environ['MGLPY'] = os.environ.get('MGLPY', '/data/Modules/ADFR/bin/python')
#        os.environ['MGLUTIL'] = os.environ.get('MGLUTIL', '/data/Modules/ADFR/CCSBpckgs/AutoDockTools/Utilities24')

        self.protein_pdb_file_name_list = protein_pdb_file_name_list
        self.num_protein_conformations = len(self.protein_pdb_file_name_list)

        if protein_conf_name_list is None:
            self.protein_conf_name_list = [None] * self.num_protein_conformations
            for protein_conf_idx in range(self.num_protein_conformations):
                self.protein_conf_name_list[protein_conf_idx] = 'protein_conf_' + str(protein_conf_idx)
        else:
            self.protein_conf_name_list = protein_conf_name_list

        self.protein_pdbqt_file_name_list = [None] * self.num_protein_conformations
        self.protein_grid_file_name_list = [None] * self.num_protein_conformations

        if kept_ligand_resname_nested_list is None:
            self.kept_ligand_resname_nested_list = [None] * self.num_protein_conformations
        else:
            self.kept_ligand_resname_nested_list = kept_ligand_resname_nested_list

        if covalent_residue_atom_info_nested_list is None:
            self.covalent_residue_atom_info_nested_list = [None] * self.num_protein_conformations
        else:
            covalent_residue_atom_info_nested_list = covalent_residue_atom_info_nested_list
            if len(covalent_residue_atom_info_nested_list) != self.num_protein_conformations:
                raise ValueError('Specified length of covalent residue atom info list does not match the number of protein conformations')

            self.covalent_residue_atom_info_nested_list = covalent_residue_atom_info_nested_list

        self.target_center_list = target_center_list

        self.box_size = box_size
        box_size_array = np.array(self.box_size)
        num_grid_points_array = box_size_array / 0.375
        num_grid_points_array = num_grid_points_array.astype(np.int32)
        self.num_grid_points = tuple(num_grid_points_array)
        self.grid_spacing = (0.375, 0.375, 0.375)

        self.generate_ad4_grids = generate_ad4_grids
        self.working_dir_name = os.path.abspath(working_dir_name)

    def run(self):
        for protein_conf_idx in range(self.num_protein_conformations):
            protein_pdb_file_name = self.protein_pdb_file_name_list[protein_conf_idx]
            covalent_residue_atom_info_list = self.covalent_residue_atom_info_nested_list[protein_conf_idx]
            protein_conf_name = self.protein_conf_name_list[protein_conf_idx]
            kept_ligand_resname_list = self.kept_ligand_resname_nested_list[protein_conf_idx]
            target_center = self.target_center_list[protein_conf_idx]
            current_working_dir_name = os.path.join(self.working_dir_name, protein_conf_name)
            os.mkdir(current_working_dir_name)

            docking_grids_generator = DockingGridsGenerator(protein_pdb_file_name,
                                                            kept_ligand_resname_list=kept_ligand_resname_list,
                                                            target_center=target_center,
                                                            num_grid_points=self.num_grid_points,
                                                            grid_spacing=self.grid_spacing,
                                                            covalent_residue_atom_info_list=covalent_residue_atom_info_list,
                                                            working_dir_name=current_working_dir_name)

            docking_grids_generator.generate_docking_grids()

            self.protein_pdbqt_file_name_list[protein_conf_idx] = os.path.join(current_working_dir_name, 'protein.pdbqt')

            if self.generate_ad4_grids:
                self.protein_grid_file_name_list[protein_conf_idx] = os.path.join(current_working_dir_name, 'protein.maps.fld')
            else:
                self.protein_grid_file_name_list[protein_conf_idx] = None

        protein_info_dict = {}
        protein_info_dict['protein_conf_name'] = self.protein_conf_name_list
        protein_info_dict['protein_pdb_file_name'] = self.protein_pdb_file_name_list
        protein_info_dict['protein_pdbqt_file_name'] = self.protein_pdbqt_file_name_list
        protein_info_dict['protein_grid_maps_fld_file_name'] = self.protein_grid_file_name_list
        self.protein_info_df = pd.DataFrame(protein_info_dict)

        protein_info_csv_file_name = os.path.join(self.working_dir_name, 'protein_conf_grids.csv')
        self.protein_info_df.to_csv(protein_info_csv_file_name, index=False)
