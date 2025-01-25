import os
from shutil import copyfile

import numpy as np

class GridParameterFileGenerator(object):
    def __init__(self,
                 protein_pdbqt_file_name,
                 target_center=(0.0, 0.0, 0.0),
                 num_grid_points=(60, 60, 60),
                 grid_spacing=(0.375, 0.375, 0.375),
                 working_dir_name='.'):

        self.protein_pdbqt_file_name = os.path.abspath(protein_pdbqt_file_name)
        self.target_center = target_center
        self.num_grid_points = num_grid_points
        self.grid_spacing = grid_spacing
        self.working_dir_name = os.path.abspath(working_dir_name)
        self.output_gpf_file_name = os.path.join(self.working_dir_name, 'protein.gpf')

        self.protein_atom_type_list = self.__get_protein_atom_type_list__(self.protein_pdbqt_file_name)
        self.ligand_atom_type_list = ['A', 'Br', 'C', 'Ca', 'Cl',
                                      'F', 'Fe', 'G', 'GA', 'H', 'HD', 'HS',
                                      'I', 'J', 'Mg', 'Mn', 'N', 'NA', 'NS',
                                      'OA', 'OS', 'P', 'Q', 'S', 'SA', 'Z', 'Zn']

        self.autodock_parameter_file_name = self.__get_autodock_parameter_file__(self.working_dir_name)

    def __get_protein_atom_type_list__(self, protein_pdbqt_file_name):
        atom_type_list = []
        with open(protein_pdbqt_file_name, 'r') as protein_pdbqt_file:
            protein_pdbqt_file_line_list = protein_pdbqt_file.readlines()

        for protein_pdbqt_file_line in protein_pdbqt_file_line_list:
            if protein_pdbqt_file_line.startswith('ATOM') or protein_pdbqt_file_line.startswith('HETATM'):
                atom_type = protein_pdbqt_file_line.strip().split()[-1]
                atom_type_list.append(atom_type)

        unique_atom_type_list = np.unique(atom_type_list).tolist()

        return unique_atom_type_list

    def __get_autodock_parameter_file__(self, working_dir_name):
        parameter_source_file_name = os.path.join(os.path.dirname(__file__), 'data', 'AD4.1_bound.dat')
        parameter_destination_file_name = os.path.join(working_dir_name, 'AD4.1_bound.dat')
        copyfile(parameter_source_file_name, parameter_destination_file_name)

        return parameter_destination_file_name

    def write_grid_parameter_file(self):
        parameter_scripts_list = []
        protein_pdbqt_base_file_name = os.path.basename(self.protein_pdbqt_file_name)
        autodock_parameter_base_file_name = os.path.basename(self.autodock_parameter_file_name)

        parameter_scripts_list.append('outlev 2')
        parameter_scripts_list.append('parameter_file ' + autodock_parameter_base_file_name)
        parameter_scripts_list.append('npts ' + ' '.join([str(num_grid_point) for num_grid_point in self.num_grid_points]))
        parameter_scripts_list.append('gridfld protein.maps.fld')
        parameter_scripts_list.append('spacing ' + ' '.join([str(grid_spacing_value) for grid_spacing_value in self.grid_spacing]))
        parameter_scripts_list.append('receptor_types ' + ' '.join(self.protein_atom_type_list))
        parameter_scripts_list.append('ligand_types ' + ' '.join(self.ligand_atom_type_list))
        parameter_scripts_list.append('receptor ' + protein_pdbqt_base_file_name)
        parameter_scripts_list.append('gridcenter ' + ' '.join([str(target_center_component) for target_center_component in self.target_center]))
        parameter_scripts_list.append('smooth 0.500000')

        for ligand_atom_type in self.ligand_atom_type_list:
            parameter_scripts_list.append('map protein.' + ligand_atom_type + '.map')

        parameter_scripts_list.append('elecmap protein.e.map')
        parameter_scripts_list.append('dsolvmap protein.d.map')
        parameter_scripts_list.append('dielectric -0.145600')

        parameter_scripts = '\n'.join(parameter_scripts_list)

        with open(self.output_gpf_file_name, 'w') as output_gpf_file:
            output_gpf_file.write(parameter_scripts)
