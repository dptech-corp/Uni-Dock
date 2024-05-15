import os
import shutil

import MDAnalysis as mda

from unidock_tools.modules.protein_prep.receptor_topology.protein_topology import prepare_protein_residue_mol_list
from unidock_tools.modules.protein_prep.receptor_topology.protein_pdbqt_writer import ProteinPDBQTWriter
from unidock_tools.modules.protein_prep.receptor_topology.small_molecule_pdbqt_writer import SmallMoleculePDBQTWriter
from unidock_tools.modules.protein_prep.receptor_topology.grid_parameter_file_generator import GridParameterFileGenerator

SUPPORTED_NONSTANDARD_RESNAME_LIST = ['NCYM', 'NASH', 'NGLH', 'NLYN', 'CCYM', 'CASH', 'CGLH', 'CLYN', 'AIB', '0AO', '200', 'ABA', 'ASQ', 'BCS', 'DAL', 'DAR', 'DAS', 'DCY', 'DGL', 'DGN', 'DHI', 'DIL', 'DLE', 'DLY', 'DPN', 'DPR', 'DSG', 'DSN', 'DTH', 'DTR', 'DTY', 'DVA', 'EBZ', 'GNC', 'HBZ', 'MEA', 'MED', 'NA8', 'NLE', 'NVA', 'PTR', 'SEM', 'SEP', 'THP', 'TPO', 'TY5', 'Z3E']

class DockingGridsGenerator(object):
    def __init__(self,
                 protein_pdb_file_name,
                 kept_ligand_resname_list=None,
                 target_center=(0.0, 0.0, 0.0),
                 num_grid_points=(60, 60, 60),
                 grid_spacing=(0.375, 0.375, 0.375),
                 covalent_residue_atom_info_list=None,
                 generate_ad4_grids=False,
                 working_dir_name='.'):

        self.protein_pdb_file_name = os.path.abspath(protein_pdb_file_name)
        self.kept_ligand_resname_list = kept_ligand_resname_list

        self.target_center = target_center
        self.num_grid_points = num_grid_points
        self.grid_spacing = grid_spacing

        self.covalent_residue_atom_info_list = covalent_residue_atom_info_list
        self.generate_ad4_grids = generate_ad4_grids
        self.working_dir_name = os.path.abspath(working_dir_name)

    def __prepare_protein_pdb_file__(self, input_pdb_file_name, output_pdb_file_name):
        protein_universe = mda.Universe(input_pdb_file_name)
        protein_selection_str = 'protein or resname ' + ' '.join(SUPPORTED_NONSTANDARD_RESNAME_LIST)
        protein_ag = protein_universe.select_atoms(protein_selection_str)
        protein_ag.write(output_pdb_file_name, bonds=None)

    def __remove_h_for_covalent_atoms_on_protein_pdbqt_line_list__(self,
                                                                   atom,
                                                                   covalent_part_pdbqt_file_line_list,
                                                                   protein_pdbqt_file_line_list):

        neighbor_h_atom_info_list = []
        for neighbor_atom in atom.GetNeighbors():
            neighbor_chain_idx = neighbor_atom.GetProp('chain_idx')
            neighbor_resname = neighbor_atom.GetProp('residue_name')
            neighbor_resid = neighbor_atom.GetIntProp('residue_idx')
            neighbor_atom_name = neighbor_atom.GetProp('atom_name')
            neighbor_atom_info = (neighbor_chain_idx, neighbor_resname, neighbor_resid, neighbor_atom_name)

            if neighbor_atom_name.startswith('H'):
                neighbor_h_atom_info_list.append(neighbor_atom_info)

            for neighbor_h_atom_info in neighbor_h_atom_info_list:
                for protein_pdbqt_file_line in protein_pdbqt_file_line_list:
                    if protein_pdbqt_file_line.startswith('ATOM') or protein_pdbqt_file_line.startswith('HETATM'):
                        pdbqt_atom_info = (protein_pdbqt_file_line[21].strip(), protein_pdbqt_file_line[17:21].strip(), int(protein_pdbqt_file_line[22:26].strip()), protein_pdbqt_file_line[12:16].strip())
                        if pdbqt_atom_info == neighbor_h_atom_info:
                            covalent_part_pdbqt_file_line_list.append(protein_pdbqt_file_line)
                            protein_pdbqt_file_line_list.remove(protein_pdbqt_file_line)
                            break

    def __prepare_covalent_bond_on_residue__(self,
                                             protein_mol,
                                             covalent_residue_atom_info_list,
                                             raw_protein_pdbqt_file_name,
                                             prepared_protein_pdbqt_file_name,
                                             covalent_part_pdbqt_file_name,
                                             covalent_atom_idx_dat_file_name):

        covalent_anchor_atom_info = covalent_residue_atom_info_list[0]
        covalent_bond_start_atom_info = covalent_residue_atom_info_list[1]
        covalent_bond_end_atom_info = covalent_residue_atom_info_list[2]

        with open(raw_protein_pdbqt_file_name, 'r') as protein_pdbqt_file:
            protein_pdbqt_file_line_list = protein_pdbqt_file.readlines()

        covalent_part_pdbqt_file_line_list = []
        covalent_atom_idx_list = [None] * 3

        num_protein_atoms = protein_mol.GetNumAtoms()
        for atom_idx in range(num_protein_atoms):
            atom = protein_mol.GetAtomWithIdx(atom_idx)
            current_chain_idx = atom.GetProp('chain_idx')
            current_resname = atom.GetProp('residue_name')
            current_resid = atom.GetIntProp('residue_idx')
            current_atom_name = atom.GetProp('atom_name')
            current_atom_info = (current_chain_idx, current_resname, current_resid, current_atom_name)

            if current_atom_info == covalent_anchor_atom_info:
                for protein_pdbqt_file_line in protein_pdbqt_file_line_list:
                    if protein_pdbqt_file_line.startswith('ATOM') or protein_pdbqt_file_line.startswith('HETATM'):
                        pdbqt_atom_info = (protein_pdbqt_file_line[21].strip(), protein_pdbqt_file_line[17:21].strip(), int(protein_pdbqt_file_line[22:26].strip()), protein_pdbqt_file_line[12:16].strip())
                        pdbqt_atom_idx = protein_pdbqt_file_line[6:11].strip()
                        if pdbqt_atom_info == current_atom_info:
                            covalent_atom_idx_list[0] = pdbqt_atom_idx
                            covalent_part_pdbqt_file_line_list.append(protein_pdbqt_file_line)
                            protein_pdbqt_file_line_list.remove(protein_pdbqt_file_line)
                            break

                self.__remove_h_for_covalent_atoms_on_protein_pdbqt_line_list__(atom, covalent_part_pdbqt_file_line_list, protein_pdbqt_file_line_list)

            elif current_atom_info == covalent_bond_start_atom_info:
                for protein_pdbqt_file_line in protein_pdbqt_file_line_list:
                    if protein_pdbqt_file_line.startswith('ATOM') or protein_pdbqt_file_line.startswith('HETATM'):
                        pdbqt_atom_info = (protein_pdbqt_file_line[21].strip(), protein_pdbqt_file_line[17:21].strip(), int(protein_pdbqt_file_line[22:26].strip()), protein_pdbqt_file_line[12:16].strip())
                        pdbqt_atom_idx = protein_pdbqt_file_line[6:11].strip()
                        if pdbqt_atom_info == current_atom_info:
                            covalent_atom_idx_list[1] = pdbqt_atom_idx
                            covalent_part_pdbqt_file_line_list.append(protein_pdbqt_file_line)
                            protein_pdbqt_file_line_list.remove(protein_pdbqt_file_line)
                            break

                self.__remove_h_for_covalent_atoms_on_protein_pdbqt_line_list__(atom, covalent_part_pdbqt_file_line_list, protein_pdbqt_file_line_list)

            elif current_atom_info == covalent_bond_end_atom_info:
                for protein_pdbqt_file_line in protein_pdbqt_file_line_list:
                    if protein_pdbqt_file_line.startswith('ATOM') or protein_pdbqt_file_line.startswith('HETATM'):
                        pdbqt_atom_info = (protein_pdbqt_file_line[21].strip(), protein_pdbqt_file_line[17:21].strip(), int(protein_pdbqt_file_line[22:26].strip()), protein_pdbqt_file_line[12:16].strip())
                        pdbqt_atom_idx = protein_pdbqt_file_line[6:11].strip()
                        if pdbqt_atom_info == current_atom_info:
                            covalent_atom_idx_list[2] = pdbqt_atom_idx
                            covalent_part_pdbqt_file_line_list.append(protein_pdbqt_file_line)
                            protein_pdbqt_file_line_list.remove(protein_pdbqt_file_line)
                            break

                self.__remove_h_for_covalent_atoms_on_protein_pdbqt_line_list__(atom, covalent_part_pdbqt_file_line_list, protein_pdbqt_file_line_list)

        protein_pdbqt_file_line_scripts = ''.join(protein_pdbqt_file_line_list)
        covalent_part_pdbqt_file_line_scripts = ''.join(covalent_part_pdbqt_file_line_list)
        covalent_atom_idx_dat_file_line_scripts = ','.join(covalent_atom_idx_list) + '\n'
        with open(prepared_protein_pdbqt_file_name, 'w') as protein_pdbqt_file:
            protein_pdbqt_file.write(protein_pdbqt_file_line_scripts)

        with open(covalent_part_pdbqt_file_name, 'w') as covalent_part_pdbqt_file:
            covalent_part_pdbqt_file.write(covalent_part_pdbqt_file_line_scripts)

        with open(covalent_atom_idx_dat_file_name, 'w') as covalent_atom_idx_dat_file:
            covalent_atom_idx_dat_file.write(covalent_atom_idx_dat_file_line_scripts)

    def __prepare_ligand_pdbqt_file__(self, working_dir_name, input_pdb_file_name, ligand_resname, start_atom_idx):
        ligand_ag = mda.Universe(input_pdb_file_name).select_atoms(f'resname {ligand_resname}')
        num_ligand_atoms = ligand_ag.n_atoms
        atom_name_list = [None] * num_ligand_atoms
        resid_list = [None] * num_ligand_atoms
        resname_list = [None] * num_ligand_atoms
        chain_idx_list = [None] * num_ligand_atoms

        for atom_idx in range(num_ligand_atoms):
            atom = ligand_ag[atom_idx]
            atom_name_list[atom_idx] = atom.name
            resid_list[atom_idx] = atom.resid
            resname_list[atom_idx] = atom.resname
            chain_idx_list[atom_idx] = atom.chainID

        ligand_universe = mda.Merge(ligand_ag)
        mda_to_rdkit = mda._CONVERTERS['RDKIT']().convert
        ligand_mol = mda_to_rdkit(ligand_universe, NoImplicit=False)
        ligand_mol.SetProp('residue_name', ligand_resname)

        for atom_idx in range(num_ligand_atoms):
            atom =  ligand_mol.GetAtomWithIdx(atom_idx)
            atom.SetProp('atom_name', atom_name_list[atom_idx])
            atom.SetIntProp('residue_idx', int(resid_list[atom_idx]))
            atom.SetProp('residue_name', resname_list[atom_idx])
            atom.SetProp('chain_idx', chain_idx_list[atom_idx])

        small_molecule_pdbqt_writer = SmallMoleculePDBQTWriter(ligand_mol,
                                                               start_atom_idx,
                                                               working_dir_name=working_dir_name)

        small_molecule_pdbqt_writer.write_small_molecule_pdbqt_file()
        next_atom_idx = small_molecule_pdbqt_writer.next_atom_idx

        return next_atom_idx

    def __prepare_complex_receptor_pdbqt_file__(self, working_dir_name, prepared_protein_pdbqt_file_name, kept_ligand_resname_list):
        with open(prepared_protein_pdbqt_file_name, 'r') as prepared_protein_pdbqt_file:
            prepared_protein_pdbqt_line_list = prepared_protein_pdbqt_file.readlines()

        complex_receptor_pdbqt_line_list = prepared_protein_pdbqt_line_list
        for ligand_resname in kept_ligand_resname_list:
            ligand_pdbqt_file_name = os.path.join(working_dir_name, ligand_resname + '.pdbqt')
            with open(ligand_pdbqt_file_name, 'r') as ligand_pdbqt_file:
                ligand_pdbqt_file_line_list = ligand_pdbqt_file.readlines()

            complex_receptor_pdbqt_line_list += ligand_pdbqt_file_line_list
            complex_receptor_pdbqt_line_list += ['TER\n']

        with open(prepared_protein_pdbqt_file_name, 'w') as complex_receptor_pdbqt_file:
            for complex_receptor_pdbqt_line in complex_receptor_pdbqt_line_list:
                complex_receptor_pdbqt_file.write(complex_receptor_pdbqt_line)

    def generate_docking_grids(self):
        protein_output_pdb_file_name = os.path.join(self.working_dir_name, 'protein.pdb')
        protein_output_raw_pdbqt_file_name = os.path.join(self.working_dir_name, 'protein_original.pdbqt')
        protein_output_pdbqt_file_name = os.path.join(self.working_dir_name, 'protein.pdbqt')
        protein_output_gpf_file_name = os.path.join(self.working_dir_name, 'protein.gpf')
        protein_output_glg_file_name = os.path.join(self.working_dir_name, 'protein.glg')
        self.__prepare_protein_pdb_file__(self.protein_pdb_file_name, protein_output_pdb_file_name)

        protein_mol, protein_residue_mol_list = prepare_protein_residue_mol_list(protein_output_pdb_file_name)

        protein_pdbqt_writer = ProteinPDBQTWriter(protein_mol,
                                                  protein_residue_mol_list,
                                                  working_dir_name=self.working_dir_name)

        protein_pdbqt_writer.write_protein_pdbqt_file()
        protein_output_raw_pdbqt_file_name = protein_pdbqt_writer.protein_pdbqt_file_name
        next_atom_idx = protein_pdbqt_writer.next_atom_idx

        if self.covalent_residue_atom_info_list is not None:
            if len(self.covalent_residue_atom_info_list) != 3:
                raise ValueError('Length of covalent bond info must be 3')
            else:
                covalent_part_pdbqt_file_name = os.path.join(self.working_dir_name, 'covalent.pdbqt')
                covalent_atom_idx_dat_file_name = os.path.join(self.working_dir_name, 'protein_covalent_atom_indices.dat')
                self.__prepare_covalent_bond_on_residue__(protein_mol,
                                                          self.covalent_residue_atom_info_list,
                                                          protein_output_raw_pdbqt_file_name,
                                                          protein_output_pdbqt_file_name,
                                                          covalent_part_pdbqt_file_name,
                                                          covalent_atom_idx_dat_file_name)

        else:
            shutil.copyfile(protein_output_raw_pdbqt_file_name, protein_output_pdbqt_file_name)

        if self.kept_ligand_resname_list is not None:
            for ligand_resname in self.kept_ligand_resname_list:
                next_atom_idx = self.__prepare_ligand_pdbqt_file__(self.working_dir_name, self.protein_pdb_file_name, ligand_resname, next_atom_idx)

            self.__prepare_complex_receptor_pdbqt_file__(self.working_dir_name, protein_output_pdbqt_file_name, self.kept_ligand_resname_list)

        if self.generate_ad4_grids:
            gpf_generator = GridParameterFileGenerator(protein_output_pdbqt_file_name,
                                                       target_center=self.target_center,
                                                       num_grid_points=self.num_grid_points,
                                                       grid_spacing=self.grid_spacing,
                                                       working_dir_name=self.working_dir_name)

            gpf_generator.write_grid_parameter_file()

            autogrid_binary_file_name = os.path.join(os.path.dirname(__file__), 'bin', 'autogrid4')
            autogrid_chmod_command = f'chmod +x {autogrid_binary_file_name}'
            autogrid_command = f'cd {self.working_dir_name}; {autogrid_binary_file_name} -p {os.path.basename(protein_output_gpf_file_name)} -l {os.path.basename(protein_output_glg_file_name)}; cd -'
            os.system(autogrid_chmod_command)
            os.system(autogrid_command)
