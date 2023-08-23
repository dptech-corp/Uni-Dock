import os
import string
import shutil
import uuid
from shutil import copyfile
import numpy as np

import MDAnalysis as mda

class ReceptorPDBReader(object):
    def __init__(self,
                 protein_pdb_file_name,
                 kept_ligand_resname_list=None,
                 ligand_as_hetatm=False,
                 output_protein_path='',):

        self.protein_pdb_file_name = protein_pdb_file_name
        self.kept_ligand_resname_list = kept_ligand_resname_list
        self.ligand_as_hetatm = ligand_as_hetatm
        self.output_protein_path = output_protein_path
        self.working_dir_name = f'/tmp/{uuid.uuid4().hex}'

    def __prepare_protein_pdb_file__(self):
        input_pdb_file_name = self.protein_pdb_file_name
        fixed_pdb_file_name = os.path.join(self.working_dir_name, 'protein_fixed.pdb')
        pdb4amber_pdb_file_name = os.path.join(self.working_dir_name, 'protein_pdb4amber.pdb')
        tleap_pdb_file_name = os.path.join(self.working_dir_name, 'protein_tleap.pdb')
        cleaned_pdb_file_name = os.path.join(self.working_dir_name, 'protein.pdb')
        pdb4amber_log_file_name = os.path.join(self.working_dir_name, 'pdb4amber.log')

        protein_universe = mda.Universe(input_pdb_file_name)
        protein_ag = protein_universe.select_atoms('protein and not name H*')

        protein_original_segid_list = protein_ag.segments.segids.tolist()
        if len(protein_original_segid_list) == 1:
            if protein_original_segid_list[0] == '' or protein_original_segid_list[0] == 'SYSTEM' or protein_original_segid_list[0] == 'SYST':
                protein_original_segid_list = ['X']

        protein_original_resid_list = protein_ag.residues.resids.tolist()
        protein_original_resname_list = protein_ag.residues.resnames.tolist()

#        removed_atom_idx_list = []
#        for atom in protein_ag:
#            if atom.name[0].isnumeric():
#                if int(atom.name[0]) == 1:
#                    atom.name = atom.name[1:]
#                else:
#                    removed_atom_idx_list.append(str(atom.index))

#        protein_fix_selection_string = 'not index ' + ' '.join(removed_atom_idx_list)
#        protein_fix_ag = protein_ag.select_atoms(protein_fix_selection_string)

#        for atom in protein_fix_ag:
#            if atom.name == 'OCT':
#                atom.name = 'OXT'

        protein_ag.write(fixed_pdb_file_name, bonds=None)

#        pdb4amber_command = f'cd {self.working_dir_name}; pdb4amber -i {fixed_pdb_file_name} -o {pdb4amber_pdb_file_name} --reduce -d -l {pdb4amber_log_file_name}; cd - >> cd.log'
#        os.system(pdb4amber_command)

        pdb4amber_command = f'cd {self.working_dir_name}; pdb4amber -i {fixed_pdb_file_name} -o {pdb4amber_pdb_file_name} -d -l {pdb4amber_log_file_name}; cd - >> cd.log'
        os.system(pdb4amber_command)

        tleap_source_file_name = os.path.join(os.path.dirname(__file__), 'templates', 'tleap_protein_template.in')
        tleap_destination_file_name = os.path.join(self.working_dir_name, 'tleap.in')
        copyfile(tleap_source_file_name, tleap_destination_file_name)

        tleap_command = f'cd {self.working_dir_name}; tleap -f tleap.in >> tleap.log; cd - >> tleap.log'
        os.system(tleap_command)

#        protein_cleaned_universe = mda.Universe(pdb4amber_pdb_file_name)
        protein_cleaned_universe = mda.Universe(tleap_pdb_file_name)

        protein_cleaned_ag = protein_cleaned_universe.atoms
        segment_group = protein_cleaned_ag.segments
        residue_group = protein_cleaned_ag.residues
        for current_segment_idx, current_segment in enumerate(segment_group):
            current_segid = protein_original_segid_list[current_segment_idx]
            current_segment.segid = current_segid

            for current_atom in current_segment.atoms:
                current_atom.chainID = current_segid
                current_atom.record_type = 'ATOM'

        for current_residue_idx, current_residue in enumerate(residue_group):
            current_resid = protein_original_resid_list[current_residue_idx]
            current_resname = protein_original_resname_list[current_residue_idx]
            current_residue.resid = current_resid
            current_residue.resname = current_resname

        protein_cleaned_ag.write(cleaned_pdb_file_name, bonds=None)

        available_segid_array = np.array(list(string.ascii_uppercase))
        current_segid_idx = np.where(available_segid_array == current_segid)[0][0]
        next_segid_idx = current_segid_idx + 1
        return next_segid_idx, cleaned_pdb_file_name

    def __prepare_ligand_pdb_file__(self, ligand_segid_idx, ligand_resname, ligand_as_hetatm):
        input_pdb_file_name = self.protein_pdb_file_name
        available_segid_list = list(string.ascii_uppercase)
        ligand_segid = available_segid_list[ligand_segid_idx]
        next_segid_idx = ligand_segid_idx + 1

        ligand_ag = mda.Universe(input_pdb_file_name).select_atoms('resname ' + ligand_resname)
        ligand_universe = mda.Merge(ligand_ag)
        segment_group = ligand_universe.segments
        ligand_segment = segment_group[0]
        ligand_segment.segid = ligand_segid
        ligand_residue = ligand_segment.residues[0]
        ligand_residue.resname = ligand_resname
        ligand_residue.resid = 1
        ligand_atoms = ligand_residue.atoms

        for atom_idx, atom in enumerate(ligand_atoms):
            atom.chainID = ligand_segid

            if ligand_as_hetatm:
                atom.record_type = 'HETATM'
            else:
                atom.record_type = 'ATOM'

            atom_element = atom.element
            atom_name = atom_element + str(atom_idx + 1)
            atom.name = atom_name

        output_pdb_file_name = os.path.join(self.working_dir_name, ligand_resname + '.pdb')
        segment_group.atoms.write(output_pdb_file_name, bonds=None)
        return next_segid_idx, output_pdb_file_name

    def run_receptor_system_cleaning(self):
        self.current_ligand_segid_idx, cleaned_protein_pdb_file_name = self.__prepare_protein_pdb_file__()
        receptor_component_pdb_file_name_list = [cleaned_protein_pdb_file_name]
        if self.kept_ligand_resname_list is not None:
            for ligand_resname in self.kept_ligand_resname_list:
                self.current_ligand_segid_idx, ligand_pdb_file_name = self.__prepare_ligand_pdb_file__(self.current_ligand_segid_idx, ligand_resname, self.ligand_as_hetatm)
                receptor_component_pdb_file_name_list.append(ligand_pdb_file_name)

        num_receptor_components = len(receptor_component_pdb_file_name_list)
        receptor_ag_list = [None] * num_receptor_components
        for component_idx in range(num_receptor_components):
            receptor_component_pdb_file_name = receptor_component_pdb_file_name_list[component_idx]
            component_ag = mda.Universe(receptor_component_pdb_file_name).atoms
            receptor_ag_list[component_idx] = component_ag

        receptor_universe = mda.Merge(*receptor_ag_list)
        receptor_ag = receptor_universe.atoms

        receptor_pdb_file_name = self.output_protein_path
        if receptor_pdb_file_name:
            receptor_pdb_file_name = os.path.splitext(self.protein_pdb_file_name)[0] + '_cleaned.pdb'
        receptor_ag.write(receptor_pdb_file_name, bonds=None)
        shutil.rmtree("templates", ignore_errors=True)

        return receptor_pdb_file_name
