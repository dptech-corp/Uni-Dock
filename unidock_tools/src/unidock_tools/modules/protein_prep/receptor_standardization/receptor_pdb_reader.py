import os
import string
from shutil import copyfile
import numpy as np

import MDAnalysis as mda

SUPPORTED_NONSTANDARD_RESNAME_LIST = ['NCYM', 'NASH', 'NGLH', 'NLYN', 'CCYM', 'CASH', 'CGLH', 'CLYN', 'AIB', '0AO', '200', 'ABA', 'ASQ', 'BCS', 'DAL', 'DAR', 'DAS', 'DCY', 'DGL', 'DGN', 'DHI', 'DIL', 'DLE', 'DLY', 'DPN', 'DPR', 'DSG', 'DSN', 'DTH', 'DTR', 'DTY', 'DVA', 'EBZ', 'GNC', 'HBZ', 'MEA', 'MED', 'NA8', 'NLE', 'NVA', 'PTR', 'SEM', 'SEP', 'THP', 'TPO', 'TY5', 'Z3E']

class ReceptorPDBReader(object):
    def __init__(self,
                 protein_pdb_file_name,
                 kept_ligand_resname_list=None,
                 ligand_as_hetatm=False,
                 prepared_hydrogen=False,
                 preserve_original_resname=True,
                 working_dir_name='.'):

        self.protein_pdb_file_name = protein_pdb_file_name
        self.kept_ligand_resname_list = kept_ligand_resname_list
        self.ligand_as_hetatm = ligand_as_hetatm
        self.prepared_hydrogen = prepared_hydrogen
        self.preserve_original_resname = preserve_original_resname
        self.working_dir_name = os.path.abspath(working_dir_name)

    def __prepare_protein_pdb_file__(self):
        input_pdb_file_name = self.protein_pdb_file_name
        fixed_pdb_file_name = os.path.join(self.working_dir_name, 'protein_fixed.pdb')
        pdb4amber_pdb_file_name = os.path.join(self.working_dir_name, 'protein_pdb4amber.pdb')
        reduce_pdb_file_name = os.path.join(self.working_dir_name, 'protein_reduce.pdb')
        reduce_pdb4amber_pdb_file_name = os.path.join(self.working_dir_name, 'protein_reduce_pdb4amber.pdb')
        tleap_input_pdb_file_name = os.path.join(self.working_dir_name, 'protein_tleap_in.pdb')
        refined_pdb_file_name = os.path.join(self.working_dir_name, 'protein_refined.pdb')
        cleaned_pdb_file_name = os.path.join(self.working_dir_name, 'protein.pdb')
        pdb4amber_log_file_name = os.path.join(self.working_dir_name, 'pdb4amber.log')

        protein_universe = mda.Universe(input_pdb_file_name)

        protein_selection_str = 'protein or resname ' + ' '.join(SUPPORTED_NONSTANDARD_RESNAME_LIST)

        if self.prepared_hydrogen:
            protein_ag = protein_universe.select_atoms(protein_selection_str)
        else:
            protein_ag = protein_universe.select_atoms(protein_selection_str + ' and not name H*')

        protein_original_segid_list = protein_ag.segments.segids.tolist()
        if len(protein_original_segid_list) == 1:
            if protein_original_segid_list[0] == '' or protein_original_segid_list[0] == 'SYSTEM' or protein_original_segid_list[0] == 'SYST':
                protein_original_segid_list = ['X']

        protein_original_resid_list = protein_ag.residues.resids.tolist()
        protein_original_resname_list = protein_ag.residues.resnames.tolist()

        protein_ag.write(fixed_pdb_file_name, bonds=None)

        pdb4amber_command = f'cd {self.working_dir_name}; pdb4amber -i {fixed_pdb_file_name} -o {pdb4amber_pdb_file_name} -d -l {pdb4amber_log_file_name}; cd - >> cd.log'
        os.system(pdb4amber_command)

        if self.prepared_hydrogen:
            protein_ag = mda.Universe(pdb4amber_pdb_file_name).atoms

            protein_segment_group = protein_ag.segments
            num_segments = protein_segment_group.n_segments
            segment_resid_nested_list = [None] * num_segments

            for segment_idx in range(num_segments):
                protein_segment = protein_segment_group[segment_idx]
                protein_segment_residues = protein_segment.residues
                num_residues = protein_segment_residues.n_residues
                resid_list = [None] * num_residues

                for residue_idx in range(num_residues):
                    protein_residue = protein_segment_residues[residue_idx]
                    resid_list[residue_idx] = protein_residue.resid

                segment_resid_nested_list[segment_idx] = resid_list

            protein_ag.write(refined_pdb_file_name, bonds=None)

        else:
            reduce_command = f'cd {self.working_dir_name}; reduce -FLIP {pdb4amber_pdb_file_name} > {reduce_pdb_file_name}; cd - >> cd.log'
            os.system(reduce_command)
            pdb4amber_command = f'cd {self.working_dir_name}; pdb4amber -i {reduce_pdb_file_name} -o {reduce_pdb4amber_pdb_file_name} -d -l {pdb4amber_log_file_name}; cd - >> cd.log'
            os.system(pdb4amber_command)

            protein_ag = mda.Universe(reduce_pdb4amber_pdb_file_name).select_atoms('protein and not name H*')
            protein_ag.write(tleap_input_pdb_file_name, bonds=None)

            tleap_source_file_name = os.path.join(os.path.dirname(__file__), 'templates', 'tleap_protein_template.in')
            tleap_destination_file_name = os.path.join(self.working_dir_name, 'tleap.in')
            copyfile(tleap_source_file_name, tleap_destination_file_name)

            tleap_command = f'cd {self.working_dir_name}; tleap -f tleap.in >> tleap.log; cd - >> tleap.log'
            os.system(tleap_command)

            ######################################################################################################
            ######################################################################################################
            ## Deal with tleap output PDB file with the lack of chain information.
            with open(refined_pdb_file_name, 'r') as f:
                tleap_pdb_line_list = f.readlines()

            segment_resid_nested_list = []

            resid_set = set()
            for tleap_pdb_line in tleap_pdb_line_list:
                if tleap_pdb_line.startswith('ATOM'):
                    resid = int(tleap_pdb_line.strip().split()[4])
                    resid_set.add(resid)
                elif tleap_pdb_line.startswith('TER'):
                    resid_list = list(resid_set)
                    segment_resid_nested_list.append(sorted(resid_list))
                    resid_set = set()

            ######################################################################################################
            ######################################################################################################

        protein_cleaned_universe = mda.Universe(refined_pdb_file_name)
        protein_cleaned_universe.add_TopologyAttr('element')
        protein_cleaned_ag = protein_cleaned_universe.atoms
        residue_group = protein_cleaned_ag.residues

        ########################################
        ## to get rid of annoying test warnings
        current_segid = 'A'
        ########################################
        for current_residue_idx, current_residue in enumerate(residue_group):
            prepared_resid = current_residue.resid
            for segment_idx, segment_resid_list in enumerate(segment_resid_nested_list):
                if prepared_resid in segment_resid_list:
                    current_segid = protein_original_segid_list[segment_idx]
                    break

            current_resid = protein_original_resid_list[current_residue_idx]
            current_resname = protein_original_resname_list[current_residue_idx]
            current_residue.resid = current_resid

            if self.preserve_original_resname:
                current_residue.resname = current_resname

            for current_atom in current_residue.atoms:
                current_atom.segment.segid = current_segid
                current_atom.chainID = current_segid
                current_atom.record_type = 'ATOM'
                current_atom.element = current_atom.type

        protein_cleaned_ag.write(cleaned_pdb_file_name, bonds=None)

        available_segid_array = np.array(list(string.ascii_uppercase))
        current_segid_idx = np.where(available_segid_array == current_segid)[0][0]
        next_segid_idx = current_segid_idx + 1
        return next_segid_idx, cleaned_pdb_file_name

    def __prepare_ligand_pdb_file__(self, ligand_segid_idx, ligand_resname, ligand_as_hetatm):
        input_pdb_file_name = self.protein_pdb_file_name
        available_segid_list = list(string.ascii_uppercase)
        ligand_resid = 1

        ligand_ag = mda.Universe(input_pdb_file_name).select_atoms('resname ' + ligand_resname)
        ligand_universe = mda.Merge(ligand_ag)

        ligand_residue_list = []
        for ligand_segment in ligand_universe.segments:
            for ligand_residue in ligand_segment.residues:
                ligand_residue_list.append(ligand_residue)

        for ligand_residue in ligand_residue_list:
            ligand_segid = available_segid_list[ligand_segid_idx]
            ligand_residue.segment.segid = ligand_segid

            ligand_residue.resname = ligand_resname
            ligand_residue.resid = ligand_resid

            ligand_atoms = ligand_residue.atoms
            for atom_idx, atom in enumerate(ligand_atoms):
                atom.chainID = ligand_segid

                if ligand_as_hetatm:
                    atom.record_type = 'HETATM'
                else:
                    atom.record_type = 'ATOM'

                if ligand_resname not in ['HOH', 'WAT', 'H2O', 'TIP3P']:
                    atom_element = atom.element
                    atom_name = atom_element + str(atom_idx + 1)
                    atom.name = atom_name

            if ligand_resname not in ['HOH', 'WAT', 'H2O', 'TIP3P', 'SO4', 'PO4', 'EDO']:
                ligand_segid_idx += 1
            else:
                ligand_resid += 1

        if ligand_resname not in ['HOH', 'WAT', 'H2O', 'TIP3P', 'SO4', 'PO4', 'EDO']:
            next_segid_idx = ligand_segid_idx
        else:
            next_segid_idx = ligand_segid_idx + 1

        output_pdb_file_name = os.path.join(self.working_dir_name, ligand_resname + '.pdb')
        ligand_universe.atoms.write(output_pdb_file_name, bonds=None)

        return next_segid_idx, output_pdb_file_name

    def run_receptor_system_cleaning(self):
        system_ag = mda.Universe(self.protein_pdb_file_name).atoms
        resname_array = system_ag.residues.resnames

        self.current_ligand_segid_idx, cleaned_protein_pdb_file_name = self.__prepare_protein_pdb_file__()
        receptor_component_pdb_file_name_list = [cleaned_protein_pdb_file_name]

        if self.kept_ligand_resname_list is not None:
            for ligand_resname in self.kept_ligand_resname_list:
                if ligand_resname in resname_array:
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

        receptor_pdb_file_name = os.path.join(self.working_dir_name, 'receptor.pdb')
        receptor_ag.write(receptor_pdb_file_name, bonds=None)

        return receptor_pdb_file_name
