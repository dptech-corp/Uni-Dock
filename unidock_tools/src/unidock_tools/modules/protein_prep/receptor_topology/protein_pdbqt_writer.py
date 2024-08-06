import os

from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock_tools.modules.protein_prep.receptor_topology.smarts_definition import HB_DONOR, HB_ACCEPTOR
from unidock_tools.modules.protein_prep.receptor_topology.amino_acid_atom_types import RESIDUE_PARAMETER_DICT

class ProteinPDBQTWriter(object):
    def __init__(self,
                 protein_mol,
                 protein_residue_mol_list,
                 working_dir_name='.'):

        self.residue_parameter_dict = RESIDUE_PARAMETER_DICT
        self.protein_mol = protein_mol
        self.protein_residue_mol_list = protein_residue_mol_list

        self.working_dir_name = os.path.abspath(working_dir_name)
        self.protein_pdbqt_file_name = os.path.join(self.working_dir_name, 'protein_original.pdbqt')

    def __get_gasteiger_charges__(self):
        # calculate Gasteiger-Marsili partial charge
        atom_charge_dict = {}
        ComputeGasteigerCharges(self.protein_mol)
        for atom in self.protein_mol.GetAtoms():
            # get atom index and charge
            atom_idx = atom.GetIntProp('atom_idx')
            charge = atom.GetDoubleProp('_GasteigerCharge')
            atom_charge_dict[atom_idx] = charge

        return atom_charge_dict

    def __get_atom_types__(self):
        atom_type_dict = {}
        for atom in self.protein_mol.GetAtoms():
            current_atom_name = atom.GetProp('atom_name')
            current_atom_idx = atom.GetIntProp('atom_idx')
            current_residue_name = atom.GetProp('residue_name')

            ##### for ASP GLU HIS, find correct resname
            if current_residue_name == 'ASP' and current_atom_name == 'HD2':
                current_residue_name ='ASH'
            if current_residue_name == 'GLU' and current_atom_name =='HE2':
                current_residue_name ='GLH'
            if current_residue_name == 'HIS':
                current_residue_name = 'HIP'
                if current_atom_name =='ND1':
                    neighbor_symbols = {neighbor.GetSymbol() for neighbor in atom.GetNeighbors()}
                    current_residue_name = 'HIE' if 'H' not in neighbor_symbols else current_residue_name
                if current_atom_name =='NE2':
                    neighbor_symbols = {neighbor.GetSymbol() for neighbor in atom.GetNeighbors()}
                    current_residue_name = 'HID' if 'H' not in neighbor_symbols else current_residue_name

            if current_residue_name in self.residue_parameter_dict.keys():
                template_atom_name_list = self.residue_parameter_dict[current_residue_name]['atom_names']
                template_atom_type_list = self.residue_parameter_dict[current_residue_name]['atom_types']

                if current_atom_name in template_atom_name_list:
                    atom_type = template_atom_type_list[template_atom_name_list.index(current_atom_name)]
                    atom_type_dict[current_atom_idx] = atom_type
                else:
                    atom_type_dict[current_atom_idx] = None

            else:
                atom_type_dict[current_atom_idx] = None

        donor_smarts = HB_DONOR + '[H]'
        accept_smarts = HB_ACCEPTOR
        donor_pattenr_mol = Chem.MolFromSmarts(donor_smarts)
        acceptor_pattern_mol = Chem.MolFromSmarts(accept_smarts)

        protein_residue_mol_list_by_chain = {}
        for protein_residue_mol in self.protein_residue_mol_list:
            chain_idx = protein_residue_mol.GetAtomWithIdx(0).GetProp('chain_idx')
            if chain_idx not in protein_residue_mol_list_by_chain:
                protein_residue_mol_list_by_chain[chain_idx] = []

            protein_residue_mol_list_by_chain[chain_idx].append(protein_residue_mol)

        last_residue_mol_by_chain = {chain_idx: protein_residue_mol_list_by_chain[chain_idx][-1] for chain_idx in protein_residue_mol_list_by_chain}
        first_residue_mol_by_chain = {chain_idx: protein_residue_mol_list_by_chain[chain_idx][0] for chain_idx in protein_residue_mol_list_by_chain}

        for _, last_residue_mol in last_residue_mol_by_chain.items():
            acceptor_match_atom_idx_list = list(last_residue_mol.GetSubstructMatches(acceptor_pattern_mol))
            acceptor_atom_idx_list = [acceptor_match_atom_idx_tuple[0] for acceptor_match_atom_idx_tuple in acceptor_match_atom_idx_list]
            for acceptor_atom_idx in acceptor_atom_idx_list:
                atom_idx = last_residue_mol.GetAtomWithIdx(acceptor_atom_idx).GetIntProp('atom_idx')
                element = last_residue_mol.GetAtomWithIdx(acceptor_atom_idx).GetSymbol()
                atom_type = element + 'A'
                atom_type_dict[atom_idx] = atom_type

        for _, first_residue_mol in first_residue_mol_by_chain.items():
            donor_match_atom_idx_list = list(first_residue_mol.GetSubstructMatches(donor_pattenr_mol))
            donor_hydrogen_atom_idx_list = [donor_match_atom_idx_tuple[1] for donor_match_atom_idx_tuple in donor_match_atom_idx_list]
            for donor_hydrogen_atom_idx in donor_hydrogen_atom_idx_list:
                atom_idx = first_residue_mol.GetAtomWithIdx(donor_hydrogen_atom_idx).GetIntProp('atom_idx')
                atom_type_dict[atom_idx] = 'HD'

        for atom_idx, atom_type in atom_type_dict.items():
            if not atom_type:
                raise ValueError(f'Residue name or atom name cannot be found in templates! atom index: {atom_idx}')

        return atom_type_dict

    def write_protein_pdbqt_file(self):
        self.atom_charge_dict = self.__get_gasteiger_charges__()
        self.atom_type_dict = self.__get_atom_types__()

        self.pdbqt_atom_line_list = []
        self.pdbqt_atom_line_format = '{:4s}  {:5d} {:^4s} {:4s}{:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}\n'

        if self.protein_mol.GetNumAtoms() == 0:
            raise ValueError('Empty protein mol!!')

        atom_idx = 0
        for atom in self.protein_mol.GetAtoms():
            atom_idx = atom.GetIntProp('atom_idx')
            atom_info_tuple = ('ATOM',  # atom record name
                               atom_idx,  # atom serial number
                               atom.GetProp('atom_name'),  # atom name
                               atom.GetProp('residue_name'),  # residue name
                               atom.GetProp('chain_idx'),  # chain identifier
                               atom.GetIntProp('residue_idx'),  # residue sequence number
                               atom.GetDoubleProp('x'),  # x coordinate
                               atom.GetDoubleProp('y'),  # y coordinate
                               atom.GetDoubleProp('z'),  # z coordinate
                               1.0,  # occupancy
                               0.0,  # temperature factor
                               self.atom_charge_dict[atom_idx],  # partial charge
                               self.atom_type_dict[atom_idx]  # atom type
                               )

            self.pdbqt_atom_line_list.append(self.pdbqt_atom_line_format.format(*atom_info_tuple))

        with open(self.protein_pdbqt_file_name, 'w') as f:
            for line in self.pdbqt_atom_line_list:
                f.write(line)

        self.next_atom_idx = atom_idx + 1
