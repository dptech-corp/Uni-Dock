import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from atom_type import AtomType
#from nashtools.modules.docking_engines.ligand_topology import utils

class SmallMoleculePDBQTWriter(object):
    def __init__(self,
                 mol,
                 start_atom_idx,
                 working_dir_name='.'):

        self.mol = mol
        self.start_atom_idx = start_atom_idx
        self.working_dir_name = os.path.abspath(working_dir_name)

        self.residue_name = self.mol.GetProp('residue_name')
        self.molecule_pdbqt_file_name = os.path.join(self.working_dir_name, f'{self.residue_name}.pdbqt')
        self.atom_typer = AtomType()

    def __assign_atom_properties__(self):
        if self.residue_name in ['HOH']:
            self.__water_special_treatments__()

        else:
            ComputeGasteigerCharges(self.mol)
            self.atom_typer.assign_atom_types(self.mol)

            num_atoms = self.mol.GetNumAtoms()
            for atom_idx in range(num_atoms):
                atom = self.mol.GetAtomWithIdx(atom_idx)
                atom.SetDoubleProp('charge', atom.GetDoubleProp('_GasteigerCharge'))

    def __water_special_treatments__(self):
        mol_h = Chem.AddHs(self.mol, addCoords=True)

        reordered_atom_idx_list = []
        for atom in mol_h.GetAtoms():
            if atom.GetSymbol() == 'O':
                bonded_hydrogen_atom_list = list(atom.GetNeighbors())
                if len(bonded_hydrogen_atom_list) != 2:
                    raise ValueError('Problematic water molecule!!')

                hydrogen_atom_1 = bonded_hydrogen_atom_list[0]
                hydrogen_atom_2 = bonded_hydrogen_atom_list[1]

                atom.SetProp('atom_name', 'O1')
                atom.SetDoubleProp('charge', -0.834)
                atom.SetProp('atom_type', 'OA')

                residue_idx = atom.GetIntProp('residue_idx')
                residue_name = atom.GetProp('residue_name')
                chain_idx = atom.GetProp('chain_idx')

                hydrogen_atom_1.SetProp('atom_name', 'H1')
                hydrogen_atom_1.SetIntProp('residue_idx', residue_idx)
                hydrogen_atom_1.SetProp('residue_name', residue_name)
                hydrogen_atom_1.SetProp('chain_idx', chain_idx)
                hydrogen_atom_1.SetDoubleProp('charge', 0.417)
                hydrogen_atom_1.SetProp('atom_type', 'HD')

                hydrogen_atom_2.SetProp('atom_name', 'H2')
                hydrogen_atom_2.SetIntProp('residue_idx', residue_idx)
                hydrogen_atom_2.SetProp('residue_name', residue_name)
                hydrogen_atom_2.SetProp('chain_idx', chain_idx)
                hydrogen_atom_2.SetDoubleProp('charge', 0.417)
                hydrogen_atom_2.SetProp('atom_type', 'HD')

                oxygen_atom_idx = atom.GetIdx()
                hydrogen_atom_idx_1 = hydrogen_atom_1.GetIdx()
                hydrogen_atom_idx_2 = hydrogen_atom_2.GetIdx()

                reordered_atom_idx_list.extend([oxygen_atom_idx, hydrogen_atom_idx_1, hydrogen_atom_idx_2])

        self.mol = Chem.RenumberAtoms(mol_h, reordered_atom_idx_list)

    def write_small_molecule_pdbqt_file(self):
        self.__assign_atom_properties__()

        self.pdbqt_atom_line_list = []
        self.pdbqt_atom_line_format = '{:4s}  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}\n'

        current_atom_idx = self.start_atom_idx
        num_atoms = self.mol.GetNumAtoms()

        molecule_positions = self.mol.GetConformer().GetPositions()

        for atom_idx in range(num_atoms):
            atom = self.mol.GetAtomWithIdx(atom_idx)
            atom_positions = molecule_positions[atom_idx, :]

            atom_info_tuple = ('ATOM',  # atom record name
                               current_atom_idx,  # atom serial number
                               atom.GetProp('atom_name'),  # atom name
                               atom.GetProp('residue_name'),  # residue name
                               atom.GetProp('chain_idx'),  # chain identifier
                               atom.GetIntProp('residue_idx'),  # residue sequence number
                               float(atom_positions[0]),  # x coordinate
                               float(atom_positions[1]),  # y coordinate
                               float(atom_positions[2]),  # z coordinate
                               1.0,  # occupancy
                               0.0,  # temperature factor
                               atom.GetDoubleProp('charge'),  # partial charge
                               atom.GetProp('atom_type')  # atom type
                               )

            self.pdbqt_atom_line_list.append(self.pdbqt_atom_line_format.format(*atom_info_tuple))

            current_atom_idx += 1

        with open(self.molecule_pdbqt_file_name, 'w') as f:
            for line in self.pdbqt_atom_line_list:
                f.write(line)

        self.next_atom_idx = current_atom_idx
