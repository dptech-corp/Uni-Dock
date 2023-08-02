from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem.rdMolAlign import AlignMol

def get_mol_without_indices(mol_input,
                            remove_indices=[],
                            keep_properties=[],
                            keep_mol_properties=[]):

    mol_property_dict = {}
    for mol_property_name in keep_mol_properties:
        mol_property_dict[mol_property_name] = mol_input.GetProp(mol_property_name)

    atom_list, bond_list, idx_map = [], [], {}  # idx_map: {old: new}

    for atom in mol_input.GetAtoms():

        props = {}
        for property_name in keep_properties:
            if property_name in atom.GetPropsAsDict():
                props[property_name] = atom.GetPropsAsDict()[property_name]

        symbol = atom.GetSymbol()

        if symbol.startswith('*'):
            atom_symbol = '*'
            props['molAtomMapNumber'] = atom.GetAtomMapNum()

        elif symbol.startswith('R'):
            atom_symbol = '*'
            if len(symbol) > 1:
                atom_map_num = int(symbol[1:])
            else:
                atom_map_num = atom.GetAtomMapNum()
            props['molAtomMapNumber'] = atom_map_num
            props['dummyLabel'] = 'R' + str(atom_map_num)
            props['_MolFileRLabel'] = str(atom_map_num)

        else:
            atom_symbol = symbol

        atom_list.append(
            (
                atom_symbol,
                atom.GetChiralTag(),
                atom.GetFormalCharge(),
                atom.GetNumExplicitHs(),
                props
            )
        )

    for bond in mol_input.GetBonds():
        bond_list.append(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType()
            )
        )

    mol = Chem.RWMol(Chem.Mol())

    new_idx = 0
    for atom_index, atom_info in enumerate(atom_list):
        if atom_index not in remove_indices:
            atom = Chem.Atom(atom_info[0])
            atom.SetChiralTag(atom_info[1])
            atom.SetFormalCharge(atom_info[2])
            atom.SetNumExplicitHs(atom_info[3])

            for property_name in atom_info[4]:
                if isinstance(atom_info[4][property_name], str):
                    atom.SetProp(property_name, atom_info[4][property_name])
                elif isinstance(atom_info[4][property_name], int):
                    atom.SetIntProp(property_name, atom_info[4][property_name])

            mol.AddAtom(atom)
            idx_map[atom_index] = new_idx
            new_idx += 1

    for bond_info in bond_list:
        if (
            bond_info[0] not in remove_indices
            and bond_info[1] not in remove_indices
        ):
            mol.AddBond(
                idx_map[bond_info[0]],
                idx_map[bond_info[1]],
                bond_info[2]
            )

        else:
            one_in = False
            if (
                (bond_info[0] in remove_indices)
                and (bond_info[1] not in remove_indices)
            ):
                keep_index = bond_info[1]
                remove_index = bond_info[0]
                one_in = True
            elif (
                (bond_info[1] in remove_indices)
                and (bond_info[0] not in remove_indices)
            ):
                keep_index = bond_info[0]
                remove_index = bond_info[1]
                one_in = True
            if one_in:
                if atom_list[keep_index][0] in ['N', 'P']:
                    old_num_explicit_Hs = mol.GetAtomWithIdx(
                        idx_map[keep_index]
                    ).GetNumExplicitHs()

                    mol.GetAtomWithIdx(idx_map[keep_index]).SetNumExplicitHs(
                        old_num_explicit_Hs + 1
                    )

    mol = Chem.Mol(mol)

    for mol_property_name in mol_property_dict:
        mol.SetProp(mol_property_name, mol_property_dict[mol_property_name])

    Chem.GetSymmSSSR(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol

def assign_atom_properties(mol):
    atom_positions = mol.GetConformer().GetPositions()
    num_atoms = mol.GetNumAtoms()

    internal_atom_idx = 0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetIntProp('sdf_atom_idx', atom_idx+1)
        if not atom.HasProp('atom_name'):
            atom_element = atom.GetSymbol()
            atom_name = atom_element + str(internal_atom_idx+1)
            atom.SetProp('atom_name', atom_name)
            atom.SetProp('residue_name', 'MOL')
            atom.SetIntProp('residue_idx', 1)
            atom.SetProp('chain_idx', 'A')
            internal_atom_idx += 1

        atom.SetDoubleProp('charge', atom.GetDoubleProp('_GasteigerCharge'))
        atom.SetDoubleProp('x', atom_positions[atom_idx, 0])
        atom.SetDoubleProp('y', atom_positions[atom_idx, 1])
        atom.SetDoubleProp('z', atom_positions[atom_idx, 2])
