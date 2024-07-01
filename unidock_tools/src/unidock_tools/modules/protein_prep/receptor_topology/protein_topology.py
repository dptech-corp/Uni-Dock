import MDAnalysis as mda
import openmm.app as app

from rdkit.Chem import SplitMolByPDBResidues, GetMolFrags, FragmentOnBonds
from rdkit.Chem.PropertyMol import PropertyMol

def is_peptide_bond(bond):
    """Checks if a bond is a peptide bond based on the residue_id and chain_id of the atoms
    on each part of the bond. Also works for disulfide bridges or any bond that
    links two residues in biopolymers.

    Parameters
    ----------
    bond: rdkit.Chem.rdchem.Bond
        The bond to check
    """

    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    begin_residue_idx = begin_atom.GetIntProp('internal_residue_idx')
    end_residue_idx = end_atom.GetIntProp('internal_residue_idx')

    begin_chain_idx = begin_atom.GetProp('chain_idx')
    end_chain_idx = end_atom.GetProp('chain_idx')

    if begin_residue_idx == end_residue_idx and begin_chain_idx == end_chain_idx:
        return False
    else:
        return True

def split_mol_by_residues(protein_mol):
    """Splits a protein_mol in multiple fragments based on residues

    Parameters
    ----------
    protein_mol: rdkit.Chem.rdchem.Mol
        The protein molecule to fragment

    Returns
    -------
    residue_mol_list : list
        A list of :class:`rdkit.Chem.rdchem.Mol` containing sorted residues of protein molecule
    """

    protein_residue_mol_list = []
    for residue_type_fragments in SplitMolByPDBResidues(protein_mol).values():
        for fragment in GetMolFrags(residue_type_fragments, asMols=True, sanitizeFrags=False):
            # split on peptide bonds
            peptide_bond_idx_list = []
            for bond in fragment.GetBonds():
                if is_peptide_bond(bond):
                    peptide_bond_idx_list.append(bond.GetIdx())

            if len(peptide_bond_idx_list) > 0:
                splitted_mol = FragmentOnBonds(fragment, peptide_bond_idx_list, addDummies=False)
                splitted_mol_list = GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False)
                protein_residue_mol_list.extend(splitted_mol_list)
            else:
                protein_residue_mol_list.append(fragment)

    protein_residue_mol_dict = {}
    for protein_residue_mol in protein_residue_mol_list:
        atom = protein_residue_mol.GetAtomWithIdx(0)
        if atom.GetSymbol() == 'H': 
            continue

        protein_residue_mol_dict[atom.GetIntProp('internal_residue_idx')] = protein_residue_mol

    return [x[1] for x in sorted(protein_residue_mol_dict.items(), key=lambda x: x[0])]

def prepare_protein_residue_mol_list(protein_pdb_file_name):
    ########################################################################
    ## The protein pdb file should be prepared by ReceptorPDBReader first, with protein part only.
    ########################################################################

    original_protein_ag = mda.Universe(protein_pdb_file_name).select_atoms('protein')
    original_num_protein_atoms = original_protein_ag.n_atoms
    original_atom_idx_list = [None] * original_num_protein_atoms
    original_atom_name_list = [None] * original_num_protein_atoms
    original_resid_list = [None] * original_num_protein_atoms
    original_resname_list = [None] * original_num_protein_atoms
    original_chain_idx_list = [None] * original_num_protein_atoms

    for atom_idx in range(original_num_protein_atoms):
        atom = original_protein_ag[atom_idx]
        original_atom_idx_list[atom_idx]  = atom.index + 1
        original_atom_name_list[atom_idx] = atom.name
        original_resid_list[atom_idx] = atom.resid
        original_resname_list[atom_idx] = atom.resname
        original_chain_idx_list[atom_idx] = atom.chainID

    openmm_pdb_file = app.PDBFile(protein_pdb_file_name)
    protein_universe = mda.Universe(openmm_pdb_file)
    protein_universe.add_TopologyAttr('record_type')

    protein_ag = protein_universe.atoms
    for atom in protein_ag:
        atom.record_type = 'ATOM'

    mda_to_rdkit = mda._CONVERTERS['RDKIT']().convert
    protein_mol = mda_to_rdkit(protein_universe)

    protein_positions = protein_mol.GetConformer().GetPositions()
    num_protein_atoms = protein_mol.GetNumAtoms()
    for atom_idx in range(num_protein_atoms):
        atom = protein_mol.GetAtomWithIdx(atom_idx)
        monomer_info = atom.GetMonomerInfo()
        atom_positions = protein_positions[atom_idx, :]

        atom.SetIntProp('atom_idx', int(original_atom_idx_list[atom_idx]))
        atom.SetProp('atom_name', original_atom_name_list[atom_idx])
        atom.SetIntProp('residue_idx', int(original_resid_list[atom_idx]))
        atom.SetProp('residue_name', original_resname_list[atom_idx])
        atom.SetProp('chain_idx', original_chain_idx_list[atom_idx])
        atom.SetIntProp('internal_residue_idx', monomer_info.GetResidueNumber())
        atom.SetDoubleProp('x', float(atom_positions[0]))
        atom.SetDoubleProp('y', float(atom_positions[1]))
        atom.SetDoubleProp('z', float(atom_positions[2]))

    protein_residue_mol_list = split_mol_by_residues(protein_mol)

    protein_property_mol = PropertyMol(protein_mol)
    protein_residue_property_mol_list = [PropertyMol(protein_residue_mol) for protein_residue_mol in protein_residue_mol_list]

    return protein_property_mol, protein_residue_property_mol_list
