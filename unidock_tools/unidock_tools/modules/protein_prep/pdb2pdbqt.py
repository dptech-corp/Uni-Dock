from typing import Union, List
import os
from math import isnan, isinf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType


def get_pdbqt_atom_lines(mol: Chem.Mol, donors: List[int], acceptors: List[int]):
    atom_lines = [line.replace('HETATM', 'ATOM  ')
                  for line in Chem.MolToPDBBlock(mol).split('\n')
                  if line.startswith('HETATM') or line.startswith('ATOM')]

    pdbqt_lines = []
    for idx, atom in enumerate(mol.GetAtoms()):
        pdbqt_line = atom_lines[idx][:56]

        pdbqt_line += '0.00  0.00    '  # append empty vdW and ele
        # Get charge
        charge = 0.
        fields = ['_MMFF94Charge', '_GasteigerCharge', '_TriposPartialCharge']
        for f in fields:
            if atom.HasProp(f):
                charge = atom.GetDoubleProp(f)
                break
        # FIXME: this should not happen, blame RDKit
        if isnan(charge) or isinf(charge):
            charge = 0.
        pdbqt_line += ('%.3f' % charge).rjust(6)

        # Get atom type
        pdbqt_line += ' '
        atomicnum = atom.GetAtomicNum()
        atomhybridization = atom.GetHybridization()
        atombondsnum = atom.GetDegree()
        if atomicnum == 6 and atom.GetIsAromatic():
            pdbqt_line += 'A '
        elif atomicnum == 7 and idx in acceptors:
            pdbqt_line += 'NA'
        elif atomicnum == 8 and idx in acceptors:
            pdbqt_line += 'OA'
        elif atomicnum == 1 and atom.GetNeighbors()[0].GetIdx() in donors:
            pdbqt_line += 'HD'
        elif atomicnum == 1 and atom.GetNeighbors()[0].GetIdx() not in donors:
            pdbqt_line += 'H '
        elif atomicnum == 16 and (
                (atomhybridization == Chem.HybridizationType.SP3 and atombondsnum != 4) or
                atomhybridization == Chem.HybridizationType.SP2):
            pdbqt_line += 'SA'
        else:
            if len(atom.GetSymbol()) > 1:
                pdbqt_line += atom.GetSymbol()
            else:
                pdbqt_line += (atom.GetSymbol() + ' ')
        pdbqt_lines.append(pdbqt_line)
    return pdbqt_lines


def receptor_mol_to_pdbqt_str(mol: Chem.Mol):
    # make a copy of molecule
    mol = Chem.Mol(mol)

    # Identify donors and acceptors for atom typing
    # Acceptors
    patt = Chem.MolFromSmarts('[$([O;H1;v2]),'
                              '$([O;H0;v2;!$(O=N-*),'
                              '$([O;-;!$(*-N=O)]),'
                              '$([o;+0])]),'
                              '$([n;+0;!X3;!$([n;H1](cc)cc),'
                              '$([$([N;H0]#[C&v4])]),'
                              '$([N&v3;H0;$(Nc)])]),'
                              '$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]')
    acceptors = list(map(lambda x: x[0], mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms())))
    # Donors
    patt = Chem.MolFromSmarts('[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),'
                              '$([$(n[n;H1]),'
                              '$(nc[n;H1])])]),'
                              # Guanidine can be tautormeic - e.g. Arginine
                              '$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),'
                              '$([O,S;H1;+0])]')
    donors = list(map(lambda x: x[0], mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms())))
    AllChem.ComputeGasteigerCharges(mol)

    atom_lines = get_pdbqt_atom_lines(mol, donors, acceptors)
    assert len(atom_lines) == mol.GetNumAtoms()

    pdbqt_lines = ['REMARK  Name = ' + (mol.GetProp('_Name') if mol.HasProp('_Name') else '')]
    pdbqt_lines.extend(atom_lines)

    return "\n".join(pdbqt_lines)


def pdb2pdbqt(pdbfile: Union[str, os.PathLike], out_file: Union[str, os.PathLike]):
    receptor_mol = Chem.MolFromPDBFile(str(pdbfile), removeHs=False)
    pdbqt_str = receptor_mol_to_pdbqt_str(receptor_mol)
    with open(out_file, 'w') as f:
        f.write(pdbqt_str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script to convert receptor from PDB to PDBQT format")
    parser.add_argument("-r", "--receptor_file", type=str, required=True,
                        help="receptor file in PDB format")
    parser.add_argument("-o", "--out_file", type=str, required=True,
                        help="output file in PDBQT format")
    args = parser.parse_args()

    pdb2pdbqt(args.receptor_file, args.out_file)
