from typing import Union, List
from pathlib import Path
import os
import logging
import traceback
from rdkit import Chem


def read_smi(smi_file:Union[str, bytes, os.PathLike]) -> List[Chem.Mol]:
    """Read SMILES file and return a list of RDKit Mol objects.

    Args:
        smi_file (Union[str, bytes, os.PathLike]): Input .smi file

    Returns:
        List[Chem.Mol]: Output list of RDKit Mol objects
    """
    with open(smi_file, "r", errors="ignore") as f:
        smi_lines = [line.strip() for line in f.readlines() if line.strip()]
    mols = []
    for line in smi_lines:
        try:
            line_list1 = line.split(" ")
            line_list2 = line.split("\t")
            smi = line_list1[0]
            name = ""
            if len(line_list1) > 1:
                name = line_list1[1]
            elif len(line_list2) > 1:
                name = line_list2[1]
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if name:
                mol.SetProp("_Name", name)
            mols.append(mol)
        except:
            continue
    return mols


def read_ligand(ligand_file:Union[str, bytes, os.PathLike]) -> List[Chem.Mol]:
    """Read ligand file with all common formats and return a list of RDKit Mol objects.

    Args:
        ligand_file (Union[str, bytes, os.PathLike]): Input ligand file

    Returns:
        List[Chem.Mol]: Output list of RDKit Mol objects
    """
    mols = []
    ligand_file = str(ligand_file)
    fmt = os.path.splitext(ligand_file)[-1]
    if fmt == ".sdf":
        mols = [m for m in Chem.SDMolSupplier(ligand_file, removeHs=False, strictParsing=False)]
    elif fmt == ".mol":
        mols = [Chem.MolFromMolFile(ligand_file, removeHs=False, strictParsing=False)]
    elif fmt == ".pdb":
        mols = [Chem.MolFromPDBFile(ligand_file, removeHs=False)]
    elif fmt == ".smi":
        mols = read_smi(ligand_file)
    final_mols = []
    for i, mol in enumerate(mols):
        if not mol:
            logging.warning(f"ligand {str(ligand_file)} idx {i} is invalid mol")
            continue
        mol.SetProp("filename", f"{Path(ligand_file).stem}_{i}" if len(mols) > 1 else Path(ligand_file).stem)
        final_mols.append(mol)
    return final_mols