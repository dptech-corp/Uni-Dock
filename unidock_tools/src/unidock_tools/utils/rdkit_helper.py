from typing import List, Union
import os
import logging
import traceback
from rdkit import Chem


def clear_properties(mol: Chem.Mol) -> Chem.Mol:
    """
    Clear all properties from a molecule.

    Args:
        mol (Chem.Mol): The molecule.

    Returns:
        Chem.Mol: The molecule with properties cleared.
    """
    for prop in mol.GetPropsAsDict():
        mol.ClearProp(prop)
    return mol

def set_properties(mol: Chem.Mol, properties: dict):
    """
    Set properties to a molecule.

    Args:
        mol (Chem.Mol): The molecule.
        properties (dict): A dictionary of properties.

    Returns:
        Chem.Mol: The molecule with properties set.
    """
    for key, value in properties.items():
        try:
            if isinstance(value, int):
                mol.SetIntProp(key, value)
            elif isinstance(value, float):
                mol.SetDoubleProp(key, value)
            else:
                mol.SetProp(value, str(value))
        except:
            logging.warning(f"set property {key} err: {traceback.format_exc()}")


def sdf_writer(mols: List[Chem.Mol],
               output_file: Union[str, os.PathLike]):
    with Chem.SDWriter(str(output_file)) as writer:
        for mol in mols:
            writer.write(mol)
