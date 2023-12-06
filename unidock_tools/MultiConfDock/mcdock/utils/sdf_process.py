from typing import List, Tuple, Dict
from pathlib import Path
import logging
import rdkit
from rdkit import Chem


def sdf_writer(
    mols:List[rdkit.Chem.rdchem.Mol], 
    output_file:Path
):
    # logging.info(f"write SDF text into {output_file}")
    writer = Chem.SDWriter(str(output_file))
    for mol in mols: writer.write(mol)
    writer.flush()
    writer.close()


def split_sdf(
    ligand:Path,
    savedir:Path=None,
    remove_original:bool=True
) -> List[Path]:
    prefix = Path(ligand).stem
    suffix = Path(ligand).suffix
    if suffix != ".sdf":
        logging.error(f"ligand {ligand} is not in SDF format")
        return []
    if savedir is None: savedir = makedirs("split_sdf")
    mollist = [mol for mol in Chem.SDMolSupplier(
        str(ligand), removeHs=False) if mol is not None]
    if len(mollist) == 0:
        logging.error(f"No molecule in {ligand}")
        return []
    if len(mollist) == 1:
        sdf_writer(mollist, Path(f"{savedir}/{prefix}.sdf"))
        if remove_original: ligand.unlink()
        return [Path(f"{savedir}/{prefix}.sdf")]
    for idx, mol in enumerate(mollist):
        sdf_writer([mol], Path(f"{savedir}/{prefix}_{idx}.sdf"))
    if remove_original: ligand.unlink()
    return [Path(f"{savedir}/{prefix}_{idx}.sdf") for idx in range(len(mollist))]


def concat_sdf(
    ligands:List[Path],
    output:Path,
    remove_original:bool=True
) -> Path:
    mollist = []
    for ligand in ligands:
        mollist += [mol for mol in Chem.SDMolSupplier(
            str(ligand), removeHs=False) if mol is not None]
        if remove_original: ligand.unlink()
    sdf_writer(mollist, output)
    return output


def remove_props_in_sdf(
    sdf:Path,
    props:List[str],
    output:Path=None
) -> Tuple[Path, Dict[str, str]]:
    if output is None: output = sdf
    mollist = []; prop_dict ={}
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
        if mol is None: continue
        for prop in props: 
            prop_dict[prop] = prop_dict.get(prop, []) \
                + [mol.GetProp(prop)]
            mol.ClearProp(prop)
        mollist.append(mol)
    sdf_writer(mollist, output)
    return output, prop_dict


def add_props_in_sdf(
    sdf:Path,
    prop_dict:Dict,
    output:Path=None
) -> Path:
    if output is None: output = sdf
    mollist = []
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
        if mol is None: continue
        for prop, values in prop_dict.items():
            if isinstance(values, list): mol.SetProp(prop, values.pop(0))
            else: mol.SetProp(prop, values)
        mollist.append(mol)
    sdf_writer(mollist, output)
    return output

