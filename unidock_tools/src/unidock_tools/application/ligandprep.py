from typing import List, Tuple, Generator
from pathlib import Path
import os
from functools import partial
from multiprocessing import Pool
import traceback
import logging
import argparse
from rdkit import Chem

from unidock_tools.utils import read_ligand
from unidock_tools.modules.ligand_prep import TopologyBuilder


def iter_ligands(ligands: List[Path], batch_size: int = 1200,
                 use_file_name: bool = False) -> Generator[List[Tuple[Chem.Mol, str]], None, None]:
    curr_mol_name_list = []
    for ligand in ligands:
        mols = read_ligand(ligand)
        for i, mol in enumerate(mols):
            if mol:
                if not use_file_name and mol.HasProp("_Name") and mol.GetProp("_Name").strip():
                    name = mol.GetProp("_Name").strip()
                else:
                    name = f"{ligand.stem}_{i}" if len(mols) > 1 else ligand.stem
                curr_mol_name_list.append((mol, name))
                if len(curr_mol_name_list) > batch_size:
                    yield curr_mol_name_list
                    curr_mol_name_list = []
            else:
                logging.warning(f"read ligand file {ligand.stem} ind {i} error")
    if len(curr_mol_name_list) > 0:
        yield curr_mol_name_list
    return


def ligprep(mol_name_tup: Tuple[Chem.Mol, str], savedir: Path, save_format: str = "sdf"):
    mol, name = mol_name_tup
    try:
        tb = TopologyBuilder(mol)
        tb.build_molecular_graph()
        if save_format == "sdf":
            tb.write_sdf_file(os.path.join(savedir, f"{name}.sdf"))
        elif save_format == "pdbqt":
            tb.write_pdbqt_file(os.path.join(savedir, f"{name}.pdbqt"))
        else:
            logging.error(f"Invalid save format: {save_format}")
    except:
        logging.error(f"ligprep failed for {name}: {traceback.format_exc()}")


def main(args: dict):
    ligands = []
    if args["ligands"]:
        for lig in args["ligands"]:
            if not Path(lig).exists():
                logging.error(f"Cannot find {lig}")
                continue
            ligands.append(Path(lig).resolve())
    if args["ligand_index"]:
        with open(args["ligand_index"], "r") as f:
            for line in f.readlines():
                if not Path(line.strip()).exists():
                    logging.error(f"Cannot find {line.strip()}")
                    continue
                ligands.append(Path(line.strip()).resolve())

    os.makedirs(Path(args["savedir"]).resolve(), exist_ok=True)
    for mol_name_tup_list in iter_ligands(ligands, args["batch_size"], args["use_file_name"]):
        with Pool(os.cpu_count()) as pool:
            pool.map(partial(ligprep, savedir=args["savedir"], 
                             save_format=args["save_format"]), mol_name_tup_list)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--ligands", type=lambda s: s.split(','), default=None,
                        help="Ligand file in sdf format. Specify multiple files separated by commas.")
    parser.add_argument("-i", "--ligand_index", type=str, default="",
                        help="A text file containing the path of ligand files in sdf format.")
    parser.add_argument("-sd", "--savedir", type=str, default="ligprep_results",
                        help="Save directory. Default: 'MultiConfDock-Result'.")
    parser.add_argument("-sf", "--save_format", type=str, default="sdf", 
                        help="Ligprep result files format. Choose from ['sdf', 'pdbqt']. Default: 'sdf'.")
    parser.add_argument("-bs", "--batch_size", type=int, default=1200,
                        help="Batch size for docking. Default: 1200.")
    parser.add_argument("-ufn", "--use_file_name", action="store_true",
                        help="use file name for output sdf file, instead of molecule name.")
    return parser


def main_cli():
    parser = get_parser()
    args = parser.parse_args().__dict__
    logging.info(f"[Params] {args}")
    main(args)


if __name__ == "__main__":
    main_cli()
