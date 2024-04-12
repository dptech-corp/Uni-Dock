import argparse
import logging
import shutil
import os
from unidock_tools.modules.protein_prep import receptor_preprocessor
from typing import List, Tuple, Dict, Optional

def main(args: dict):
    
    def parse_covalent_residue_atom_info(covalent_residue_atom_info_str: str) -> List[List[Tuple[str, str, int, str]]]:
        residue_info_list = []
        residue_atoms = covalent_residue_atom_info_str.split(',')
        for residue_atom in residue_atoms:
            residue_info = residue_atom.strip().split()
            chain_id, residue_name, residue_number, atom_name = residue_info
            residue_info_list.append((chain_id, residue_name, int(residue_number), atom_name))
        return residue_info_list
    
    protein_pdbqt_file_name = receptor_preprocessor(
        protein_pdb_file_name=args['protein_pdb'],
        protein_conf_name='protein_conf_0',
        kept_ligand_resname_list=args['kept_ligand_resname_list'],
        prepared_hydrogen=args['prepared_hydrogen'],
        preserve_original_resname=args['preserve_resname'],
        target_center=tuple(args['target_center']),
        box_size=tuple(args['box_size']),
        generate_ad4_grids=args['generate_grids'],
        covalent_residue_atom_info_list = parse_covalent_residue_atom_info(args['covalent_residue_atom_info']) if args['covalent_residue_atom_info'] is not None else None,
        working_dir_name=args['working_dir']
    )

    protein_pdbqt_dst = os.path.join(args['working_dir'], args['protein_pdbqt'])
    shutil.copy(protein_pdbqt_file_name, protein_pdbqt_dst)

def get_parser():
    parser = argparse.ArgumentParser(description="Receptor Preprocessor")
    parser.add_argument("-r", "--protein_pdb", type=str, required=True,
                        help="protein PDB file name")
    parser.add_argument("-kr", "--kept_ligand_resname_list", type=str, nargs="+", default=None,
                        help="list of ligand residue names to keep. To use it like this: -kr Lig1 Lig2 ")
    parser.add_argument("-ph", "--prepared_hydrogen", action="store_false",
                        help="prepare hydrogen atoms")
    parser.add_argument("-pr", "--preserve_resname", action="store_false",
                        help="preserve original residue names")
    parser.add_argument("-c", "--target_center", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="target center coordinates (x, y, z)")
    parser.add_argument("-s", "--box_size", nargs=3, type=float, default=[22.5, 22.5, 22.5],
                        help="box size")
    parser.add_argument("-g", "--generate_grids", action="store_true",
                        help="generate AD4 grids")
    parser.add_argument("-cra", "--covalent_residue_atom_info", type=str, default=None,
                        help="Atom information for covalent residues during receptor preprocessing. To use it like this: -cra 'A VAL 1 CA, A VAL 1 CB, A VAL 1 O'((chain_id, residue_name, residue_number, atom_name)")
    parser.add_argument("-wd", "--working_dir", type=str, default=".",
                        help="working directory")
    parser.add_argument("-o", "--protein_pdbqt", type=str, required=True,
                        help="protein PDBQT file name")
    return parser

def main_cli():
    parser = get_parser()
    args = vars(parser.parse_args())

    logging.info(f"Running receptor_preprocessor with args: {args}")
    main(args)

if __name__ == "__main__":
    main_cli()