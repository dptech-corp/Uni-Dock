import argparse
import logging
import shutil
import os
from unidock_tools.modules.protein_prep import receptor_preprocessor

def main(args: dict):
    protein_pdbqt_file_name = receptor_preprocessor(
        protein_pdb_file_name=args['protein_pdb'],
        protein_conf_name='protein_conf_0',
        kept_ligand_resname_list=args['ligand_resname'],
        prepared_hydrogen=args['prepared_hydrogen'],
        preserve_original_resname=args['preserve_resname'],
        target_center=tuple(args['target_center']),
        box_size=tuple(args['box_size']),
        generate_ad4_grids=args['generate_grids'],
        working_dir_name=args['working_dir']
    )

    protein_pdbqt_dst = os.path.join(args['working_dir'], args['protein_pdbqt'])
    shutil.copy(protein_pdbqt_file_name, protein_pdbqt_dst)

def get_parser():
    parser = argparse.ArgumentParser(description="Receptor Preprocessor")
    parser.add_argument("-r", "--protein_pdb", type=str, required=True,
                        help="protein PDB file name")
    parser.add_argument("-l", "--ligand_resname", nargs="+", default=None,
                        help="list of ligand residue names to keep")
    parser.add_argument("-H", "--prepared_hydrogen", action="store_true",
                        help="prepare hydrogen atoms")
    parser.add_argument("-p", "--preserve_resname", action="store_false",
                        help="preserve original residue names")
    parser.add_argument("-c", "--target_center", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help="target center coordinates (x, y, z)")
    parser.add_argument("-s", "--box_size", nargs=3, type=float, default=[22.5, 22.5, 22.5],
                        help="box size")
    parser.add_argument("-g", "--generate_grids", action="store_true",
                        help="generate AD4 grids")
    parser.add_argument("-w", "--working_dir", type=str, default=".",
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