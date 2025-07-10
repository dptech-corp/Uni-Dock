import argparse
from unidock_tools.modules.protein_prep.pdb2pdbqt import pdb2pdbqt


def main(args: dict):
    pdb2pdbqt(args["pdb_file"], args["pdbqt_file"])


def get_parser():
    parser = argparse.ArgumentParser(description="Protein PDB to PDBQT Converter")
    parser.add_argument("-r", "--pdb_file", type=str, required=True,
                        help="protein PDB file name")
    parser.add_argument("-o", "--pdbqt_file", type=str, required=True,
                        help="protein PDBQT file name")
    return parser


def main_cli():
    parser = get_parser()
    args = parser.parse_args()
    main(vars(args))


if __name__ == "__main__":
    main_cli()