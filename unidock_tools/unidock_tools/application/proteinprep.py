import logging
import argparse
from unidock_tools.modules.protein_prep import pdb2pdbqt


def main(args: dict):
    pdb2pdbqt(args["receptor_file"], args["output_file"])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--receptor_file", type=str, required=True,
                        help="Input receptor file in PDB format")
    parser.add_argument("-o", "--output_file", type=str, default="output.pdbqt",
                        help="Output file in PDBQT format")
    return parser


def main_cli():
    parser = get_parser()
    args = parser.parse_args().__dict__
    logging.info(f"[Params] {args}")
    main(args)


if __name__ == "__main__":
    main_cli()
