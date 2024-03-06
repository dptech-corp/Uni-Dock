from typing import List
from pathlib import Path
import os
import time
import shutil
import argparse
import logging

from unidock_tools.utils import time_logger, randstr
from unidock_tools.modules.confgen import generate_conf
from .unidock_pipeline import UniDock


DEFAULT_ARGS = {
    "receptor": None,
    "ligands": None,
    "ligand_index": None,
    "gen_conf": False,
    "max_num_confs_per_ligand": 200,
    "min_rmsd": 0.3,
    "center_x": None,
    "center_y": None,
    "center_z": None,
    "size_x": 22.5,
    "size_y": 22.5,
    "size_z": 22.5,
    "workdir": "mcdock_workdir",
    "savedir": "mcdock_results",
    "batch_size": 1200,
    "scoring_function_rigid_docking": "vina",
    "search_mode_rigid_docking": "",
    "exhaustiveness_rigid_docking": 128,
    "max_step_rigid_docking": 20,
    "num_modes_rigid_docking": 3,
    "refine_step_rigid_docking": 3,
    "topn_rigid_docking": 100,
    "scoring_function_local_refine": "vina",
    "search_mode_local_refine": "",
    "exhaustiveness_local_refine": 512,
    "max_step_local_refine": 40,
    "num_modes_local_refine": 1,
    "refine_step_local_refine": 3,
    "topn_local_refine": 1,
    "seed": 181129,
    "debug": False,
}


class MultiConfDock(UniDock):
    def __init__(self,
                 receptor: Path,
                 ligands: List[Path],
                 center_x: float,
                 center_y: float,
                 center_z: float,
                 size_x: float = 22.5,
                 size_y: float = 22.5,
                 size_z: float = 22.5,
                 gen_conf: bool = True,
                 max_nconf: int = 1000,
                 min_rmsd: float = 0.5,
                 workdir: Path = Path("UniDock"),
                 ):
        """
        Initializes a MultiConfDock object.

        Args:
            receptor (Path): Path to the receptor file in pdbqt format.
            ligands (List[Path]): List of paths to the ligand files in sdf format.
            center_x (float): X-coordinate of the center of the docking box.
            center_y (float): Y-coordinate of the center of the docking box.
            center_z (float): Z-coordinate of the center of the docking box.
            size_x (float, optional): Size of the docking box in the x-dimension. Defaults to 22.5.
            size_y (float, optional): Size of the docking box in the y-dimension. Defaults to 22.5.
            size_z (float, optional): Size of the docking box in the z-dimension. Defaults to 22.5.
            workdir (Path, optional): Path to the working directory. Defaults to Path("MultiConfDock").
        """
        super(MultiConfDock, self).__init__(receptor=receptor, ligands=ligands,
                                            center_x=center_x, center_y=center_y, center_z=center_z,
                                            size_x=size_x, size_y=size_y, size_z=size_z,
                                            workdir=workdir)
        self.check_dependencies()
        if gen_conf:
            self.generate_conformation(max_nconf=max_nconf, min_rmsd=min_rmsd)

    def check_dependencies(self):
        """
        Checks if all dependencies are installed.
        """
        if not shutil.which("confgen") and not shutil.which("obabel"):
            raise ModuleNotFoundError("To run MultiConfDock, you need to install CDPKit confgen or OpenBabel")

    @time_logger
    def generate_conformation(self,
                              max_nconf: int = 1000,
                              min_rmsd: float = 0.5):
        for idx, mol in enumerate(self.mol_group):
            prefix = mol.get_prop("file_prefix", "")
            gen_mol_confs = generate_conf(mol.get_first_mol(), name=prefix,
                                          max_num_confs_per_ligand=max_nconf,
                                          min_rmsd=min_rmsd)
            self.mol_group.update_mol_confs(idx, gen_mol_confs)
            self.mol_group.write_sdf(save_dir=f"{self.workdir}/confgen_results", seperate_conf=False)


def main(args: dict):
    args = {**DEFAULT_ARGS, **args}
    if args["debug"]:
        logging.getLogger().setLevel("DEBUG")

    workdir = Path(args["workdir"]).resolve()
    savedir = Path(args["savedir"]).resolve()

    ligands = []
    if args.get("ligands"):
        for lig in args["ligands"]:
            if not Path(lig).exists():
                logging.error(f"Cannot find {lig}")
                continue
            ligands.append(Path(lig).resolve())
    if args.get("ligand_index"):
        with open(args["ligand_index"], "r") as f:
            for line in f.readlines():
                if not Path(line.strip()).exists():
                    logging.error(f"Cannot find {line.strip()}")
                    continue
                ligands.append(Path(line.strip()).resolve())

    if len(ligands) == 0:
        logging.error("No ligands found.")
        exit(1)
    logging.info(f"[MultiConfDock] {len(ligands)} ligands found.")

    logging.info("[MultiConfDock] Start")
    start_time = time.time()
    mcd = MultiConfDock(
        receptor=Path(args["receptor"]).resolve(),
        ligands=ligands,
        center_x=float(args["center_x"]),
        center_y=float(args["center_y"]),
        center_z=float(args["center_z"]),
        size_x=float(args["size_x"]),
        size_y=float(args["size_y"]),
        size_z=float(args["size_z"]),
        gen_conf=bool(args["gen_conf"]),
        max_nconf=int(args["max_num_confs_per_ligand"]),
        min_rmsd=float(args["min_rmsd"]),
        workdir=workdir
    )
    logging.info("[MultiConfDock] Start rigid docking")
    mcd.run_unidock(
        scoring_function=str(args["scoring_function_rigid_docking"]),
        search_mode=str(args["search_mode_rigid_docking"]),
        exhaustiveness=int(args["exhaustiveness_rigid_docking"]),
        max_step=int(args["max_step_rigid_docking"]),
        num_modes=int(args["num_modes_rigid_docking"]),
        refine_step=int(args["refine_step_rigid_docking"]),
        seed=args["seed"],
        topn=int(args["topn_rigid_docking"]),
        batch_size=int(args["batch_size"]),
        docking_dir_name="rigid_docking",
        props_list=["fragAllInfo", "atomInfo"],
    )
    logging.info("[MultiConfDock] Start local refine")
    mcd.run_unidock(
        scoring_function=str(args["scoring_function_local_refine"]),
        search_mode=str(args["search_mode_local_refine"]),
        exhaustiveness=int(args["exhaustiveness_local_refine"]),
        max_step=int(args["max_step_local_refine"]),
        num_modes=int(args["num_modes_local_refine"]),
        refine_step=int(args["refine_step_local_refine"]),
        seed=args["seed"],
        topn=int(args["topn_local_refine"]),
        batch_size=int(args["batch_size"]),
        local_only=True,
        docking_dir_name="local_refine_docking",
        props_list=["fragInfo", "torsionInfo", "atomInfo"],
    )
    mcd.save_results(save_dir=savedir)
    end_time = time.time()
    logging.info(f"[MultiConfDock] Workflow finished ({end_time - start_time:.2f} s)")
    if not args["debug"]:
        shutil.rmtree(workdir, ignore_errors=True)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MultiConfDock")

    parser.add_argument("-r", "--receptor", type=str, required=True,
                        help="Receptor file in pdbqt format.")
    parser.add_argument("-l", "--ligands", type=lambda s: s.split(','), default=None,
                        help="Ligand file in sdf format. Specify multiple files separated by commas.")
    parser.add_argument("-i", "--ligand_index", type=str, default=None,
                        help="A text file containing the path of ligand files in sdf format.")

    parser.add_argument("-g", "--gen_conf", action="store_true",
                        help="Whether to generate conformers for the ligands. Default: False.")
    parser.add_argument("-n", "--max_num_confs_per_ligand", type=int, default=200,
                        help="Maximum number of conformers to generate for each ligand. Default: 200.")
    parser.add_argument("-m", "--min_rmsd", type=float, default=0.3,
                        help="Minimum rmsd for conformer generation. Default: 0.3.")

    parser.add_argument("-cx", "--center_x", type=float, required=True,
                        help="X-coordinate of the docking box center.")
    parser.add_argument("-cy", "--center_y", type=float, required=True,
                        help="Y-coordinate of the docking box center.")
    parser.add_argument("-cz", "--center_z", type=float, required=True,
                        help="Z-coordinate of the docking box center.")
    parser.add_argument("-sx", "--size_x", type=float, default=22.5,
                        help="Width of the docking box in X direction. Default: 22.5.")
    parser.add_argument("-sy", "--size_y", type=float, default=22.5,
                        help="Width of the docking box in Y direction. Default: 22.5.")
    parser.add_argument("-sz", "--size_z", type=float, default=22.5,
                        help="Width of the docking box in Z direction. Default: 22.5.")

    parser.add_argument("-wd", "--workdir", type=str, default=f"mcdock_{randstr(5)}",
                        help="Working directory. Default: 'MultiConfDock'.")
    parser.add_argument("-sd", "--savedir", type=str, default="mcdock_results",
                        help="Save directory. Default: 'MultiConfDock-Result'.")
    parser.add_argument("-bs", "--batch_size", type=int, default=1200,
                        help="Batch size for docking. Default: 1200.")

    parser.add_argument("-sf_rd", "--scoring_function_rigid_docking",
                        type=str, default="vina",
                        help="Scoring function used in rigid docking. Default: 'vina'.")
    parser.add_argument("-sm_rd", "--search_mode_rigid_docking",
                        type=str, default="",
                        help="Search mode used in rigid docking. Default: <empty string>.")
    parser.add_argument("-ex_rd", "--exhaustiveness_rigid_docking",
                        type=int, default=128,
                        help="Exhaustiveness used in rigid docking. Default: 128.")
    parser.add_argument("-ms_rd", "--max_step_rigid_docking",
                        type=int, default=20,
                        help="Max step used in rigid docking. Default: 20.")
    parser.add_argument("-nm_rd", "--num_modes_rigid_docking",
                        type=int, default=3,
                        help="Number of modes used in rigid docking. Default: 3.")
    parser.add_argument("-rs_rd", "--refine_step_rigid_docking",
                        type=int, default=3,
                        help="Refine step used in rigid docking. Default: 3.")
    parser.add_argument("-topn_rd", "--topn_rigid_docking",
                        type=int, default=100,
                        help="Top N results used in rigid docking. Default: 100.")

    parser.add_argument("-sf_lr", "--scoring_function_local_refine",
                        type=str, default="vina",
                        help="Scoring function used in local refine. Default: 'vina'.")
    parser.add_argument("-sm_lr", "--search_mode_local_refine",
                        type=str, default="",
                        help="Search mode used in local refine. Default <empty string>.")
    parser.add_argument("-ex_lr", "--exhaustiveness_local_refine",
                        type=int, default=512,
                        help="Exhaustiveness used in rigid docking. Default: 512.")
    parser.add_argument("-ms_lr", "--max_step_local_refine",
                        type=int, default=40,
                        help="Max step used in rigid docking. Default: 40.")
    parser.add_argument("-nm_lr", "--num_modes_local_refine",
                        type=int, default=1,
                        help="Number of modes used in local refine. Default: 1.")
    parser.add_argument("-rs_lr", "--refine_step_local_refine",
                        type=int, default=3,
                        help="Refine step used in local refine. Default: 3.")
    parser.add_argument("-topn_lr", "--topn_local_refine",
                        type=int, default=1,
                        help="Top N results used in local refine. Default: 1.")

    parser.add_argument("--seed", type=int, default=181129, 
                        help="Uni-Dock random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode")
    return parser


def main_cli():
    """
    Command line interface for MultiConfDock.

    Input files:
    -r, --receptor: receptor file in pdbqt format
    -l, --ligands: ligand file in sdf format, separated by commas(,)
    -i, --ligand_index: a text file containing the path of ligand files in sdf format

    ConfGen arguments:
    -g, --gen_conf: whether to generate conformers for the ligands
    -n, --max_num_confs_per_ligand: maximum number of conformers to generate for each ligand (default: 200)
    -m, --min_rmsd: minimum rmsd for conformer generation (default: 0.3)

    Docking box:
    -cx, --center_x: center_x of docking box
    -cy, --center_y: center_y of docking box
    -cz, --center_z: center_z of docking box
    -sx, --size_x: size_x of docking box (default: 22.5)
    -sy, --size_y: size_y of docking box (default: 22.5)
    -sz, --size_z: size_z of docking box (default: 22.5)

    Optional arguments:
    -wd, --workdir: working directory (default: MultiConfDock)
    -sd, --savedir: save directory (default: MultiConfDock-Result)
    -bs, --batch_size: batch size for docking (default: 20)

    Scoring function and search mode for rigid docking:
    -sf_rd, --scoring_function_rigid_docking: scoring function used in rigid docking (default: vina)
    -ex_rd, --exhaustiveness_rigid_docking: exhaustiveness used in rigid docking (default: 256)
    -ms_rd, --max_step_rigid_docking: maxstep used in rigid docking (default: 10)
    -nm_rd, --num_modes_rigid_docking: num_modes used in rigid docking (default: 3)
    -rs_rd, --refine_step_rigid_docking: refine_step used in rigid docking (default: 3)
    -topn_rd, --topn_rigid_docking: topn used in rigid docking (default: 200)

    Scoring function and search mode for local refine:
    -sf_lr, --scoring_function_local_refine: scoring function used in local refine (default: vina)
    -ex_lr, --exhaustiveness_local_refine: exhaustiveness used in local refine (default: 32)
    -ms_lr, --max_step_local_refine: maxstep used in local refine (default: 40)
    -nm_lr, --num_modes_local_refine: num_modes used in local refine (default: 3)
    -rs_lr, --refine_step_local_refine: refine_step used in local refine (default: 5)
    -topn_lr, --topn_local_refine: topn used in local refine (default: 100)
    """
    parser = get_parser()
    args = parser.parse_args().__dict__
    logging.info(f"[Params] {args}")
    main(args)


if __name__ == "__main__":
    main_cli()
