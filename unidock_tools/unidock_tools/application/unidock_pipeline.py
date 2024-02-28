from typing import List, Tuple, Union, Iterator
import copy
from pathlib import Path
import os
import time
import shutil
import argparse
import logging
from multiprocessing import Pool
from rdkit import Chem

from unidock_tools.utils import time_logger, randstr, make_tmp_dir, MolGroup
from unidock_tools.modules.protein_prep import pdb2pdbqt
from unidock_tools.modules.ligand_prep import TopologyBuilder
from unidock_tools.modules.docking import run_unidock
from .base import Base


class UniDock(Base):
    def __init__(self,
                 receptor: Path,
                 ligands: List[Path],
                 center_x: float,
                 center_y: float,
                 center_z: float,
                 size_x: float = 22.5,
                 size_y: float = 22.5,
                 size_z: float = 22.5,
                 workdir: Path = Path("docking_pipeline"),
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
        self.check_dependencies()

        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        if receptor.suffix == ".pdb":
            pdb2pdbqt(receptor, workdir.joinpath(receptor.stem + ".pdbqt"))
            receptor = workdir.joinpath(receptor.stem + ".pdbqt")
        if receptor.suffix != ".pdbqt":
            logging.error("receptor file must be pdb/pdbqt format")
            exit(1)
        for ligand in ligands:
            if ligand.suffix != ".sdf":
                logging.error(f"ligand file must be sdf format (file: {str(ligand)})")
                exit(1)
        self.receptor = receptor
        self.mol_group = MolGroup(ligands)
        self.build_topology()
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z

    def check_dependencies(self):
        """
        Checks if all dependencies are installed.
        """
        if not shutil.which("unidock"):
            raise ModuleNotFoundError("To run Uni-Dock, you need to install Uni-Dock")

    def _build_topology(self, id_mol_tup: Tuple[int, Chem.Mol]):
        """
        Build topology for a molecule.
        :param id_mol_tup:
        :return:
        """
        idx, mol = id_mol_tup
        topo_builder = TopologyBuilder(mol=mol)
        topo_builder.build_molecular_graph()
        fragment_info, torsion_info, atom_info = topo_builder.get_sdf_torsion_tree_info()
        return idx, fragment_info, torsion_info, atom_info

    @time_logger
    def build_topology(self):
        id_mol_tup_list = [(idx, mol.get_first_mol()) for idx, mol in enumerate(self.mol_group)]
        with Pool(os.cpu_count()) as pool:
            for idx, frag_info, torsion_info, atom_info in pool \
                    .imap_unordered(self._build_topology, id_mol_tup_list):
                self.mol_group.update_property_by_idx(idx, "fragInfo", frag_info)
                self.mol_group.update_property_by_idx(idx, "torsionInfo", torsion_info)
                self.mol_group.update_property_by_idx(idx, "atomInfo", atom_info)

    @time_logger
    def init_docking_data(self, input_dir: Union[str, os.PathLike], batch_size: int = 20):
        for sub_idx_list in self.mol_group.iter_idx_list(batch_size):
            input_list = []
            for idx in sub_idx_list:
                input_list += self.mol_group.write_sdf_by_idx(idx, save_dir=input_dir, seperate_conf=True)
            yield input_list, input_dir

    @time_logger
    def postprocessing(self, ligand_scores_list: zip,
                       topn_conf: int = 10,
                       score_name: str = "score"):
        mol_score_dict = dict()
        for ligand, scores in ligand_scores_list:
            fprefix = ligand.stem.rpartition("_out")[0].split("_CONF")[0]
            result_mols = [mol for mol in Chem.SDMolSupplier(str(ligand), removeHs=False)]
            mol_score_dict[fprefix] = mol_score_dict.get(fprefix, []) + [(mol, s) for mol, s in zip(
                result_mols, scores)]
        for fprefix in mol_score_dict:
            mol_score_list = mol_score_dict[fprefix]
            mol_score_list.sort(key=lambda x: x[1], reverse=False)
            logging.debug(fprefix)
            logging.debug(mol_score_list)
            self.mol_group.update_mol_confs_by_file_prefix(fprefix,
                                                           [copy.copy(mol) for mol, _ in
                                                            mol_score_list[:topn_conf]])
            self.mol_group.update_property_by_file_prefix(fprefix, score_name,
                                                          [score for _, score in
                                                           mol_score_list[:topn_conf]])

    @time_logger
    def run_unidock(self,
                    scoring_function: str = "vina",
                    exhaustiveness: int = 256,
                    max_step: int = 10,
                    num_modes: int = 3,
                    refine_step: int = 3,
                    topn: int = 10,
                    batch_size: int = 20,
                    local_only: bool = False,
                    score_name: str = "docking_score"
                    ):
        input_dir = make_tmp_dir(f"{self.workdir}/docking_inputs")
        output_dir = make_tmp_dir(f"{self.workdir}/docking_results")
        for ligand_list, input_dir in self.init_docking_data(
                input_dir=input_dir,
                batch_size=batch_size
        ):
            # Run docking
            ligands, scores_list = run_unidock(
                receptor=self.receptor, ligands=ligand_list, output_dir=output_dir,
                center_x=self.center_x, center_y=self.center_y, center_z=self.center_z,
                size_x=self.size_x, size_y=self.size_y, size_z=self.size_z,
                scoring=scoring_function, num_modes=num_modes,
                exhaustiveness=exhaustiveness, max_step=max_step,
                refine_step=refine_step, local_only=local_only,
            )
            # Ranking
            self.postprocessing(zip(ligands, scores_list), topn, score_name)

        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)

    @time_logger
    def save_results(self, save_dir: Union[str, os.PathLike] = ""):
        if not save_dir:
            save_dir = f"{self.workdir}/results_dir"
        save_dir = make_tmp_dir(str(save_dir), False, False)
        res_list = self.mol_group.write_sdf(save_dir=save_dir, seperate_conf=False, conf_prefix="_unidock", conf_props=["docking_score"])
        return res_list


def main(args: dict):
    workdir = Path(args["workdir"]).resolve()
    savedir = Path(args["savedir"]).resolve()

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

    if len(ligands) == 0:
        logging.error("No ligands found.")
        exit(1)
    logging.info(f"[UniDock Pipeline] {len(ligands)} ligands found.")

    logging.info("[UniDock Pipeline] Start")
    start_time = time.time()
    runner = UniDock(
        receptor=Path(args["receptor"]).resolve(),
        ligands=ligands,
        center_x=float(args["center_x"]),
        center_y=float(args["center_y"]),
        center_z=float(args["center_z"]),
        size_x=float(args["size_x"]),
        size_y=float(args["size_y"]),
        size_z=float(args["size_z"]),
        workdir=workdir
    )
    logging.info("[UniDock Pipeline] Start docking")
    runner.run_unidock(
        scoring_function=str(args["scoring_function"]),
        exhaustiveness=int(args["exhaustiveness"]),
        max_step=int(args["max_step"]),
        num_modes=int(args["num_modes"]),
        refine_step=int(args["refine_step"]),
        topn=int(args["topn"]),
        batch_size=int(args["batch_size"]),
    )
    runner.save_results(save_dir=savedir)
    end_time = time.time()
    logging.info(f"UniDock Pipeline finished ({end_time - start_time:.2f} s)")
    shutil.rmtree(workdir, ignore_errors=True)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniDock")

    parser.add_argument("-r", "--receptor", type=str, required=True,
                        help="Receptor file in pdbqt format.")
    parser.add_argument("-l", "--ligands", type=lambda s: s.split(','), default=None,
                        help="Ligand file in sdf format. Specify multiple files separated by commas.")
    parser.add_argument("-i", "--ligand_index", type=str, default="",
                        help="A text file containing the path of ligand files in sdf format.")

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

    parser.add_argument("-wd", "--workdir", type=str, default=f"unidock_pipeline_{randstr(5)}",
                        help="Working directory. Default: 'MultiConfDock'.")
    parser.add_argument("-sd", "--savedir", type=str, default="unidock_results",
                        help="Save directory. Default: 'MultiConfDock-Result'.")
    parser.add_argument("-bs", "--batch_size", type=int, default=20,
                        help="Batch size for docking. Default: 20.")

    parser.add_argument("-sf", "--scoring_function",
                        type=str, default="vina",
                        help="Scoring function used in rigid docking. Default: 'vina'.")
    parser.add_argument("-ex", "--exhaustiveness",
                        type=int, default=256,
                        help="Exhaustiveness used in rigid docking. Default: 128.")
    parser.add_argument("-ms", "--max_step",
                        type=int, default=10,
                        help="Max step used in rigid docking. Default: 10.")
    parser.add_argument("-nm", "--num_modes",
                        type=int, default=3,
                        help="Number of modes used in rigid docking. Default: 3.")
    parser.add_argument("-rs", "--refine_step",
                        type=int, default=3,
                        help="Refine step used in rigid docking. Default: 3.")
    parser.add_argument("-topn", "--topn",
                        type=int, default=100,
                        help="Top N results used in rigid docking. Default: 100.")
    return parser


def main_cli():
    """
    Command line interface for UniDock.

    Input files:
    -r, --receptor: receptor file in pdbqt format
    -l, --ligands: ligand file in sdf format, separated by commas(,)
    -i, --ligand_index: a text file containing the path of ligand files in sdf format

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

    Uni-Dock arguments:
    -sf, --scoring_function: scoring function used in rigid docking (default: vina)
    -ex, --exhaustiveness: exhaustiveness used in rigid docking (default: 256)
    -ms, --max_step: max_step used in rigid docking (default: 10)
    -nm, --num_modes: num_modes used in rigid docking (default: 3)
    -rs, --refine_step: refine_step used in rigid docking (default: 3)
    -topn, --topn: topn used in rigid docking (default: 200)
    """
    parser = get_parser()
    args = parser.parse_args().__dict__
    logging.info(f"[Params] {args}")
    main(args)


if __name__ == "__main__":
    main_cli()
