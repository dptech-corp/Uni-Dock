import os
import time
import shutil
import argparse
import subprocess as sp
import multiprocessing as mlp

from rdkit import Chem
from pathlib import Path
from typing import List, Dict
from copy import deepcopy as dc

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s][%(levelname)s]%(message)s',
)

from .utils import makedirs
from .utils import time_logger
from .utils import split_sdf, concat_sdf, sdf_writer
from .utils import remove_props_in_sdf, add_props_in_sdf
from .utils import unidock_runner
from .utils import confgen, topogen
from .utils import MoleculeGroup


class MultiConfDock:
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
            workdir: Path = Path("MultiConfDock"),
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
            gen_conf (bool, optional): Whether to generate conformers for the ligands. Defaults to True.
            max_nconf (int, optional): Maximum number of conformers to generate for each ligand. Defaults to 1000.
            workdir (Path, optional): Path to the working directory. Defaults to Path("MultiConfDock").
        """
        if receptor.suffix != ".pdbqt":
            logging.error("receptor file must be pdbqt format")
            exit(1)
        for lig in ligands:
            if lig.suffix != ".sdf":
                logging.error(f"ligand file must be sdf format (file: {str(lig)})")
                exit(1)
        self.receptor = receptor
        self.workdir = workdir
        # [[mol1-conf1, ..., mol1-confM], ..., [molN-conf1, ..., molN-confM]]
        self.molgroup = MoleculeGroup(ligands)
        os.makedirs(workdir, exist_ok=True)
        if gen_conf: self.generate_conformation(max_nconf, min_rmsd)
        self.build_topology()
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        
    @time_logger
    def generate_conformation(self,
        max_nconf: int = 1000,
        min_rmsd: float = 0.5,
    ):
        for idx, mollist, propdict in self.molgroup:
            mol = mollist[0]
            prefix = propdict["file_prefix"]
            mollist = confgen(mol, prefix, max_nconf, min_rmsd)
            self.molgroup.update_molecule_list(idx, mollist)
    
    @time_logger
    def build_topology(self):
        for idx, mollist, _ in self.molgroup:
            fraginfo, torsioninfo, atominfo, fraginfo_all = topogen(mollist[0])
            self.molgroup.update_property(idx, "fraginfo", fraginfo)
            self.molgroup.update_property(idx, "torsioninfo", torsioninfo)
            self.molgroup.update_property(idx, "atominfo", atominfo)
            self.molgroup.update_property(idx, "fraginfo_all", fraginfo_all)

    @time_logger
    def init_docking_data(self, prefix:str="init", prop_list:List[str]=[], batch_size:int=20):
        os.chdir(self.workdir)
        input_dir = makedirs(f"{prefix}_inputs")
        index_list = self.molgroup.get_index()
        for idx in range(0, len(index_list), batch_size):
            input_list = []
            for i in index_list[idx:idx+batch_size]:
                input_list += self.molgroup.get_sdf_by_index(i, prop_list, save_dir=input_dir, seperate_confs=True)
            logging.info(f"[{prefix}] {len(input_list)} inputs generated.")
            yield input_list, input_dir

    @time_logger
    def postprocessing(self, ligands: List[Path], scores: List[float], topn: int = 10, score_name: str = "score"):
        os.chdir(self.workdir)
        mol_score_dict = {}
        for ligand, score in zip(ligands, scores):
            fprefix = ligand.stem.replace("_out", "").split("_CONF")[0]
            mol_score_dict[fprefix] = mol_score_dict.get(fprefix, []) + [(mol, s) for mol, s in zip(
                Chem.SDMolSupplier(str(ligand), removeHs=False), score)]
        for fprefix in mol_score_dict.keys():
            mol_score_list = mol_score_dict[fprefix]
            logging.info(f"[PostProc] {fprefix} has generated {len(mol_score_list)} poses.")
            mol_score_list.sort(key=lambda x: x[1], reverse=False)
            self.molgroup.update_molecule_list_by_file_prefix(fprefix, [dc(mol) for mol, score in mol_score_list[:topn]])
            self.molgroup.update_property_by_file_prefix(fprefix, score_name, [score for mol, score in mol_score_list[:topn]])

    @time_logger
    def rigid_docking(self, 
        scoring_function: str = "vina",
        exhaustiveness: int = 256,
        maxstep: int = 10,
        num_modes: int = 3,
        refine_step: int = 3,
        topn: int = 10,
        batch_size: int = 20,
    ):
        os.chdir(self.workdir)
        for ligand_list, input_dir in self.init_docking_data(
            prefix="rigid_docking", 
            prop_list=["fraginfo_all", "atominfo"],
            batch_size=batch_size
        ):
            # Run rigid docking
            output_dir = makedirs("rigid_docking")
            ligands, scores = unidock_runner(
                self.receptor, ligand_list, 
                self.center_x, self.center_y, self.center_z, 
                self.size_x, self.size_y, self.size_z,
                scoring_function, output_dir,
                exhaustiveness, maxstep, 
                num_modes, refine_step
            )
            shutil.rmtree(input_dir)
            # Ranking
            self.postprocessing(ligands, scores, topn, "rigid_docking_score")
            shutil.rmtree(output_dir)
        
    @time_logger
    def local_refine(self,
        scoring_function: str = "vina",
        exhaustiveness: int = 32,
        maxstep: int = 40,
        num_modes: int = 3,
        refine_step: int = 3,
        topn: int = 10,
        batch_size: int = 20,
    ):
        os.chdir(self.workdir)
        for ligand_list, input_dir in self.init_docking_data(
            prefix="local_refine", 
            prop_list=["fraginfo", "atominfo", "torsioninfo"],
            batch_size=batch_size
        ):
            # Run local refine
            output_dir = makedirs(prefix="local_refine")
            ligands, scores = unidock_runner(
                self.receptor, ligand_list, 
                self.center_x, self.center_y, self.center_z, 
                self.size_x, self.size_y, self.size_z,
                scoring_function, output_dir,
                exhaustiveness, maxstep, 
                num_modes, refine_step, True
            )
            shutil.rmtree(input_dir)
            # Ranking
            self.postprocessing(ligands, scores, topn, "local_refine_score")
            shutil.rmtree(output_dir)
        
    @time_logger
    def save_result(self, savedir: Path = None):
        os.chdir(self.workdir)
        if savedir is None:
            savedir = "multi_conf_dock_result"
        savedir = makedirs(str(savedir), False, False)
        res_list = [p for p in self.molgroup.get_sdf(
            ["all"], save_dir=savedir, seperate_confs=False,
            additional_suffix="_mcdock")]
        return res_list
        

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
    -ms_rd, --maxstep_rigid_docking: maxstep used in rigid docking (default: 10)
    -nm_rd, --num_modes_rigid_docking: num_modes used in rigid docking (default: 3)
    -rs_rd, --refine_step_rigid_docking: refine_step used in rigid docking (default: 3)
    -topn_rd, --topn_rigid_docking: topn used in rigid docking (default: 200)

    Scoring function and search mode for local refine:
    -sf_lr, --scoring_function_local_refine: scoring function used in local refine (default: vina)
    -ex_lr, --exhaustiveness_local_refine: exhaustiveness used in local refine (default: 32)
    -ms_lr, --maxstep_local_refine: maxstep used in local refine (default: 40)
    -nm_lr, --num_modes_local_refine: num_modes used in local refine (default: 3)
    -rs_lr, --refine_step_local_refine: refine_step used in local refine (default: 5)
    -topn_lr, --topn_local_refine: topn used in local refine (default: 100)
    """
    parser = argparse.ArgumentParser(description="MultiConfDock")

    parser.add_argument("-r", "--receptor", type=str, required=True, 
        help="Receptor file in pdbqt format.")
    parser.add_argument("-l", "--ligands", type=lambda s: s.split(','), default=None, 
        help="Ligand file in sdf format. Specify multiple files separated by commas.")
    parser.add_argument("-i", "--ligand_index", type=str, default="",
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

    parser.add_argument("-wd", "--workdir", type=str, default="MultiConfDock",
        help="Working directory. Default: 'MultiConfDock'.")
    parser.add_argument("-sd", "--savedir", type=str, default="MultiConfDock-Result",
        help="Save directory. Default: 'MultiConfDock-Result'.")
    parser.add_argument("-bs", "--batch_size", type=int, default=20,
        help="Batch size for docking. Default: 20.")

    parser.add_argument("-sf_rd", "--scoring_function_rigid_docking", 
        type=str, default="vina",
        help="Scoring function used in rigid docking. Default: 'vina'.")
    parser.add_argument("-ex_rd", "--exhaustiveness_rigid_docking",
        type=int, default=256,
        help="Exhaustiveness used in rigid docking. Default: 128.")
    parser.add_argument("-ms_rd", "--maxstep_rigid_docking",
        type=int, default=10,
        help="Max step used in rigid docking. Default: 10.")
    parser.add_argument("-nm_rd", "--num_modes_rigid_docking",
        type=int, default=3,
        help="Number of modes used in rigid docking. Default: 3.")
    parser.add_argument("-rs_rd", "--refine_step_rigid_docking",
        type=int, default=3,
        help="Refine step used in rigid docking. Default: 3.")
    parser.add_argument("-topn_rd", "--topn_rigid_docking",
        type=int, default=100,
        help="Top N results used in rigid docking. Default: 100.")
    parser.add_argument("-ex_lr", "--exhaustiveness_local_refine",
        type=int, default=32,
        help="Exhaustiveness used in rigid docking. Default: 128.")
    parser.add_argument("-ms_lr", "--maxstep_local_refine",
        type=int, default=40,
        help="Max step used in rigid docking. Default: 10.")
    parser.add_argument("-sf_lr", "--scoring_function_local_refine",
        type=str, default="vina",
        help="Scoring function used in local refine. Default: 'vina'.")
    
    parser.add_argument("-nm_lr", "--num_modes_local_refine",
        type=int, default=1,
        help="Number of modes used in local refine. Default: 1.")
    parser.add_argument("-rs_lr", "--refine_step_local_refine",
        type=int, default=5,
        help="Refine step used in local refine. Default: 5.")
    parser.add_argument("-topn_lr", "--topn_local_refine",
        type=int, default=1,
        help="Top N results used in local refine. Default: 1.")

    args = parser.parse_args()
    logging.info(f"[Params] {args.__dict__}")
    
    workdir = Path(args.workdir).resolve()
    savedir = Path(args.savedir).resolve()
    
    ligands = []
    if args.ligands:
        for lig in args.ligands:
            if not Path(lig).exists(): 
                logging.error(f"Cannot find {lig}")
                continue
            ligands.append(Path(lig).resolve())
    if args.ligand_index:
        with open(args.ligand_index, "r") as f:
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
        receptor  = Path(args.receptor).resolve(),
        ligands   = ligands,
        center_x  = float(args.center_x),
        center_y  = float(args.center_y),
        center_z  = float(args.center_z),
        size_x    = float(args.size_x),
        size_y    = float(args.size_y),
        size_z    = float(args.size_z),
        gen_conf  = bool(args.gen_conf),
        max_nconf = int(args.max_num_confs_per_ligand),
        min_rmsd  = float(args.min_rmsd),
        workdir   = workdir
    )
    logging.info("[MultiConfDock] Start rigid docking")
    mcd.rigid_docking(
        scoring_function = str(args.scoring_function_rigid_docking),
        exhaustiveness   = int(args.exhaustiveness_rigid_docking),
        maxstep          = int(args.maxstep_rigid_docking),
        num_modes        = int(args.num_modes_rigid_docking),
        refine_step      = int(args.refine_step_rigid_docking),
        topn             = int(args.topn_rigid_docking),
        batch_size       = int(args.batch_size),
    )
    logging.info("[MultiConfDock] Start local refine")
    mcd.local_refine(
        scoring_function = str(args.scoring_function_local_refine),
        exhaustiveness   = int(args.exhaustiveness_local_refine),
        maxstep          = int(args.maxstep_local_refine),
        num_modes        = int(args.num_modes_local_refine),
        refine_step      = int(args.refine_step_local_refine),
        topn             = int(args.topn_local_refine),
        batch_size       = int(args.batch_size),
    )
    mcd.save_result(
        savedir = savedir
    )
    end_time = time.time()
    logging.info(f"[MultiConfDock] Workflow finished ({end_time - start_time:.2f} s)")
    shutil.rmtree(workdir)

    
if __name__ == "__main__":
    main_cli()