from typing import List, Tuple, Iterable, Union, Optional
from pathlib import Path
import os
import time
import shutil
import argparse
import logging
import traceback
import math
from functools import partial
from multiprocess import Pool
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol


from unidock_tools.utils import time_logger, randstr, make_tmp_dir, read_ligand, sdf_writer
from unidock_tools.modules.protein_prep import receptor_preprocessor
from unidock_tools.modules.ligand_prep import TopologyBuilder
from unidock_tools.modules.docking import run_unidock
from .base import Base


DEFAULT_ARGS = {
    "receptor": None,
    "ligands": None,
    "ligand_index": None,
    "center_x": None,
    "center_y": None,
    "center_z": None,
    "size_x": 22.5,
    "size_y": 22.5,
    "size_z": 22.5,
    "workdir": "docking_workdir",
    "savedir": "docking_results",
    "batch_size": 1200,
    "scoring_function": "vina",
    "search_mode": "",
    "exhaustiveness": 128,
    "max_step": 20,
    "num_modes": 3,
    "refine_step": 3,
    "energy_range": 3.0,
    "topn": 100,
    "score_only": False,
    "local_only": False,
    "seed": 181129,
    "debug": False,
    "bias_file": None,
    "multi_bias": False,
    "multi_bias_file": None,
    "multi_bias_index": None,
}

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
                 kept_ligand_resname_list: Optional[List[str]] = None,
                 prepared_hydrogen: bool = True,
                 preserve_original_resname: bool = True,
                 covalent_residue_atom_info_list: Optional[List[Tuple[str, str]]] = None,
                 generate_ad4_grids: bool = False,
                 bias_file: Optional[Path] = None,
                 multi_bias_files: List[Path] = [],
                 workdir: Path = Path("docking_pipeline"),
                 ):
        """
        Initializes a UniDock object.

        Args:
            receptor (Path): Path to the receptor file in PDB format.
            ligands (List[Path]): List of paths to the ligand files in SDF format.
            center_x (float): X-coordinate of the center of the docking box.
            center_y (float): Y-coordinate of the center of the docking box.
            center_z (float): Z-coordinate of the center of the docking box.
            size_x (float, optional): Size of the docking box in the x-dimension. Defaults to 22.5.
            size_y (float, optional): Size of the docking box in the y-dimension. Defaults to 22.5.
            size_z (float, optional): Size of the docking box in the z-dimension. Defaults to 22.5.
            kept_ligand_resname_list (List[str], optional): List of ligand residue names to keep during receptor preprocessing. Defaults to None.
            prepared_hydrogen (bool, optional): Whether to prepare hydrogen during receptor preprocessing. Defaults to True.
            preserve_original_resname (bool, optional): Whether to preserve the original residue names during receptor preprocessing. Defaults to True.
            covalent_residue_atom_info_list (List[Tuple[str, str]], optional): Atom information for covalent residues during receptor preprocessing. Defaults to None.
            workdir (Path, optional): Path to the working directory. Defaults to Path("docking_pipeline").
        """
        self.check_dependencies()

        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        if receptor.suffix != '.pdb':
            logging.error('receptor file must be in PDB format!')
            exit(1)
        else:
            receptor_pdbqt_file_name, protein_grid_prefix = receptor_preprocessor(str(receptor),
                                                                                  kept_ligand_resname_list=kept_ligand_resname_list,
                                                                                  prepared_hydrogen=prepared_hydrogen,
                                                                                  preserve_original_resname=preserve_original_resname,
                                                                                  target_center=(center_x, center_y, center_z),
                                                                                  box_size=(size_x, size_y, size_z),
                                                                                  covalent_residue_atom_info_list=covalent_residue_atom_info_list,
                                                                                  generate_ad4_grids=generate_ad4_grids,
                                                                                  working_dir_name=str(workdir))

            self.receptor = receptor_pdbqt_file_name
            self.ad4_map_prefix = protein_grid_prefix

        self.mols = sum([read_ligand(ligand) for ligand in ligands], [])
        self.mols = [PropertyMol(mol) for mol in self.mols]
        self.bias_file = bias_file
        self.multi_bias_files_dict = {f.stem: f for f in multi_bias_files}
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

    def _prepare_topology_sdf(self, mol: Chem.Mol, 
                              savedir: Path) -> Optional[Path]:
        """
        Build topology for a molecule.
        :param id_mol_tup:
        :return:
        """
        filename = mol.GetProp("filename")
        try:
            topo_builder = TopologyBuilder(mol=mol)
            topo_builder.build_molecular_graph()
            topo_builder.write_sdf_file(savedir / f"{filename}.sdf", do_rigid_docking=False)
            return savedir / f"{filename}.sdf"
        except:
            logging.error(f"{filename} failed to build topology: {traceback.format_exc()}")
            return None

    @time_logger
    def prepare_topology_sdf(self, mol_list: List[Chem.Mol], savedir: Path) -> List[Path]:
        prepared_sdf_list = []
        with Pool(os.cpu_count()) as pool:
            prepared_sdf_list = pool.map(partial(self._prepare_topology_sdf, savedir=savedir), mol_list)
        prepared_sdf_list = [prepared_sdf for prepared_sdf in prepared_sdf_list if prepared_sdf is not None]
        return prepared_sdf_list

    @time_logger
    def init_docking_data(self, input_dir: Path, multi_bias: bool = False, batch_size: int = 20):
        real_batch_size = math.ceil(len(self.mols) / math.ceil(len(self.mols) / batch_size))
        batched_mol_list = [self.mols[i:i+real_batch_size] for i in range(0, len(self.mols), real_batch_size)]
        for one_batch_mol_list in batched_mol_list:
            input_list = self.prepare_topology_sdf(one_batch_mol_list, input_dir)
            if multi_bias:
                for input_file in input_list:
                    filename = input_file.stem
                    logging.info(filename)
                    logging.info(self.multi_bias_files_dict)
                    if filename in self.multi_bias_files_dict:
                        shutil.copyfile(self.multi_bias_files_dict[filename], os.path.join(input_dir, f"{filename}.bpf"))
                    else:
                        logging.warning(f"Cannot find bias file in multi-bias mode for {filename}")
            yield input_list

    @time_logger
    def docking(self,
                save_dir: Path,
                scoring_function: str = "vina",
                search_mode: str = "",
                exhaustiveness: int = 256,
                max_step: int = 10,
                num_modes: int = 3,
                refine_step: int = 3,
                energy_range: float = 3.0,
                seed : int = 181129,
                batch_size: int = 1200,
                score_only: bool = False,
                local_only: bool = False,
                multi_bias: bool = False,
                score_name: str = "docking_score",
                docking_dir_name : str = "docking_dir",
                topn: int = 10,
        ):
        input_dir = make_tmp_dir(f"{self.workdir}/{docking_dir_name}/docking_inputs", date=False)
        output_dir = make_tmp_dir(f"{self.workdir}/{docking_dir_name}/docking_results", date=False)
        for ligand_list in self.init_docking_data(
                input_dir=input_dir,
                batch_size=batch_size,
                multi_bias=multi_bias,
        ):
            # Run docking
            ligands, scores_list = run_unidock(
                receptor=self.receptor, ligands=ligand_list, output_dir=output_dir,
                center_x=self.center_x, center_y=self.center_y, center_z=self.center_z,
                size_x=self.size_x, size_y=self.size_y, size_z=self.size_z,
                scoring=scoring_function, ad4_map_prefix=self.ad4_map_prefix, num_modes=num_modes,
                search_mode=search_mode, exhaustiveness=exhaustiveness, max_step=max_step, 
                seed=seed, refine_step=refine_step, energy_range=energy_range, bias_file=self.bias_file,
                score_only=score_only, local_only=local_only, multi_bias=multi_bias,
            )
            self.postprocessing(ligand_scores_list=zip(ligands, scores_list), 
                                save_dir=save_dir,
                                topn_conf=topn, score_name=score_name)

    def _postprocessing(self, ligand_scores_tup: Tuple,
                       save_dir: Path,
                       topn_conf: int = 10,
                       score_name: str = "score"):
        ligand, scores = ligand_scores_tup
        result_mols = [mol for mol in Chem.SDMolSupplier(str(ligand), removeHs=False)]
        if topn_conf and topn_conf < len(result_mols):
            result_mols = result_mols[:topn_conf]
        for i, mol in enumerate(result_mols):
            mol.SetDoubleProp(score_name, scores[i])
            for prop_name in ["Uni-Dock RESULT", "filename", "fragInfo", "torsionInfo", "atomInfo"]:
                if mol.HasProp(prop_name):
                    mol.ClearProp(prop_name)
        sdf_writer(result_mols, os.path.join(save_dir, f"{Path(ligand).stem.rstrip('_out')}.sdf"))

    @time_logger
    def postprocessing(self, ligand_scores_list: Iterable,
                       save_dir: Path,
                       topn_conf: int = 10,
                       score_name: str = "score"):
        os.makedirs(save_dir, exist_ok=True)
        with Pool(os.cpu_count()) as pool:
            pool.map(partial(self._postprocessing, save_dir=save_dir, topn_conf=topn_conf, score_name=score_name), ligand_scores_list)   


def main(args: dict):
    args = {**DEFAULT_ARGS, **args}
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
            index_content = f.read()
        index_lines1 = [line.strip() for line in index_content.split("\n") if line.strip()]
        index_lines2 = [line.strip() for line in index_content.split(" ") if line.strip()]
        ligands.extend(index_lines2 if len(index_lines2) > len(index_lines1) else index_lines1)
        ligands = [Path(ligand).resolve() for ligand in ligands if Path(ligand).exists()]

    if len(ligands) == 0:
        logging.error("No ligands found.")
        exit(1)
    logging.info(f"[UniDock Pipeline] {len(ligands)} ligands found.")

    if args['scoring_function'] == 'ad4':
        generate_ad4_grids = True
    else:
        generate_ad4_grids = False

    bias_file = None
    if args.get("bias_file"):
        bias_file = Path(args["bias_file"]).resolve()
        if not bias_file.exists():
            logging.error(f"Cannot find {bias_file}")
            exit(1)

    multi_bias_file_list = []
    if args["multi_bias"]:
        if args.get("multi_bias_file"):
            for multi_bias_file in args["multi_bias_file"]:
                if not Path(multi_bias_file).exists():
                    logging.error(f"Cannot find {multi_bias_file}")
                    continue
                multi_bias_file_list.append(Path(multi_bias_file).resolve())
        elif args.get("multi_bias_index"):
            with open(args["multi_bias_index"], "r") as f:
                index_content = f.read()
            index_lines1 = [line.strip() for line in index_content.split("\n") if line.strip()]
            index_lines2 = [line.strip() for line in index_content.split(" ") if line.strip()]
            multi_bias_file_list.extend(index_lines2 if len(index_lines2) > len(index_lines1) else index_lines1)
            multi_bias_file_list = [Path(multi_bias_file).resolve() for multi_bias_file in multi_bias_file_list if Path(multi_bias_file).exists()]
        
        if len(multi_bias_file_list) != len(ligands):
            logging.error("Number of ligands and bias files should be equal in multi-bias mode.")
            exit(1)

    logging.info("[UniDock Pipeline] Start")
    start_time = time.time()
    def parse_covalent_residue_atom_info(covalent_residue_atom_info_str: str) -> List[List[Tuple[str, str, int, str]]]:
        residue_info_list = []
        residue_atoms = covalent_residue_atom_info_str.split(',')
        for residue_atom in residue_atoms:
            residue_info = residue_atom.strip().split()
            chain_id, residue_name, residue_number, atom_name = residue_info
            residue_info_list.append((chain_id, residue_name, int(residue_number), atom_name))
        return residue_info_list
    
    runner = UniDock(
        receptor=Path(args["receptor"]).resolve(),
        ligands=ligands,
        center_x=float(args["center_x"]),
        center_y=float(args["center_y"]),
        center_z=float(args["center_z"]),
        size_x=float(args["size_x"]),
        size_y=float(args["size_y"]),
        size_z=float(args["size_z"]),
        kept_ligand_resname_list=args.get("kept_ligand_resname_list"),
        prepared_hydrogen=args.get("prepared_hydrogen", True),
        preserve_original_resname=args.get("preserve_original_resname", True),
        covalent_residue_atom_info_list=parse_covalent_residue_atom_info(args.get("covalent_residue_atom_info")) if args.get("covalent_residue_atom_info") is not None else None,
        generate_ad4_grids=generate_ad4_grids,
        bias_file=bias_file,
        multi_bias_files=multi_bias_file_list,
        workdir=workdir,
    )

    logging.info("[UniDock Pipeline] Start docking")
    runner.docking(
        save_dir=savedir,
        scoring_function=str(args["scoring_function"]),
        search_mode=str(args["search_mode"]),
        exhaustiveness=int(args["exhaustiveness"]),
        max_step=int(args["max_step"]),
        num_modes=int(args["num_modes"]),
        refine_step=int(args["refine_step"]),
        energy_range=float(args["energy_range"]),
        seed=args["seed"],
        batch_size=int(args["batch_size"]),
        score_only=bool(args["score_only"]),
        local_only=bool(args["local_only"]),
        multi_bias=bool(args["multi_bias"]),
        score_name="docking_score",
        docking_dir_name="docking_dir",
        topn=int(args["topn"]),
    )
    end_time = time.time()
    logging.info(f"UniDock Pipeline finished ({end_time - start_time:.2f} s)")
    if not args["debug"]:
        shutil.rmtree(workdir, ignore_errors=True)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniDock")

    parser.add_argument("-r", "--receptor", type=str, required=True,
                        help="Receptor file in PDB format.")
    parser.add_argument("-l", "--ligands", type=lambda s: s.split(','), default=None,
                        help="Ligand file in SDF format. Specify multiple files separated by commas.")
    parser.add_argument("-i", "--ligand_index", type=str, default=None,
                        help="A text file containing the path of ligand files in sdf format.")
    parser.add_argument("-b", "--bias_file", type=str, default=None,
                        help="Bias file in bpf format. Default: None.")
    parser.add_argument("-mbf", "--multi_bias_file", type=lambda s: s.split(','), default=None,
                        help="multi Bias file in bpf format separated by commas. Number should be equal to ligands. Default: None.")
    parser.add_argument("-mbi", "--multi_bias_index", type=str, default=None,
                        help="A text file containing the path of multi bias files in bpf format. Number should be equal to ligands. Default: None.")

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

    parser.add_argument("-kr", "--kept_ligand_resname_list", type=str, nargs='+', default=None,
                        help="List of ligand residue names to keep during receptor preprocessing. Default:None")
    parser.add_argument("-ph", "--prepared_hydrogen", action="store_false",
                        help="Whether to prepare hydrogen during receptor preprocessing.")
    parser.add_argument("-pr", "--preserve_resname", action="store_false",
                        help="Whether to preserve the original residue names during receptor preprocessing.")
    parser.add_argument("-cra", "--covalent_residue_atom_info", type=str, default=None,
                        help="Atom information for covalent residues during receptor preprocessing.To use it like this: -cra 'A VAL 1 CA, A VAL 1 CB, A VAL 1 O'")

    parser.add_argument("-wd", "--workdir", type=str, default=f"unidock_pipeline_{randstr(5)}",
                        help="Working directory. Default: 'MultiConfDock'.")
    parser.add_argument("-sd", "--savedir", type=str, default="unidock_results",
                        help="Save directory. Default: 'MultiConfDock-Result'.")
    parser.add_argument("-bs", "--batch_size", type=int, default=18000,
                        help="Batch size for docking. Default: 1200.")

    parser.add_argument("-sf", "--scoring_function",
                        type=str, default="vina",
                        help="Scoring function. Default: 'vina'.")
    parser.add_argument("-sm", "--search_mode",
                        type=str, default="",
                        help="Searching mode. Default: <empty string>.")
    parser.add_argument("-ex", "--exhaustiveness",
                        type=int, default=128,
                        help="Exhaustiveness. Default: 128.")
    parser.add_argument("-ms", "--max_step",
                        type=int, default=20,
                        help="Max step. Default: 20.")
    parser.add_argument("-nm", "--num_modes",
                        type=int, default=3,
                        help="Number of modes. Default: 3.")
    parser.add_argument("-rs", "--refine_step",
                        type=int, default=3,
                        help="Refine step. Default: 3.")
    parser.add_argument("-er", "--energy_range", 
                        type=float, default=3.0,
                        help="Energy range. Default: 3.0")
    parser.add_argument("-topn", "--topn",
                        type=int, default=100,
                        help="Top N results pose to keep. Default: 100.")
    parser.add_argument("--score_only", action="store_true",
                        help="Whether to use score_only mode.")
    parser.add_argument("--local_only", action="store_true",
                        help="Whether to use local_only mode.")
    parser.add_argument("--multi_bias", action="store_true",
                        help="Whether to use multi_bias mode.")

    parser.add_argument("--seed", type=int, default=181129, 
                        help="Uni-Dock random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Whether to use debug mode (debug-level log, keep workdir)")
    return parser


def main_cli():
    """
    Command line interface for UniDock.

    Input files:
    -r, --receptor: receptor file in PDB format
    -l, --ligands: ligand file in SDF format, separated by commas(,)
    -i, --ligand_index: a text file containing the path of ligand files in sdf format

    Docking box:
    -cx, --center_x: center_x of docking box
    -cy, --center_y: center_y of docking box
    -cz, --center_z: center_z of docking box
    -sx, --size_x: size_x of docking box (default: 22.5)
    -sy, --size_y: size_y of docking box (default: 22.5)
    -sz, --size_z: size_z of docking box (default: 22.5)
    
    Receptor processor argument:
    -kr, --kept_ligand_resname_list: List of ligand residue names to keep during receptor preprocessing (Default: None)
    -ph, --prepared_hydrogen: Whether to prepare hydrogen during receptor preprocessing  (Default: False)
    -pr, --preserve_resname: Whether to preserve the original residue names during receptor preprocessing  (Default: False)
    -cra, --covalent_residue_atom_info: Atom information for covalent residues during receptor preprocessing  (Default: None). To use it like this: -cra 'A VAL 1 CA, A VAL 1 CB, A VAL 1 O'

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
