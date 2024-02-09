from typing import List, Tuple, Union, Optional
from pathlib import Path
import logging
import os
import shutil
import glob
import subprocess
import math

from unidock_tools.utils import randstr, make_tmp_dir, time_logger


class UniDockRunner:
    def __init__(self,
                 receptor: Union[str, os.PathLike],
                 ligands: List[Path],
                 center_x: float,
                 center_y: float,
                 center_z: float,
                 size_x: float = 22.5,
                 size_y: float = 22.5,
                 size_z: float = 22.5,
                 output_dir: Optional[Union[str, os.PathLike]] = None,
                 scoring: str = "vina",
                 num_modes: int = 10,
                 search_mode: str = "",
                 exhaustiveness: int = 256,
                 max_step: int = 10,
                 refine_step: int = 5,
                 score_only: bool = False,
                 local_only: bool = False
                 ):
        self.mgltools_python_path = ""
        self.prepare_gpf4_script_path = ""
        self.ad4_map_data_path = ""
        self.check_env(scoring == "ad4")

        self.workdir = make_tmp_dir("unidock")
        cmd = ["unidock"]
        if scoring.lower() == "ad4":
            map_prefix = self.gen_ad4_map(
                receptor, ligands,
                center_x, center_y, center_z,
                size_x, size_y, size_z,
            )
            cmd += ["--maps", str(map_prefix)]
        else:
            cmd += ["--receptor", str(receptor)]

        ligand_index_path = os.path.join(self.workdir, f"ligand_index_{randstr()}.txt")
        with open(ligand_index_path, "w") as f:
            f.write("\n".join([str(ligand) for ligand in ligands]))
        cmd += ["--ligand_index", ligand_index_path]

        if not output_dir:
            output_dir = os.path.join(self.workdir, "results_dir")
        cmd += ["--dir", str(output_dir)]

        if search_mode:
            cmd += ["--search_mode", search_mode]
        else:
            cmd += [
                "--exhaustiveness", str(exhaustiveness),
                "--max_step", str(max_step),
            ]

        cmd += [
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(size_x),
            "--size_y", str(size_y),
            "--size_z", str(size_z),
            "--scoring", scoring,
            "--num_modes", str(num_modes),
            "--refine_step", str(refine_step),
            "--verbosity", "2",
            "--keep_nonpolar_H",
        ]
        if score_only:
            cmd.append("--score_only")
        if local_only:
            cmd.append("--local_only")

        logging.info(f"unidock cmd: {cmd}")
        self.cmd = cmd

        self.pre_result_ligands = [Path(os.path.join(output_dir, f"{l.stem}_out.sdf")) for l in ligands]

    def check_env(self, use_ad4: bool = False):
        if not shutil.which("unidock"):
            raise ModuleNotFoundError("UniDock is not installed.")
        if use_ad4:
            mgltools_python_path = shutil.which("pythonsh")
            if not mgltools_python_path:
                raise ModuleNotFoundError("MGLTools is not installed.")
            prepare_gpf4_script_path = os.path.join(
                os.path.dirname(os.path.dirname(mgltools_python_path)),
                "MGLToolsPckgs",
                "AutoDockTools",
                "Utilities24",
                "prepare_gpf4.py",
            )
            if not os.path.exists(prepare_gpf4_script_path):
                raise ModuleNotFoundError("MGLTools is not installed.")
            if not shutil.which("autogrid4"):
                raise ModuleNotFoundError("AutoGrid4 is not installed.")
            self.mgltools_python_path = mgltools_python_path
            self.prepare_gpf4_script_path = prepare_gpf4_script_path
            self.ad4_map_data_path = str(Path(__file__).parent.parent.parent.joinpath("data/docking/AD4.1_bound.dat"))

    def gen_ad4_map(
            self,
            receptor: Path,
            ligands: List[Path],
            center_x: float,
            center_y: float,
            center_z: float,
            size_x: float = 22.5,
            size_y: float = 22.5,
            size_z: float = 22.5,
            spacing: float = 0.375,
    ) -> str:
        """
        Generates AD4 map files for AutoDock4,

        Args:
            receptor (Path): Input receptor file.
            ligands (List[Path]): Input ligand files.
            center_x (float): X-coordinate of the grid center.
            center_y (float): Y-coordinate of the grid center.
            center_z (float): Z-coordinate of the grid center.
            size_x (float): Grid size in the X-axis. Default is 22.5.
            size_y (float): Grid size in the Y-axis. Default is 22.5.
            size_z (float): Grid size in the Z-axis. Default is 22.5.
            spacing (float): Grid spacing. Default is 0.375.

        Returns:
            str: Path of the generated AD4 map file.
        """
        # Initialize variables
        prefix = receptor.stem
        map_dir = os.path.join(self.workdir, "mapdir")
        os.makedirs(map_dir, exist_ok=True)
        shutil.copyfile(receptor, os.path.join(map_dir, receptor.name))
        receptor = Path(os.path.join(map_dir, receptor.name))
        # Extract atom types from ligands
        atom_types = set()
        for ligand in ligands:
            tag = False
            with open(ligand, "r") as f:
                for line in f.readlines():
                    if line.strip():
                        if line.startswith(">  <atomInfo>"):
                            tag = True
                        elif tag and (line.startswith(">  <") or line.startswith("$$$$")):
                            tag = False
                        if tag:
                            atom_types.add(line[13:].strip())
        atom_types = list(atom_types)
        logging.info(f"atom_types: {atom_types}")

        npts = [math.ceil(size / spacing) for size in [size_x, size_y, size_z]]

        cmd = "".join([
            f"{self.mgltools_python_path} {self.prepare_gpf4_script_path} ",
            f"-r {receptor.name} ",
            f"-p gridcenter='{center_x},{center_y},{center_z}' ",
            f"-p npts='{npts[0]},{npts[1]},{npts[2]}' ",
            f"-p spacing={spacing} -p ligand_types='{','.join(atom_types)}' ",
            f"-o {prefix}.gpf && ",
            f"sed -i '1i parameter_file {self.ad4_map_data_path}' {prefix}.gpf && ",
            f"autogrid4 -p {prefix}.gpf -l {prefix}.glg"
        ])
        logging.info(cmd)
        resp = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            encoding="utf-8",
            cwd=map_dir,
        )
        logging.info(f"Gen ad4 map log: {resp.stdout}")
        if resp.returncode != 0:
            logging.error(f"Gen ad4 map err: {resp.stderr}")

        return os.path.join(map_dir, prefix)

    def run(self):
        resp = subprocess.run(
            self.cmd,
            capture_output=True,
            encoding="utf-8",
        )
        logging.info(f"Run Uni-Dock log: {resp.stdout}")
        if resp.returncode != 0:
            logging.error(f"Run Uni-Dock error: {resp.stderr}")

        result_ligands = [f for f in self.pre_result_ligands if os.path.exists(f)]
        return result_ligands

    @staticmethod
    def read_scores(ligand_file: Union[str, os.PathLike]) -> List[float]:
        score_list = []
        with open(ligand_file, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line.startswith("> <Uni-Dock RESULT>"):
                    score = float(lines[idx + 1].partition(
                        "LOWER_BOUND=")[0][len("ENERGY="):])
                    score_list.append(score)
        return score_list

    def clean_workdir(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


@time_logger
def run_unidock(
        receptor: Path,
        ligands: List[Path],
        output_dir: Path,
        center_x: float,
        center_y: float,
        center_z: float,
        size_x: float = 22.5,
        size_y: float = 22.5,
        size_z: float = 22.5,
        scoring: str = "vina",
        num_modes: int = 10,
        search_mode: str = "",
        exhaustiveness: int = 256,
        max_step: int = 10,
        refine_step: int = 5,
        score_only: bool = False,
        local_only: bool = False,
) -> Tuple[List[Path], List[List[float]]]:
    runner = UniDockRunner(
        receptor=receptor, ligands=ligands,
        center_x=center_x, center_y=center_y, center_z=center_z,
        size_x=size_x, size_y=size_y, size_z=size_z,
        output_dir=output_dir,
        scoring=scoring, num_modes=num_modes,
        search_mode=search_mode,
        exhaustiveness=exhaustiveness, max_step=max_step, refine_step=refine_step,
        score_only=score_only, local_only=local_only,
    )
    result_ligands = runner.run()
    scores_list = [UniDockRunner.read_scores(ligand) for ligand in result_ligands]
    runner.clean_workdir()

    return result_ligands, scores_list
