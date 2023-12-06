import subprocess, logging, os, math, shutil
from pathlib import Path
from typing import List
import logging
import shutil
from .utils import makedirs, time_logger


@time_logger
def unidock_runner(
    receptor:Path,
    ligands:List[Path],
    center_x:float,
    center_y:float,
    center_z:float,
    size_x:float=22.5,
    size_y:float=22.5,
    size_z:float=22.5,
    scoring_function:str='vina',
    output_dir:Path=Path('output'),
    exhaustiveness:int=256,
    maxstep:int=10,
    num_modes:int=10,
    refine_step:int=5,
    local_only:bool=False,
) -> (List[Path], List[List[float]]):
    if scoring_function.lower() == "ad4":
        map_prefix = gen_ad4_map(
            receptor, ligand, 
            center_x, center_y, center_z, 
            size_x, size_y, size_z,
        )
        cmd = ["unidock_ad4", "--maps", str(map_prefix)]
    else:
        cmd = ['unidock', '--receptor', str(receptor)]
        
    # datadir = makedirs("input")
    
    # for lig in ligands: shutil.move(lig, f"{datadir}/{lig.name}")
    # ligands = [Path(f"{datadir}/{lig.name}").resolve() for lig in ligands]
    
    with open("ligand_index.txt", "w") as f:
        f.write("\n".join([str(lig.resolve()) for lig in ligands]))
        
    cmd += [
        '--ligand_index', str(Path("ligand_index.txt").resolve()),
        '--center_x', str(center_x),
        '--center_y', str(center_y),
        '--center_z', str(center_z),
        '--size_x', str(size_x),
        '--size_y', str(size_y),
        '--size_z', str(size_z),
        '--scoring', scoring_function,
        '--dir', str(output_dir),
        '--exhaustiveness', str(exhaustiveness),
        '--max_step', str(maxstep),
        '--num_modes', str(num_modes),
        "--verbosity", "2",
        "--refine_step", str(refine_step),
        "--keep_nonpolar_H",
    ]

    if local_only: cmd.append("--local_only")
    
    logging.info("[Uni-Dock] %s"%(" ".join(cmd)))
    status = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # shutil.rmtree(datadir)
    
    if status.returncode != 0:
        return [], []
    
    docked_ligands = [Path(f"{output_dir}/{lig.stem}_out.sdf") for lig in ligands]
    scores = []
    for dlig in docked_ligands:
        with open(dlig, "r") as f: lines = f.readlines()
        _scores = []
        for idx, line in enumerate(lines):
            if line.startswith("> <Uni-Dock RESULT>"):
                _scores.append(float(lines[idx+1].partition(
                    "LOWER_BOUND=")[0][len("ENERGY="):]))
        scores.append(_scores)
        
    return docked_ligands, scores


def generate_ad4_map(
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
    Generates an AD4 map file for AutoDock4, given a receptor and a list of ligands.
    
    Args:
        receptor (Path): The receptor file.
        ligands (List[Path]): A list of ligand files.
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
    map_dir = makedirs(f"{prefix}-map")
    shutil.copyfile(receptor, f"{map_dir}/{receptor.name}")
    receptor = Path(f"{map_dir}/{receptor.name}")
    atom_types = set()

    # Extract atom types from ligands
    for ligand in ligands:
        with open(ligand, "r") as file:
            content = file.readlines()
        for line in content:
            if line.startswith("ATOM"):
                atom_types.add(line[77:].strip())
    atom_types = list(atom_types)

    logging.info(f"atom_types: {atom_types}")

    # Prepare gpf4 file
    npts = [math.ceil(size / spacing) for size in [size_x, size_y, size_z]]
    data_path = "/opt/data/unidock/AD4.1_bound.dat"
    mgltools_python_path = shutil.which("pythonsh")

    if not mgltools_python_path:
        raise KeyError("No mgltools env")

    mgltools_python_path = str(mgltools_python_path)
    prepare_gpf4_script_path = os.path.join(
        os.path.dirname(os.path.dirname(mgltools_python_path)),
        "MGLToolsPckgs",
        "AutoDockTools",
        "Utilities24",
        "prepare_gpf4.py",
    )

    cmd = (
        f'{mgltools_python_path} {prepare_gpf4_script_path} -r {receptor.name} '
        f'-p gridcenter="{center_x},{center_y},{center_z}" '
        f'-p npts="{npts[0]},{npts[1]},{npts[2]}" '
        f'-p spacing={spacing} -p ligand_types="{",".join(atom_types)}" '
        f'-o {prefix}.gpf && '
        f'sed -i "1i parameter_file {data_path}" {prefix}.gpf && '
        f'autogrid4 -p {prefix}.gpf -l {prefix}.glg'
    )

    logging.info(cmd)
    response = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        cwd=map_dir,
    )
    logging.info(response.stdout)

    if response.stderr:
        logging.info(response.stderr)

    return f"{map_dir}/{prefix}"
