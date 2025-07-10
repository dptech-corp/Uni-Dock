import os
import shutil
import subprocess as sp


GRID_SPACING = 0.375
LIGAND_ATOM_TYPES = ['A', 'Br', 'C', 'Ca', 'Cl', 'F', 'Fe', 'G', 'GA', 'H', 'HD', 'HS','I', 'J', 'Mg', 'Mn', 
                     'N', 'NA', 'NS', 'OA', 'OS', 'P', 'Q', 'S', 'SA', 'Z', 'Zn']

GPF_SCRIPTS = """outlev 2
parameter_file AD4.1_bound.dat
npts {npts}
gridfld protein.maps.fld
spacing {spacing}
receptor_types {receptor_types}
ligand_types {ligand_types}
receptor {receptor}
gridcenter {gridcenter}
smooth 0.500000
{map_lines}
elecmap protein.e.map
dsolvmap protein.d.map
dielectric -0.145600"""

AUTOGRID_PARAMETER_FILE = os.path.join(os.path.dirname(__file__), 'data', 'AD4.1_bound.dat')
AUTOGRID_BINARY = os.path.join(os.path.dirname(__file__), 'bin', 'autogrid4')


def get_protein_atom_types(pdbqt_file:str) -> list[str]:
    atom_types = []
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_types.append(line.strip().split()[-1])
    return list(set(atom_types))


def generate_ad4_grid(pdbqt_file:str, workdir:str, 
                      center:tuple[float, float, float], 
                      size:tuple[float, float, float]) -> str:
    os.makedirs(workdir, exist_ok=True)

    receptor_atom_types = get_protein_atom_types(pdbqt_file)
    npts = [int(s / GRID_SPACING) for s in size]

    parameter_scripts = GPF_SCRIPTS.format(
        npts=" ".join([str(n) for n in npts]),
        spacing=" ".join([str(GRID_SPACING) for _ in range(3)]),
        receptor_types=" ".join(receptor_atom_types),
        ligand_types=" ".join(LIGAND_ATOM_TYPES),
        receptor=os.path.basename(pdbqt_file),
        gridcenter=" ".join([str(c) for c in center]),
        map_lines="\n".join([f'map protein.{ligand_type}.map' for ligand_type in LIGAND_ATOM_TYPES]),
    )

    with open(os.path.join(workdir, 'protein.gpf'), 'w') as f:
        f.write(parameter_scripts)

    shutil.copyfile(pdbqt_file, os.path.join(workdir, os.path.basename(pdbqt_file)))

    shutil.copyfile(AUTOGRID_PARAMETER_FILE, os.path.join(workdir, 'AD4.1_bound.dat'))

    os.chmod(AUTOGRID_BINARY, 0o755)
    resp = sp.run(f'{AUTOGRID_BINARY} -p protein.gpf -l protein.glg', shell=True, 
                  capture_output=True, encoding='utf-8', cwd=workdir)
    if resp.returncode != 0:
        raise RuntimeError(f'autogrid4 failed: {resp.stdout}\n{resp.stderr}')

    return os.path.join(workdir, 'protein')