from pathlib import Path
import os
import shutil
import uuid
import pytest


@pytest.fixture
def receptor():
    return Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             "inputs", "1iep_protein.pdbqt"))


@pytest.fixture
def ligand():
    return Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             "inputs", "1iep_ligand_prep.sdf"))


@pytest.fixture
def pocket():
    return [15.11, 53.98, 17.14, 19.23, 27.64, 24.19]


def test_run_unidock_vina(receptor, ligand, pocket):
    from unidock_tools.modules.docking import run_unidock

    workdir = Path(f"./tmp+{uuid.uuid4()}")
    workdir.mkdir(parents=True, exist_ok=True)

    result_ligands, scores_list = run_unidock(
        receptor=receptor,
        ligands=[ligand],
        output_dir=workdir,
        center_x=pocket[0],
        center_y=pocket[1],
        center_z=pocket[2],
        size_x=pocket[3],
        size_y=pocket[4],
        size_z=pocket[5],
        scoring="vina",
        num_modes=10,
        energy_range=6.0,
        seed=181129,
    )

    result_ligand = result_ligands[0]
    assert os.path.exists(result_ligand)

    scores = scores_list[0]
    assert len(scores) == 10
    for score in scores:
        assert -20 <= score <= 0

    shutil.rmtree(workdir, ignore_errors=True)


def test_run_unidock_ad4(receptor, ligand, pocket):
    from unidock_tools.modules.docking import run_unidock

    workdir = Path(f"./tmp+{uuid.uuid4()}")
    workdir.mkdir(parents=True, exist_ok=True)

    result_ligands, scores_list = run_unidock(
        receptor=receptor,
        ligands=[ligand],
        output_dir=workdir,
        center_x=pocket[0],
        center_y=pocket[1],
        center_z=pocket[2],
        size_x=pocket[3],
        size_y=pocket[4],
        size_z=pocket[5],
        scoring="ad4",
        num_modes=5,
        energy_range=6.0,
        seed=181129,
    )

    result_ligand = result_ligands[0]
    assert os.path.exists(result_ligand)

    scores = scores_list[0]
    assert len(scores) == 5

    shutil.rmtree(workdir, ignore_errors=True)