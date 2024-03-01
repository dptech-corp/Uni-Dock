from pathlib import Path
import os
import pytest


@pytest.fixture
def pdb_file():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "inputs", "protein.pdb")


def test_pdb2pdbqt(pdb_file):
    from unidock_tools.modules.protein_prep import pdb2pdbqt
    pdbqt_file = Path("protein.pdbqt")
    pdb2pdbqt(pdb_file, pdbqt_file)
    assert os.path.exists(pdbqt_file)
    pdbqt_file.unlink(missing_ok=True)