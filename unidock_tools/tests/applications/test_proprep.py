from pathlib import Path
import os
import subprocess
import pytest


@pytest.fixture
def pdb_file():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "protein.pdb")


def test_pdb2pdbqt_app(pdb_file):
    pdbqt_file = "protein.pdbqt"
    cmd = f"unidocktools proteinprep -r {pdb_file} -o {pdbqt_file}"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode==0, f"run proteinprep app err:\n{resp.stderr}"
    Path(pdbqt_file).unlink(missing_ok=True)