from pathlib import Path
import os
import shutil
import subprocess
import pytest


@pytest.fixture
def input_ligand():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "Imatinib.sdf")


def test_ligprep_app_ligand_file(input_ligand):
    results_dir = "prepared_ligands"
    cmd = f"unidocktools ligandprep -l {input_ligand} -sd {results_dir}"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode==0, f"run ligprep app err:\n{resp.stderr}"
    shutil.rmtree(results_dir, ignore_errors=True)


def test_ligprep_app_ligand_index(input_ligand):
    index_file = "ligand_index.txt"
    with open(index_file, "w") as f:
        f.write(input_ligand)
    results_dir = "prepared_ligands"
    cmd = f"unidocktools ligandprep -i {index_file} -sd {results_dir}"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode==0, f"run ligprep app err:\n{resp.stderr}"
    Path(index_file).unlink(missing_ok=True)
    shutil.rmtree(results_dir, ignore_errors=True)