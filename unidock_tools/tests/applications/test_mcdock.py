from pathlib import Path
import os
import shutil
import subprocess
import pytest


@pytest.fixture
def receptor():
    return Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "1bcu_protein.pdb"))


@pytest.fixture
def ligand():
    return Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "1bcu_ligand.sdf"))


@pytest.fixture
def pocket():
    return [5.0, 15.0, 50.0, 15, 15, 15]


def read_scores(sdf_file, score_name):
    score_list = []
    with open(sdf_file, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if line.startswith(f">  <{score_name}>"):
                score = float(lines[idx + 1].strip())
                score_list.append(score)
    return score_list


def test_mcdock_default(receptor, ligand, pocket):
    results_dir = "mcdock_results"
    cmd = f"unidocktools mcdock -r {receptor} -l {ligand} -sd {results_dir} \
        -cx {pocket[0]} -cy {pocket[1]} -cz {pocket[2]} -sx {pocket[3]} -sy {pocket[4]} -sz {pocket[5]} \
        -g"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode == 0, f"run mcdock pipeline app err:\n{resp.stderr}"

    result_file = os.path.join(results_dir, "1bcu_ligand.sdf")
    assert os.path.exists(result_file), f"docking result file not found"

    score_list = read_scores(result_file, "docking_score")
    score = score_list[0]
    assert -20 <= score <= 0, f"docking score not in range: {score}"
    shutil.rmtree(results_dir, ignore_errors=True)