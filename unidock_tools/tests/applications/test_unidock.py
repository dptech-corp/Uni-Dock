from pathlib import Path
import os
import shutil
import glob
import json
import subprocess
import pytest


@pytest.fixture
def receptor():
    return Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", 
                             "unidock_pipeline", "1bcu", "protein.pdb"))


@pytest.fixture
def ligand():
    return Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", 
                             "unidock_pipeline", "1bcu", "ligand.sdf"))


@pytest.fixture
def pocket():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", 
                           "unidock_pipeline", "1bcu", "docking_grid.json")) as f:
        pocket = json.load(f)
    return pocket


testset_dir_path = Path(__file__).parent.parent / "inputs" / "unidock_pipeline"
testset_name_list = ["bigsdf", "1bcu", "one_ligand_failed"]


def get_docking_args(testset_name):
    receptor = os.path.join(testset_dir_path, testset_name, "protein.pdb")
    ligand = os.path.join(testset_dir_path, testset_name, "ligand.sdf")
    with open(os.path.join(testset_dir_path, testset_name, "docking_grid.json")) as f:
        pocket = json.load(f)
    testset_info = {
        "receptor": receptor,
        "ligand": ligand,
        "pocket": pocket,
    }
    return testset_info


def read_scores(sdf_file, score_name):
    score_list = []
    with open(sdf_file, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if line.startswith(f">  <{score_name}>"):
                score = float(lines[idx + 1].strip())
                score_list.append(score)
    return score_list


def test_unidock_pipeline_ligand_index(receptor, ligand, pocket):
    index_file = Path("ligand_index.txt")
    with open(index_file, "w") as f:
        f.write(str(ligand))
    results_dir = "unidock_results_input_index"
    cmd = f"unidocktools unidock_pipeline -r {receptor} -i {index_file} -sd {results_dir} \
            -cx {pocket['center_x']} -cy {pocket['center_y']} -cz {pocket['center_z']} \
            -sx {pocket['size_x']} -sy {pocket['size_y']} -sz {pocket['size_z']} \
            -sf vina -nm 1 --seed 181129"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode == 0, f"run unidock pipeline app err:\n{resp.stderr}"

    result_file = os.path.join(results_dir, Path(ligand).name)
    assert os.path.exists(result_file), f"docking result file not found"

    score_list = read_scores(result_file, "docking_score")
    score = score_list[0]
    assert -20 <= score <= 0, f"Uni-Dock score not in range: {score}"
    index_file.unlink(missing_ok=True)
    shutil.rmtree(results_dir, ignore_errors=True)


def test_unidock_pipeline_scoring_ad4(receptor, ligand, pocket):
    results_dir = "unidock_results_ad4"
    cmd = f"unidocktools unidock_pipeline -r {receptor} -l {ligand} -sd {results_dir} \
            -cx {pocket['center_x']} -cy {pocket['center_y']} -cz {pocket['center_z']} \
            -sx {pocket['size_x']} -sy {pocket['size_y']} -sz {pocket['size_z']} \
            -sf ad4 -nm 1 --seed 181129"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode == 0, f"run unidock pipeline app err:\n{resp.stderr}"

    result_file = os.path.join(results_dir, Path(ligand).name)
    assert os.path.exists(result_file), f"docking result file not found"

    score_list = read_scores(result_file, "docking_score")
    score = score_list[0]
    assert -20 <= score <= 0, f"Uni-Dock score not in range: {score}"
    shutil.rmtree(results_dir, ignore_errors=True)


def test_unidock_pipeline_multi_pose(receptor, ligand, pocket):
    results_dir = "unidock_results_multi_pose"
    cmd = f"unidocktools unidock_pipeline -r {receptor} -l {ligand} -sd {results_dir} \
            -cx {pocket['center_x']} -cy {pocket['center_y']} -cz {pocket['center_z']} \
            -sx {pocket['size_x']} -sy {pocket['size_y']} -sz {pocket['size_z']} \
            -sf vina -nm 4 --seed 181129"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode == 0, f"run unidock pipeline app err:\n{resp.stderr}"

    result_file = os.path.join(results_dir, Path(ligand).name)
    assert os.path.exists(result_file), f"docking result file not found"

    score_list = read_scores(result_file, "docking_score")
    assert len(score_list) == 4, f"docking result pose num({len(score_list)}) not match"
    for score in score_list:
        assert -20 <= score <= 0, f"Uni-Dock score not in range: {score}"
    shutil.rmtree(results_dir, ignore_errors=True)


@pytest.mark.parametrize("testset_name", testset_name_list)
def test_unidock_pipeline_default_arg(testset_name):
    testset_info = get_docking_args(testset_name)
    receptor, ligand, pocket = testset_info["receptor"], testset_info["ligand"], testset_info["pocket"]
    results_dir = f"unidock_results_{testset_name}"
    cmd = f"unidocktools unidock_pipeline -r {receptor} -l {ligand} -sd {results_dir} \
        -cx {pocket['center_x']} -cy {pocket['center_y']} -cz {pocket['center_z']} \
        -sx {pocket['size_x']} -sy {pocket['size_y']} -sz {pocket['size_z']} \
        -sf vina -nm 1 --seed 181129"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode == 0, f"run unidock pipeline app err:\n{resp.stderr}"

    result_files = glob.glob(os.path.join(results_dir, "*.sdf"))
    assert len(result_files) > 0, f"failed to run all ligands"

    for result_file in result_files:
        score_list = read_scores(result_file, "docking_score")
        score = score_list[0]
        assert score <= 0, f"Uni-Dock score is abnormal"
    shutil.rmtree(results_dir, ignore_errors=True)