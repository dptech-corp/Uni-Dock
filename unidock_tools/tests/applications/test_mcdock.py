from pathlib import Path
import os
import shutil
import json
import subprocess
import pytest


testset_dir_path = Path(__file__).parent.parent / "inputs" / "mcdock"
testset_name_list = ["8AJX"]


def get_docking_args(testset_name):
    receptor = os.path.join(testset_dir_path, testset_name, f"{testset_name}_receptor.pdb")
    ref_ligand = os.path.join(testset_dir_path, testset_name, f"{testset_name}_ligand_ori.sdf")
    calc_ligand = os.path.join(testset_dir_path, testset_name, f"{testset_name}_ligand_prep.sdf")
    with open(os.path.join(testset_dir_path, testset_name, "docking_grid.json")) as f:
        pocket = json.load(f)
    return receptor, ref_ligand, calc_ligand, pocket


@pytest.mark.parametrize("testset_name", testset_name_list)
def test_mcdock_cmd_default(testset_name):
    from unidock_tools.modules.docking import calc_rmsd

    receptor, ref_ligand, ligand, pocket = get_docking_args(testset_name)
    print(receptor)
    print(ref_ligand)
    print(ligand)
    print(pocket)
    results_dir = "mcdock_results"
    cmd = f"unidocktools mcdock -r {receptor} -l {ligand} -sd {results_dir} -g \
        -cx {pocket['center_x']} -cy {pocket['center_y']} -cz {pocket['center_z']} \
        -sx {pocket['size_x']} -sy {pocket['size_y']} -sz {pocket['size_z']}"
    print(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    print(resp.stdout)
    assert resp.returncode == 0, f"run mcdock pipeline app err:\n{resp.stderr}"

    result_ligand = os.path.join(results_dir, f"{testset_name}_ligand_prep.sdf")
    assert os.path.exists(result_ligand), f"docking result file not found"

    rmsd = calc_rmsd(ref_ligand, result_ligand)[0]
    assert rmsd <= 4.0, f"rmsd not satisfied: {rmsd}"
    shutil.rmtree(results_dir, ignore_errors=True)