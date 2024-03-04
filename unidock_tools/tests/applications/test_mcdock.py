from pathlib import Path
import os
import shutil
import glob
import json
import subprocess
import logging
import pytest


testset_dir_path = Path(__file__).parent.parent / "inputs" / "mcdock"
testset_name_list = ["8AJX"]


def get_docking_args(testset_name):
    receptor = os.path.join(testset_dir_path, testset_name, f"{testset_name}_receptor.pdb")
    ref_ligand = os.path.join(testset_dir_path, testset_name, f"{testset_name}_ligand_ori.sdf")
    calc_ligand = os.path.join(testset_dir_path, testset_name, f"{testset_name}_ligand_prep.sdf")
    with open(os.path.join(testset_dir_path, testset_name, "docking_grid.json")) as f:
        pocket = json.load(f)
    testset_info = {
        "receptor": receptor,
        "ref_ligand": ref_ligand,
        "ligand": calc_ligand,
        "pocket": pocket,
        "confgen_ref_ligand": os.path.join(testset_dir_path, testset_name, f"{testset_name}_ligand_confgen.sdf")
    }
    return testset_info


@pytest.mark.parametrize("testset_name", testset_name_list)
def test_mcdock_steps(testset_name):
    from rdkit import Chem
    from unidock_tools.application.mcdock import MultiConfDock
    import uuid
    workdir = Path(f"mcdock_testdir_{uuid.uuid4().hex[:6]}")
    testset_info = get_docking_args(testset_name)
    pocket = testset_info["pocket"]
    mcd = MultiConfDock(
        receptor=Path(testset_info["receptor"]).resolve(),
        ligands=[Path(testset_info["ligand"]).resolve()],
        center_x=float(pocket["center_x"]),
        center_y=float(pocket["center_y"]),
        center_z=float(pocket["center_z"]),
        size_x=float(pocket["size_x"]),
        size_y=float(pocket["size_y"]),
        size_z=float(pocket["size_z"]),
        workdir=workdir
    )
    mcd.generate_conformation(max_nconf=200, min_rmsd=0.3)
    confgen_result_file = glob.glob(os.path.join(workdir, "confgen_results", "*.sdf"))[0]
    mols = Chem.SDMolSupplier(confgen_result_file, removeHs=False)
    confgen_ref_ligand = testset_info["confgen_ref_ligand"]
    confgen_ref_mols = Chem.SDMolSupplier(confgen_ref_ligand, removeHs=False)
    for i in range(len(mols)):
        assert (mols[i].GetConformer(0).GetPositions() == confgen_ref_mols[i].GetConformer(0).GetPositions()).all(), \
            "confgen results error"
    
    mcd.run_unidock(
        scoring_function="vina",
        exhaustiveness=128,
        max_step=20,
        num_modes=3,
        refine_step=5,
        seed=181129,
        topn=100,
        batch_size=20,
        docking_dir_name="rigid_docking",
        props_list=["fragAllInfo", "atomInfo"],
    )
    rigid_result_files = glob.glob(os.path.join(workdir, "rigid_docking", "*.sdf"))
    for rigid_result_file in rigid_result_files:
        mols = Chem.SDMolSupplier(rigid_result_file, removeHs=False)
        assert len(mols) == 3, "rigid docking failed to generate and keep 3 pose"
        assert not mols[0].HasProp("torsionInfo"), "rigid docking should not have torsion"
        assert mols[0].GetProp("fragInfo").strip() == " ".join([str(i) for i in range(1, 1 + mols[0].GetNumAtoms())]), "rigid docking fragment should be the whole molecular"

    mcd.run_unidock(
        scoring_function="vina",
        exhaustiveness=512,
        max_step=40,
        num_modes=1,
        refine_step=5,
        seed=181129,
        topn=1,
        batch_size=20,
        local_only=True,
        docking_dir_name="local_refine_docking",
        props_list=["fragInfo", "torsionInfo", "atomInfo"],
    )
    local_refine_result_files = glob.glob(os.path.join(workdir, "local_refine_docking", "*.sdf"))
    for local_refine_result_file in local_refine_result_files:
        mols = Chem.SDMolSupplier(local_refine_result_file, removeHs=False)
        assert len(mols) == 1, "local refine docking failed to generate and keep 1 pose"
        assert mols[0].HasProp("torsionInfo"), "local refine docking should have torsion"
        assert mols[0].HasProp("fragInfo"), "local refine docking should have fragment"
    
    shutil.rmtree(workdir)


@pytest.mark.parametrize("testset_name", testset_name_list)
def test_mcdock_cmd_default(testset_name):
    from unidock_tools.modules.docking import calc_rmsd

    testset_info = get_docking_args(testset_name)
    receptor, ref_ligand, ligand, pocket = testset_info["receptor"], testset_info["ref_ligand"], testset_info["ligand"], testset_info["pocket"]
    results_dir = "mcdock_results"
    cmd = f"unidocktools mcdock -r {receptor} -l {ligand} -sd {results_dir} -g \
        -cx {pocket['center_x']} -cy {pocket['center_y']} -cz {pocket['center_z']} \
        -sx {pocket['size_x']} -sy {pocket['size_y']} -sz {pocket['size_z']}"
    logging.debug(cmd)
    resp = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    logging.info(resp.stdout)
    assert resp.returncode == 0, f"run mcdock pipeline app err:\n{resp.stderr}"

    result_ligand = os.path.join(results_dir, f"{testset_name}_ligand_prep.sdf")
    assert os.path.exists(result_ligand), f"docking result file not found"

    rmsd = calc_rmsd(ref_ligand, result_ligand)[0]
    #assert rmsd <= 8.0, f"best pose rmsd not satisfied: {rmsd}"
    shutil.rmtree(results_dir, ignore_errors=True)