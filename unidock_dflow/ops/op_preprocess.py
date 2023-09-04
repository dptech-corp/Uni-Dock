from typing import List
from pathlib import Path
import os
import shutil
import subprocess

from dflow.python import (
    OP, 
    OPIO, 
    Artifact,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def preprocess_receptor_op(
        input_receptor:Artifact(Path),
) -> {"output_receptor": Artifact(Path)}:

    protein_basename = os.path.basename(input_receptor.as_posix())
    output_receptor = os.path.splitext(protein_basename)[0] + '.pdbqt'

    mgltools_python_path = shutil.which("pythonsh")
    if not mgltools_python_path:
        raise KeyError("mgltools env not found, please install first")
    mgltools_python_path = str(mgltools_python_path)
    prepare_receptor_script_path = os.path.join(os.path.dirname(os.path.dirname(mgltools_python_path)), 
            "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_receptor4.py")
    cmd = f"{mgltools_python_path} {prepare_receptor_script_path} \
        -r {input_receptor.as_posix()} -o {output_receptor} -U nphs_lps_nonstdres"
    resp = subprocess.run(cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8")
    print(resp.stdout)
    if resp.stderr:
        print(resp.stderr)
    return OPIO({"output_receptor":Path(output_receptor)})


@OP.function
def preprocess_ligands_op(
        ligands_dir: Artifact(Path),
) -> {"content_json": Artifact(Path)}:
    import json
    import glob
    import uuid
    from unidock_tools.ligand_prepare.ligand_prep import LigandPrepareRunner

    input_ligands = [f for f in glob.glob(ligands_dir.as_posix() + "/*.sdf")]
    results_dir = Path(f"./tmp/{uuid.uuid4().hex}")
    results_dir.mkdir(parents=True, exist_ok=True)
    runner = LigandPrepareRunner(input_ligands, workdir=results_dir, standardize=True)
    result_ligands = runner.prepare_ligands()

    result_content_list = []
    for result_ligand in result_ligands:
        with open(result_ligand, "r") as f:
            content = f.read()
        result_content_list.append({"name": os.path.basename(result_ligand), "content": content})

    result_json_path = results_dir / "result.json"
    with open(result_json_path.as_posix(), "w") as f:
        json.dump(result_content_list, f)

    return OPIO({"content_json": result_json_path})