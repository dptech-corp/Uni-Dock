from typing import List
from pathlib import Path
from dflow.python import (
    OP, 
    OPIO, 
    Artifact,
    Parameter,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def run_unimol_pose_refine_op(input_receptor:Artifact(Path), ligands_dir:Artifact(Path), 
    center_x:Parameter(float), center_y:Parameter(float), center_z:Parameter(float), 
    size_x:Parameter(float), size_y:Parameter(float), size_z:Parameter(float)) -> {"refine_content_json": Artifact(Path)}:
    import os
    import glob
    import json
    from argparse import Namespace
    import sys

    os.environ["MKL_THREADING_LAYER"] = "GNU"

    code_path = os.environ.get("CODE_PATH", "/opt/code")
    sys.path.append(os.path.join(code_path, "interface"))
    from demo import main

    model_path = os.environ.get("MODEL_PATH", "/model/checkpoint_best.pt")

    with open("docking_grid.json", "w") as f:
        json.dump({
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z,
        }, f)
    input_ligands = glob.glob(os.path.join(ligands_dir.as_posix(), "*.sdf"))
    input_csv_content = "input_ligand,input_docking_grid,output_ligand_name\n"
    for input_ligand in input_ligands:
        ligand_name = os.path.splitext(os.path.basename(input_ligand))[0]
        input_csv_content += f"{input_ligand},docking_grid.json,{ligand_name}\n"
    with open("input_batch.csv", "w") as f:
        f.write(input_csv_content)

    results_dir = "refined_ligands"
    args = Namespace(
        mode="batch_one2many", 
        batch_size=4, 
        input_protein=input_receptor.as_posix(),
        input_batch_file="input_batch.csv",
        output_ligand_dir=results_dir,
        model_dir=model_path,
        nthreads=8
    )
    main(args)

    result_content_list = []
    result_ligands = glob.glob(results_dir + "/*.sdf")
    for result_ligand in result_ligands:
        with open(result_ligand, "r") as f:
            content = f.read()
        result_content_list.append({"name": os.path.basename(result_ligand), "content": content})

    result_file_path = "./result.json"
    with open(result_file_path, "w") as f:
        json.dump(result_content_list, f)

    return OPIO({"refine_content_json": Path(result_file_path)})