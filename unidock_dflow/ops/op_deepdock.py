from typing import List
from pathlib import Path
from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign,
    Artifact,
    Parameter,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def run_deepdock_pose_refine_op(system_dir:Artifact(Path)) -> {"refine_content_json": Artifact(Path)}:
    import os
    import glob
    import json
    import subprocess

    refine_content_list = []

    print(system_dir)
    system_dir_list = glob.glob(os.path.join(system_dir.as_posix(), "*"))
    print(system_dir_list)
    for system_dir in system_dir_list:
        cmd = f"python /opt/code/run_docking.py -d {system_dir}"
        resp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        print(resp.stdout)
        if resp.stderr:
            print(resp.stderr)
        try:
            result_ligand = glob.glob(os.path.join(system_dir, "*_opt.sdf"))[0]
            with open(result_ligand, "r") as f:
                content = f.read()
            refine_content_list.append({"name": os.path.basename(system_dir), "content": content})
        except:
            print(f"{system_dir} failed")

    result_file_path = "result.json"
    with open(result_file_path, "w") as f:
        json.dump(refine_content_list, f)
    return OPIO({"refine_content_json": Path(result_file_path)})


@OP.function
def collect_deep_dock_op(sub_result_list:Artifact(List[Path], archive=None)) -> {"result_json": Artifact(Path)}:
    import json

    refine_content_list = []
    for sub_result in sub_result_list:
        with open(sub_result, "r") as f:
            sub_refine_content_list = json.load(f)
        refine_content_list.extend(sub_refine_content_list)
    
    result_json = "result.json"
    with open(result_json, "w") as f:
        json.dump(refine_content_list, f)
    return OPIO({"result_json": Path(result_json)})