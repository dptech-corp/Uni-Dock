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
def get_pbgbsa_inputs_from_docking_op(results_json_list: Artifact(List[Path])) -> {"ligands_dir": Artifact(Path)}:
    import json

    ligands_dir = Path("./pbgbsa_ligands_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)
    for results_json_path in results_json_list:
        with open(results_json_path.as_posix(), "r") as f:
            results_map = json.load(f)
        for basename, result in results_map.items():
            content = result["content"]
            with open(ligands_dir.joinpath(basename), "w") as f:
                f.write(content)
    return OPIO({"ligands_dir": ligands_dir})

@OP.function
def run_pbgbsa_op(receptor_file:Artifact(Path), ligands_dir:Artifact(Path),
                  pbgbsa_params:Parameter(dict)) -> {"results_dir": Artifact(Path)}:
    import os
    import glob
    import json
    import subprocess

    results_dir = Path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    param_file = results_dir.joinpath("pbgbsa_params.json")
    with open(param_file.as_posix(), "w") as f:
        json.dump(pbgbsa_params, f)
    cmd = f"unigbsa-pipeline -i {receptor_file.as_posix()} -d {ligands_dir} \
        -c {param_file.name} -nt {os.cpu_count()-1}"
    print(cmd)
    res = subprocess.run(cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", cwd=results_dir)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return OPIO({"results_dir": results_dir})