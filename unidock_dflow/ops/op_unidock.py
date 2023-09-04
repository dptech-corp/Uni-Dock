from pathlib import Path
from dflow import (
    Step,
    Steps,
    InputArtifact,
    OutputArtifact,
    Executor,
)
from dflow.python import (
    OP, 
    OPIO, 
    Artifact,
    Parameter,
    PythonOPTemplate,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def gen_ad4_map_op(receptor:Artifact(Path), ligand_content_json:Artifact(Path), 
    docking_params:Parameter(dict)) -> {"map_dir":Artifact(Path)}:
    import os
    import math
    import shutil
    import json
    import subprocess

    center_x, center_y, center_z = docking_params["center_x"], docking_params["center_y"], docking_params["center_z"]
    size_x, size_y, size_z = docking_params["size_x"], docking_params["size_y"], docking_params["size_z"]

    map_dir = "mapdir"
    os.makedirs(map_dir, exist_ok=True)
    spacing = 0.375

    protein_name = os.path.splitext(os.path.basename(receptor))[0]
    shutil.copyfile(receptor, os.path.join(map_dir, os.path.basename(receptor)))
    
    atom_types = set()
    with open(ligand_content_json.as_posix(), "r") as f:
        ligand_content_list = json.load(f)
    for ligand_content_item in ligand_content_list:
        content = ligand_content_item["content"]
        tag = False
        for line in content.split("\n"):
            if line.strip():
                if line.startswith(">  <atomInfo>"):
                    tag = True
                elif tag and (line.startswith(">  <") or line.startswith("$$$$")):
                    tag = False
                elif tag:
                    atom_types.add(line[13:].strip())

    atom_types = list(atom_types)
    npts = [math.ceil(s / spacing) for s in [size_x, size_y, size_z]]

    data_path = "/opt/data/unidock/AD4.1_bound.dat"
    mgltools_python_path = shutil.which("pythonsh")
    if not mgltools_python_path:
        raise KeyError("No mgltools env")
    mgltools_python_path = str(mgltools_python_path)
    prepare_gpf4_script_path = os.path.join(os.path.dirname(os.path.dirname(mgltools_python_path)), 
        "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_gpf4.py")
    cmd = f'{mgltools_python_path} {prepare_gpf4_script_path} -r {os.path.basename(receptor)} \
        -p gridcenter="{center_x},{center_y},{center_z}" -p npts="{npts[0]},{npts[1]},{npts[2]}" \
        -p spacing={spacing} -p ligand_types="{",".join(atom_types)}" -o {protein_name}.gpf && \
        sed -i "1i parameter_file {data_path}" {protein_name}.gpf && \
        autogrid4 -p {protein_name}.gpf -l {protein_name}.glg'

    print(cmd)
    resp = subprocess.run(cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            encoding="utf-8", cwd=map_dir)
    print(resp.stdout)
    if resp.stderr:
        print(resp.stderr)
    return {"map_dir": Path(map_dir)}


@OP.function
def run_unidock_op(receptor:Artifact(Path), ligand_content_json:Artifact(Path), 
        docking_params:Parameter(dict), bias_content_json:Artifact(Path, optional=True), 
        batch_size:Parameter(int, default=1200)
) -> {"result_json": Artifact(Path)}:
    import os
    import shutil
    import json
    import glob
    import subprocess

    print(os.getcwd())

    scoring_func = docking_params["scoring"]
    receptor_path_str = receptor.as_posix()
    if scoring_func == "ad4":
        mapdir = "mapdir"
        map_name = os.path.basename(os.path.splitext(glob.glob(os.path.join(receptor_path_str, "*.glg"))[0])[0])
        shutil.copytree(receptor, mapdir)
        receptor_path_str = os.path.join(mapdir, map_name)

    output_dir = Path("./tmp/results_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    ligands_dir = Path("./tmp/inputs_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)

    ligand_files = []
    with open(ligand_content_json.as_posix(), "r") as f:
        ligand_content_list = json.load(f)
    for ligand_content_item in ligand_content_list:
        name = ligand_content_item["name"]
        content = ligand_content_item["content"]
        with open(ligands_dir / name, "w") as f:
            f.write(content)
        ligand_files.append(ligands_dir / name)

    if bias_content_json:
        if "multi_bias" in docking_params:
            with open(bias_content_json.as_posix(), "r") as f:
                bias_content_list = json.load(f)
            for bias_content_item in bias_content_list:
                name = bias_content_item["name"]
                content = bias_content_item["content"]
                with open(ligands_dir / f"{name}.bpf", "w") as f:
                    f.write(content)

    real_batch_size = int(len(ligand_files) // (len(ligand_files) // batch_size + 1) + 1)
    batch_ligand_files = [ligand_files[i:i+real_batch_size] for i in range(0, len(ligand_files), real_batch_size)]
    for batch_ind, sub_ligand_files in enumerate(batch_ligand_files):
        docking_file_list = output_dir.joinpath(f"file_list_{batch_ind}")
        with open(docking_file_list, "w") as f:
            for ligand_file in sub_ligand_files:
                f.write(ligand_file.as_posix() + "\n")
        if scoring_func == "ad4":
            cmd = "unidock --maps {} --ligand_index {} --dir {}".format(receptor_path_str, 
                        docking_file_list.as_posix(), output_dir.as_posix())
        else:
            cmd = "unidock --receptor {} --ligand_index {} --dir {}".format(receptor_path_str, 
                        docking_file_list.as_posix(), output_dir.as_posix())

        for k, v in docking_params.items():
            if isinstance(v, bool) and v:
                cmd += f' --{k}'
            else:
                cmd += f' --{k} {v}'

        cmd += " --verbosity 2"
        cmd += " --refine_step 3"
        cmd += " --keep_nonpolar_H"
        print(cmd)

        res = subprocess.run(args=cmd, shell=True, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf-8")
        print(res.stdout)
        if res.stderr:
            print(res.stderr)

    result_content_list = []
    for result_ligand in glob.glob(output_dir.as_posix() + "/*.sdf"):
        with open(result_ligand, "r") as f:
            content = f.read()
        conf_id = 1
        curr_conf_content = ""
        for line in content.split("\n"):
            curr_conf_content += line + "\n"
            if line.startswith("$$$$"):
                name, fmt = os.path.splitext(os.path.basename(result_ligand))
                result_content_list.append({"name": f"{name}_{conf_id}{fmt}", "content": curr_conf_content})
                conf_id += 1
                curr_conf_content = ""

    result_json_path = "result.json"
    with open(result_json_path, "w") as f:
        json.dump(result_content_list, f)

    return OPIO({"result_json": Path(result_json_path)})


@OP.function
def run_unidock_score_only_op(receptor:Artifact(Path), ligand_content_json:Artifact(Path), 
        docking_params:Parameter(dict)) -> {"score_table":Artifact(Path)}:
    import os
    import shutil
    import json
    import glob
    import subprocess
    import pandas as pd

    def read_score_from_sdf_content(content:str) -> float:
        score = None
        content_split = content.split("\n")
        score_line_ind = None
        for i, line in enumerate(content_split):
            if line.startswith("> <Uni-Dock RESULT>"):
                score_line_ind = i + 1
                break
        if score_line_ind:
            score_line = content_split[score_line_ind]
            score = float(score_line.partition("LOWER_BOUND=")[0][len("ENERGY="):])
        return score

    df = pd.DataFrame(columns=["basename", "score"])

    with open(ligand_content_json.as_posix(), "r") as f:
        ligand_content_list = json.load(f)

    if not docking_params.get("multi_bias") and not docking_params.get("bias"):
        for ligand_content_item in ligand_content_list:
            name = ligand_content_item["name"]
            content = ligand_content_item["content"]
            score = read_score_from_sdf_content(content)
            df.loc[df.shape[0]] = [name, score]
        df = df.sort_values(by=["score"])
        df.to_csv("score_table.csv", index=False)
        return OPIO({"score_table": Path("score_table.csv")})

    ligands_dir = Path("./tmp/inputs_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)
    ligand_files = []
    for ligand_content_item in ligand_content_list:
        name = ligand_content_item["name"]
        content = ligand_content_item["content"]
        with open(ligands_dir / name, "w") as f:
            f.write(content)
        ligand_files.append(ligands_dir / name)

    scoring_func = docking_params["scoring"]
    center_x, center_y, center_z = docking_params["center_x"], docking_params["center_y"], docking_params["center_z"]
    size_x, size_y, size_z = docking_params["size_x"], docking_params["size_y"], docking_params["size_z"]

    receptor_path_str = receptor.as_posix()

    docking_file_list = "docking_file_list"
    with open(docking_file_list, "w") as f:
        for ligand_file in ligand_files:
            f.write(ligand_file.as_posix() + "\n")

    cmd = f"unidock --scoring {scoring_func} --exhaustiveness 1 --score_only \
        --center_x {center_x} --center_y {center_y} --center_z {center_z} \
        --size_x {size_x} --size_y {size_y} --size_z {size_z} \
        --ligand_index {docking_file_list} --dir ."

    if scoring_func == "ad4":
        mapdir = "mapdir"
        map_name = os.path.basename(os.path.splitext(glob.glob(os.path.join(receptor_path_str, "*.glg"))[0])[0])
        shutil.copytree(receptor, mapdir)
        receptor_path_str = os.path.join(mapdir, map_name)
        cmd += f" --maps {receptor_path_str}"
    else:
        cmd += f" --receptor {receptor_path_str}"

    print(cmd)
    res = subprocess.run(args=cmd, shell=True, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    with open("scores.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("REMARK"):
                line_list = line.strip("\n").split(" ")
                basename = line_list[1]
                score = line_list[2]
                df.loc[df.shape[0]] = [basename, score]

    df = df.sort_values(by=["score"])
    df.to_csv("score_table.csv", index=False)

    return OPIO({"score_table": Path("score_table.csv")})