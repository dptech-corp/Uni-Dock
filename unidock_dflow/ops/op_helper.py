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
def collect_parallel_docking_results_op(
        result_content_json_list:Artifact(List[Path]), score_table_list:Artifact(List[Path]),
) -> {"result_json_list": Artifact(List[Path]), "total_score_table":Artifact(Path)}:
    import os
    import json
    import pandas as pd

    result_json_list = []
    total_score_table = pd.DataFrame(columns=["basename", "score", "conf_id"])
    for i in range(len(score_table_list)):
        result_json_path = f"result_{i}.json"
        result_map = dict()
        score_table_path = score_table_list[i]
        result_content_json = result_content_json_list[i]
        score_table = pd.read_csv(score_table_path, index_col=None)
        total_score_table = pd.concat([total_score_table, score_table], axis=0, ignore_index=True)
        with open(result_content_json, "r") as f:
            result_content_map = json.load(f)
        for _, row in score_table.iterrows():
            basename = row["basename"]
            score = float(row["score"])
            conf_id = int(row["conf_id"])
            content = result_content_map[basename][conf_id-1]
            name, fmt = os.path.splitext(basename)
            result_map[f"{name}_{conf_id}{fmt}"] = {"score": score, "content": content}
        with open(result_json_path, "w") as f:
            json.dump(result_map, f)
        result_json_list.append(Path(result_json_path))
    total_score_table = total_score_table.sort_values(by="score")
    total_score_table.to_csv("score_table.csv", index=False)
    
    return OPIO({"result_json_list": result_json_list, "total_score_table": Path("score_table.csv")})


@OP.function
def pick_refine_ligands_op(
        results_json_list:Artifact(List[Path]), results_table:Artifact(Path),
        refine_topnum:Parameter(int), ligand_num_per_batch:Parameter(int),
) -> {"ligands_dir_list": Artifact(List[Path], archive=None)}:
    import os
    import glob
    import json
    import pandas as pd

    sub_dir_ind = 1
    curr_dir_ligand_num = 0
    ligands_dir = Path("./ligands_dir")
    curr_sub_dir = ligands_dir.joinpath(f"subdir_{sub_dir_ind}")
    curr_sub_dir.mkdir(parents=True, exist_ok=True)

    top_lignames = []
    score_df = pd.read_csv(results_table, index_col=None)
    top_df = score_df.loc[score_df["conf_id"]==1].iloc[:refine_topnum]
    for _, row in top_df.iterrows():
        basename = row["basename"]
        top_lignames.append(basename)
    
    for result_content_json in results_json_list:
        with open(result_content_json, "r") as f:
            result_content_map = json.load(f)
        for ligbasename in top_lignames:
            ligname, fmt = os.path.splitext(ligbasename)
            if result_content_map.get(f"{ligname}_1{fmt}"):
                content = result_content_map[f"{ligname}_1{fmt}"]["content"]
                with open(curr_sub_dir.joinpath(f"{ligname[:-4]}.sdf"), "w") as f:
                    f.write(content)
                curr_dir_ligand_num += 1
                if curr_dir_ligand_num > ligand_num_per_batch:
                    sub_dir_ind += 1
                    curr_dir_ligand_num = 0
                    curr_sub_dir = ligands_dir.joinpath(f"subdir_{sub_dir_ind}")
                    curr_sub_dir.mkdir(parents=True, exist_ok=True)
                    curr_dir_ligand_num = 0
    
    return OPIO({"ligands_dir_list": [Path(f) for f in glob.glob(os.path.join(ligands_dir, "*"))]})


@OP.function
def update_results_op(
        old_results_json_list:Artifact(List[Path]), new_results_json_list:Artifact(List[Path], optional=True),
        old_score_table:Artifact(Path), new_score_table:Artifact(Path, optional=True),
) -> {"results_json_list": Artifact(List[Path], global_name="results_json_list"), 
      "score_table":Artifact(Path, global_name="score_table")}:
    import json
    import glob
    import uuid
    import pandas as pd

    if not new_results_json_list or not new_score_table:
        return OPIO({"results_json_list": old_results_json_list, "score_table": old_score_table})

    res_dir = Path(f"tmp_{uuid.uuid4().hex}")
    res_dir.mkdir(parents=True, exist_ok=True)

    res_list = []
    for i, old_result_json_path in enumerate(old_results_json_list):
        with open(old_result_json_path, "r") as f:
            old_map = json.load(f)
        for new_result_json_path in new_results_json_list:
            with open(new_result_json_path, "r") as f:
                new_map = json.load(f)
            for key, value in new_map.items():
                if key in old_map:
                    old_map[key] = value
        res_path = res_dir.joinpath(f"result_{i}.json")
        with open(res_path.as_posix(), "w") as f:
            json.dump(old_map, f)
        res_list.append(res_path)
    
    old_df = pd.read_csv(old_score_table, index_col=None)
    new_df = pd.read_csv(new_score_table, index_col=None)
    for _, row in new_df.iterrows():
        basename = row["basename"]
        conf_id = int(row["conf_id"])
        score = float(row["score"])
        old_df.loc[(old_df["basename"]==basename) & (old_df["conf_id"]==conf_id), "score"] = score
    
    res_df_path = Path("final_score_table.csv")
    old_df = old_df.sort_values(by="score")
    old_df.to_csv(res_df_path.as_posix(), index=False)

    return OPIO({"results_json_list": res_list, "score_table": res_df_path})