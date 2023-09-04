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
def run_deltavina_score_op(system_list_dir:Artifact(Path)) -> {"result_table": Artifact(Path)}:
    import os
    import glob
    import json
    import subprocess
    import pandas as pd

    cmd = f"run-dxgb --runfeatures --datadir {system_list_dir} --average --outfile score.csv --multi"
    resp = subprocess.run(cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    print(resp.stdout)
    if resp.stderr:
        print(resp.stderr)
    
    result_file = os.path.join(system_list_dir, "score.csv")
    return OPIO({"result_table": Path(result_file)})


@OP.function
def collect_deltavina_op(result_table_list:Artifact(List[Path])) -> {"result_table": Artifact(Path)}:
    import pandas as pd

    df = None
    for result_table in result_table_list:
        sub_df = pd.read_csv(result_table)
        if df is None:
            df = sub_df
        else:
            df = pd.concat([df, sub_df], axis=0, ignore_index=True)

    res_path = "result_all.csv"
    df.to_csv(res_path, index=False)
    return OPIO({"result_table": Path(res_path)})