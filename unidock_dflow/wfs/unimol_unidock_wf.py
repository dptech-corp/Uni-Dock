from pathlib import Path
import os
import sys
import glob
import json
import argparse

from dflow import (
    Executor,
    Workflow, 
    Step,
    Steps,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    upload_artifact
)
from dflow.python import (
    PythonOPTemplate, 
)

sys.path.append(os.path.dirname(__file__))
from utils import (
    setup,
    get_local_executor,
    get_dispatcher_executor,
)

ops_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ops")
sys.path.append(ops_folder_path)
from op_gen_bpf import gen_multi_bpf_op
from op_preprocess import preprocess_receptor_op, preprocess_ligands_op
from op_pose_refine import run_unimol_pose_refine_op
from op_unidock import gen_ad4_map_op, run_unidock_op, run_unidock_score_only_op


def get_pose_bias_superop(docking_config:dict, image_dict:dict, local_executor:Executor, remote_executor:Executor) -> Steps:
    unimol_image = image_dict.get("unimol_image", "")
    unidock_tools_image = image_dict.get("unidock_tools_image", "")

    pose_refine_pipeline = Steps("pose-bias-superop")
    pose_refine_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "ligands_dir": InputArtifact(),
    }
    unimol_pose_refine_step = Step(
        name="unimol-pose-refine",
        artifacts={
            "input_receptor": pose_refine_pipeline.inputs.artifacts["input_receptor"],
            "ligands_dir": pose_refine_pipeline.inputs.artifacts["ligands_dir"],
        },
        parameters={
            "center_x":docking_config["center_x"], "center_y":docking_config["center_y"], "center_z":docking_config["center_z"], 
            "size_x":docking_config["size_x"], "size_y":docking_config["size_y"], "size_z":docking_config["size_z"]},
        template=PythonOPTemplate(
            run_unimol_pose_refine_op,
            image=unimol_image,
            image_pull_policy="IfNotPresent",
        ),
        executor=remote_executor
    )
    pose_refine_pipeline.add(unimol_pose_refine_step)
    refine_content_json = unimol_pose_refine_step.outputs.artifacts["refine_content_json"]

    gen_bpf_step = Step(
        name = "gen-bpf",
        artifacts={
            "refine_content_json": refine_content_json,
        },
        template=PythonOPTemplate(
            gen_multi_bpf_op,
            image=unidock_tools_image,
            image_pull_policy="IfNotPresent",
            requests={"cpu": "1", "memory": "2Gi"},
            limits={"cpu": "2", "memory": "4Gi"},
        ),
        executor=local_executor
    )
    pose_refine_pipeline.add(gen_bpf_step)

    bias_content_json = gen_bpf_step.outputs.artifacts["bias_content_json"]
    pose_refine_pipeline.outputs.artifacts["bias_content_json"] = OutputArtifact(_from=bias_content_json)

    return pose_refine_pipeline


def get_unidock_superop(docking_config:dict, image_dict:dict, local_executor:Executor, remote_executor:Executor) -> Steps:
    mgltools_image = image_dict.get("mgltools_image", "")
    unidock_tools_image = image_dict.get("unidock_tools_image", "")

    unidock_superop = Steps()
    unidock_superop.inputs.artifacts = {
        "receptor": InputArtifact(),
        "ligand_content_json": InputArtifact(),
        "bias_content_json": InputArtifact(optional=True)
    }
    receptor = unidock_superop.inputs.artifacts["receptor"]
    ligand_content_json = unidock_superop.inputs.artifacts["ligand_content_json"]
    bias_content_json = unidock_superop.inputs.artifacts["bias_content_json"]

    if docking_config["scoring"] == "ad4":
        gen_ad4_map_step = Step(
            name="gen-ad4-map",
            artifacts={
                "receptor": receptor,
                "ligand_content_json": ligand_content_json,
            },
            parameters={
                "docking_params":docking_config
            },
            template=PythonOPTemplate(
                gen_ad4_map_op,
                image=mgltools_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=local_executor,
        )
        unidock_superop.add(gen_ad4_map_step)
        receptor = gen_ad4_map_step.outputs.artifacts["map_dir"]

    unidock_step = Step(
        name="unidock",
        artifacts={
            "receptor": receptor, 
            "ligand_content_json": ligand_content_json,
            "bias_content_json": bias_content_json, 
        },
        parameters={
            "docking_params": docking_config,
        },
        template=PythonOPTemplate(
            run_unidock_op,
            image=unidock_tools_image,
            image_pull_policy="IfNotPresent",
        ),
        executor=remote_executor
    )
    unidock_superop.add(unidock_step)
    result_content_json = unidock_step.outputs.artifacts["result_json"]

    unidock_score_step = Step(
        name="unidock-score",
        artifacts={
            "receptor": receptor, 
            "ligand_content_json": result_content_json,
        },
        parameters={
            "docking_params": docking_config,
        },
        template=PythonOPTemplate(
            run_unidock_score_only_op,
            image=unidock_tools_image,
            image_pull_policy="IfNotPresent",
        ),
        executor=local_executor
    )
    unidock_superop.add(unidock_score_step)
    score_table = unidock_score_step.outputs.artifacts["score_table"]

    unidock_superop.outputs.artifacts["results_json"] = OutputArtifact(_from=result_content_json)
    unidock_superop.outputs.artifacts["score_table"] = OutputArtifact(_from=score_table)

    return unidock_superop


def get_unidock_pipeline_superop(config:dict, image_dict:dict, local_executor:Executor, remote_executor:Executor) -> Steps:
    image_dict = config.get("image_dict", dict())
    unidock_tools_image = image_dict.get("unidock_tools_image", "")

    docking_config = config.get("docking_params", dict())

    unidock_pipeline = Steps(name="unidock-pipeline")
    unidock_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "prepared_receptor": InputArtifact(),
        "ligands_dir": InputArtifact(),
    }

    prepared_receptor = unidock_pipeline.inputs.artifacts["prepared_receptor"]
    ligands_dir = unidock_pipeline.inputs.artifacts["ligands_dir"]

    bias_content_json = None
    next_step_list = []
    if docking_config.get("pose_refine"):
        pose_refine_method = docking_config.pop("pose_refine")
        if pose_refine_method == "unimol":
            pose_bias_step = Step(
                name="pose-refine-bias",
                artifacts={
                    "input_receptor": unidock_pipeline.inputs.artifacts["input_receptor"],
                    "ligands_dir": ligands_dir,
                },
                template=get_pose_bias_superop(docking_config, image_dict, local_executor, remote_executor),
            )
            next_step_list.append(pose_bias_step)
            bias_content_json = pose_bias_step.outputs.artifacts["bias_content_json"]
            docking_config["multi_bias"] = True

    ligprep_step = Step(
        name="ligand-prep",
        artifacts={
            "ligands_dir": ligands_dir,
        },
        template=PythonOPTemplate(
            preprocess_ligands_op,
            image=unidock_tools_image,
            image_pull_policy="IfNotPresent",
        ),
        executor=local_executor,
    )
    next_step_list.append(ligprep_step)
    unidock_pipeline.add(next_step_list)

    ligand_content_json = ligprep_step.outputs.artifacts["content_json"]

    docking_step = Step(
        name="docking",
        artifacts={
            "receptor": prepared_receptor,
            "ligand_content_json": ligand_content_json,
            "bias_content_json": bias_content_json,
        },
        template=get_unidock_superop(docking_config, image_dict, local_executor, remote_executor)
    )
    unidock_pipeline.add(docking_step)

    results_json = docking_step.outputs.artifacts["results_json"]
    score_table = docking_step.outputs.artifacts["score_table"]
    unidock_pipeline.outputs.artifacts["results_json"] = OutputArtifact(_from=results_json)
    unidock_pipeline.outputs.artifacts["score_table"] = OutputArtifact(_from=score_table)
    return unidock_pipeline


def get_parallel_unidock_pipeline_superop(config:dict, image_dict:dict, local_executor:Executor, remote_executor:Executor):
    parallel_unidock_pipeline = Steps(name="parallel-unidock-pipeline")
    parallel_unidock_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "prepared_receptor": InputArtifact(),
        "ligands_dir": InputArtifact(),
    }


def run_unimol_unidock(config:dict, out_yaml_path:str=""):
    setup(config)
    wf_ctx = None
    cpu_executor = get_local_executor()
    if os.environ.get("HERMITE_MODE"):
        from lbg_plugins import LebesgueContext, LebesgueExecutor

        hmt_lbg_config = config["hmt_lbg_config"]
        extra = hmt_lbg_config.get("extra", dict())
        wf_ctx = LebesgueContext(
            app_name=hmt_lbg_config.get("app_name", ""),
            lbg_url=hmt_lbg_config["lbg_url"],
            executor="mixed",
            extra={}
        )
        gpu_executor = LebesgueExecutor(executor="lebesgue_v2", extra=extra)
    else:
        dispatcher_config = config.get("dispatcher_config", dict())
        gpu_executor = get_dispatcher_executor(dispatcher_config)
    image_dict = config.get("image_dict", dict())
    mgltools_image = image_dict.get("mgltools_image", "")

    receptor_artifact = upload_artifact(config["receptor_file"])
    ligands_dir_artifact_list = []
    for _dir in glob.glob(os.path.join(config["ligands_dir"], "*")):
        ligands_dir_artifact = upload_artifact(Path(_dir))
        ligands_dir_artifact_list.append(ligands_dir_artifact)

    wf = Workflow(name="unidock-workflow", context=wf_ctx, parallelism=100)

    prepare_receptor_step = Step(
        name="prepare-receptor",
        artifacts={
            "input_receptor": receptor_artifact, 
        },
        template=PythonOPTemplate(
            preprocess_receptor_op,
            image=mgltools_image,
            image_pull_policy="IfNotPresent",
        ),
        executor=cpu_executor
    )
    wf.add(prepare_receptor_step)
    prepared_receptor = prepare_receptor_step.outputs.artifacts["output_receptor"]

    unidock_temp = get_unidock_pipeline_superop(config, image_dict, cpu_executor, gpu_executor)
    unidock_step_list = []
    for i, ligands_dir_artifact in enumerate(ligands_dir_artifact_list):
        unidock_step = Step(
            name=f"unidock-step-{i}",
            artifacts={
                "input_receptor": receptor_artifact,
                "prepared_receptor": prepared_receptor,
                "ligands_dir": ligands_dir_artifact,
            },
            template=unidock_temp,
        )
        unidock_step_list.append(unidock_step)
    wf.add(unidock_step_list)

    if out_yaml_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_yaml_path)), exist_ok=True)
        with open(out_yaml_path, "w") as f:
            f.write(wf.to_yaml())
    else:
        wf.submit()


def main_cli():
    parser = argparse.ArgumentParser(description="workflow:unimol-bias-unidock-docking")
    parser.add_argument("-c", "--config_file", type=str, help="config file")
    parser.add_argument("-o", "--out_yaml_path", type=str, default="", 
        help="path of output yaml, submit workflow if not provided")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)
    run_unimol_unidock(config, args.out_yaml_path)

if __name__=="__main__":
    main_cli()
