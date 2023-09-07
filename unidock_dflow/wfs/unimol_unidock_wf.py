from pathlib import Path
import os
import sys
import glob
import json
import copy
from enum import IntEnum
import argparse

from dflow import (
    Executor,
    Workflow, 
    Step,
    Steps,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    upload_artifact,
)
from dflow.python import (
    PythonOPTemplate, 
    Slices,
)
from dflow.utils import randstr

sys.path.append(os.path.dirname(__file__))
from utils import (
    setup,
    get_local_executor,
    get_remote_executor
)

ops_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ops")
sys.path.append(ops_folder_path)
from op_bias import gen_pose_refine_bpf_op, gen_hbond_bpf_op, merge_bpf_op, gen_substructure_bpf_op, insert_substructure_ind_op, gen_shape_bpf_op, gen_mcs_bpf_op
from op_preprocess import preprocess_receptor_op, preprocess_ligands_op
from op_pose_refine import run_unimol_pose_refine_op
from op_unidock import gen_ad4_map_op, run_unidock_op, run_unidock_score_only_op, read_score_op
from op_helper import collect_parallel_docking_results_op, pick_refine_ligands_op, update_results_op
from op_pbgbsa import get_pbgbsa_inputs_from_docking_op, run_pbgbsa_op


pose_refine_min_num = int(os.environ.get("POSE_REFINE_MIN_NUM", "10000"))
ligand_num_per_batch = int(os.environ.get("LIGAND_NUM_PER_BATCH", "18000"))
local_executor = None

class ConstrainedDockingType(IntEnum):
    HBond = 1
    Substructure = 2
    MCS = 3
    Shape = 4
    MetalBond = 5

def get_pose_bias_superop(docking_config:dict, image_dict:dict, remote_executor:Executor) -> Steps:
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
            gen_pose_refine_bpf_op,
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


def get_unidock_superop(docking_config:dict, image_dict:dict, remote_executor:Executor, has_bias:bool=False) -> Steps:
    mgltools_image = image_dict.get("mgltools_image", "")
    unidock_tools_image = image_dict.get("unidock_tools_image", "")

    unidock_superop = Steps()
    unidock_superop.inputs.artifacts = {
        "receptor": InputArtifact(),
        "ligand_content_json": InputArtifact(),
        "bias_file": InputArtifact(optional=True),
        "bias_content_json": InputArtifact(optional=True)
    }
    receptor = unidock_superop.inputs.artifacts["receptor"]
    ligand_content_json = unidock_superop.inputs.artifacts["ligand_content_json"]
    bias_file = unidock_superop.inputs.artifacts["bias_file"]
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
            "bias_file": bias_file,
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

    if has_bias:
        unidock_score_step = Step(
            name="unidock-score",
            artifacts={
                "receptor": receptor, 
                "result_content_json": result_content_json,
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
    else:
        unidock_score_step = Step(
            name="unidock-score",
            artifacts={
                "result_content_json": result_content_json,
            },
            template=PythonOPTemplate(
                read_score_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=local_executor
        )
    unidock_superop.add(unidock_score_step)
    score_table = unidock_score_step.outputs.artifacts["score_table"]
    result_content_json = unidock_score_step.outputs.artifacts["result_json"]

    unidock_superop.outputs.artifacts["results_json"] = OutputArtifact(_from=result_content_json)
    unidock_superop.outputs.artifacts["score_table"] = OutputArtifact(_from=score_table)

    return unidock_superop


def get_unidock_pipeline_superop(docking_config:dict, image_dict:dict, remote_executor:Executor) -> Steps:
    unidock_tools_image = image_dict.get("unidock_tools_image", "")

    unidock_pipeline = Steps(name="unidock-pipeline")
    unidock_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "prepared_receptor": InputArtifact(),
        "ligands_dir": InputArtifact(),
        "ref_sdf_file": InputArtifact(optional=True),
    }

    prepared_receptor = unidock_pipeline.inputs.artifacts["prepared_receptor"]
    ligands_dir = unidock_pipeline.inputs.artifacts["ligands_dir"]
    ref_sdf_file = unidock_pipeline.inputs.artifacts["ref_sdf_file"]

    bias_file = None
    bias_content_json = None
    next_step_list = []

    has_bias = False
    constrain_type = 0
    constrain_params = dict()
    if docking_config.get("constaint"):
        constrain_params = docking_config.pop("constaint")
        constrain_type = constrain_params.pop("constrained_type")
        if constrain_type not in [ConstrainedDockingType.HBond.value]:
            try:
                docking_config.pop("pose_refine")
            except:
                pass
        has_bias = True

    if docking_config.get("pose_refine"):
        pose_refine_method = docking_config.pop("pose_refine")
        if pose_refine_method == "unimol":
            pose_bias_step = Step(
                name="pose-refine-bias",
                artifacts={
                    "input_receptor": unidock_pipeline.inputs.artifacts["input_receptor"],
                    "ligands_dir": ligands_dir,
                },
                template=get_pose_bias_superop(docking_config, image_dict, remote_executor),
            )
            next_step_list.append(pose_bias_step)
            bias_content_json = pose_bias_step.outputs.artifacts["bias_content_json"]
            has_bias = True

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

    if constrain_type == ConstrainedDockingType.HBond:
        hbond_bias_step = Step(
            name="hbond-bias-step",
            artifacts={
                "receptor_path": unidock_pipeline.inputs.artifacts["input_receptor"],
            },
            parameters={
                "hbond_sites": constrain_params.get("h_bond_sites", ""),
            },
            template=PythonOPTemplate(
                gen_hbond_bpf_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=local_executor,
        )
        next_step_list.append(hbond_bias_step)
        bias_file = hbond_bias_step.outputs.artifacts["bias_file"]
    elif constrain_type == ConstrainedDockingType.Substructure:
        substructure_bias_step = Step(
            name="substructure-bias-step",
            artifacts={
                "ref_sdf_file": ref_sdf_file,
            },
            parameters={
                "ind_list": constrain_params.get("indices_list", []),
            },
            template=PythonOPTemplate(
                gen_substructure_bpf_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=local_executor,
        )
        next_step_list.append(substructure_bias_step)
        bias_file = substructure_bias_step.outputs.artifacts["bias_file"]
    elif constrain_type == ConstrainedDockingType.Shape:
        shape_bias_step = Step(
            name="shape-bias-step",
            artifacts={
                "ref_sdf_file": ref_sdf_file,
            },
            parameters={
                "shape_scale": constrain_params.get("shape_scale", 1),
            },
            template=PythonOPTemplate(
                gen_shape_bpf_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=local_executor,
        )
        next_step_list.append(shape_bias_step)
        bias_file = shape_bias_step.outputs.artifacts["bias_file"]

    unidock_pipeline.add(next_step_list)
    ligand_content_json = ligprep_step.outputs.artifacts["content_json"]

    if constrain_type == ConstrainedDockingType.HBond and bias_content_json:
        merge_bias_step = Step(
            name="merge-bias-step",
            artifacts={
                "bias_content_json": bias_content_json,
                "bpf_file": bias_file,
            },
            template=PythonOPTemplate(
                merge_bpf_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=local_executor,
        )
        unidock_pipeline.add(merge_bias_step)
        bias_content_json = merge_bias_step.outputs.artifacts["bias_content_json"]
        bias_file = None
    elif constrain_type == ConstrainedDockingType.Substructure:
        insert_substructure_ind_step = Step(
            name="insert-substructure-ind-step",
            artifacts={
                "ref_sdf_file": ref_sdf_file,
                "ligand_content_json": ligand_content_json,
            },
            parameters={
                "ind_list": constrain_params.get("indices_list", []),
            },
            template=PythonOPTemplate(
                insert_substructure_ind_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=local_executor,
        )
        unidock_pipeline.add(insert_substructure_ind_step)
        ligand_content_json = insert_substructure_ind_step.outputs.artifacts["ligand_content_json"]
    elif constrain_type == ConstrainedDockingType.MCS:
        gen_mcs_bpf_step = Step(
            name="gen-mcs-bpf-step",
            artifacts={
                "ref_sdf_file": ref_sdf_file,
                "ligand_content_json": ligand_content_json,
            },
            template=PythonOPTemplate(
                gen_mcs_bpf_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=local_executor,
        )
        unidock_pipeline.add(gen_mcs_bpf_step)
        bias_content_json = gen_mcs_bpf_step.outputs.artifacts["bias_content_json"]
        ligand_content_json = gen_mcs_bpf_step.outputs.artifacts["ligand_content_json"]

    docking_step = Step(
        name="docking",
        artifacts={
            "receptor": prepared_receptor,
            "ligand_content_json": ligand_content_json,
            "bias_file": bias_file,
            "bias_content_json": bias_content_json,
        },
        template=get_unidock_superop(docking_config, image_dict, remote_executor, has_bias)
    )
    unidock_pipeline.add(docking_step)

    results_json = docking_step.outputs.artifacts["results_json"]
    score_table = docking_step.outputs.artifacts["score_table"]
    unidock_pipeline.outputs.artifacts["results_json"] = OutputArtifact(_from=results_json)
    unidock_pipeline.outputs.artifacts["score_table"] = OutputArtifact(_from=score_table)
    return unidock_pipeline


def get_parallel_unidock_pipeline_superop(docking_config:dict, image_dict:dict, remote_executor:Executor) -> Steps:
    unidock_tools_image = image_dict.get("unidock_tools_image", "")
    parallel_unidock_pipeline = Steps(name=f"parallel-unidock-pipeline-temp-{randstr(5)}")
    parallel_unidock_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "prepared_receptor": InputArtifact(),
        "ligands_dir_list": InputArtifact(archive=None),
        "ref_sdf_file": InputArtifact(optional=True),
    }
    unidock_temp = get_unidock_pipeline_superop(docking_config, image_dict, remote_executor)
    parallel_unidock_pipeline_step = Step(
        name="unidock-step",
        slices=Slices(sub_path=True, input_artifact=["ligands_dir"], output_artifact=["results_json", "score_table"]),
        artifacts={
            "input_receptor": parallel_unidock_pipeline.inputs.artifacts["input_receptor"],
            "prepared_receptor": parallel_unidock_pipeline.inputs.artifacts["prepared_receptor"],
            "ligands_dir": parallel_unidock_pipeline.inputs.artifacts["ligands_dir_list"],
            "ref_sdf_file": parallel_unidock_pipeline.inputs.artifacts["ref_sdf_file"],
        },
        template=unidock_temp,
    )
    parallel_unidock_pipeline.add(parallel_unidock_pipeline_step)

    collect_result_step = Step(
        name="collect-unidock-results",
        artifacts={
            "result_content_json_list": parallel_unidock_pipeline_step.outputs.artifacts["results_json"], 
            "score_table_list": parallel_unidock_pipeline_step.outputs.artifacts["score_table"]
        },
        template=PythonOPTemplate(
            collect_parallel_docking_results_op,
            image=unidock_tools_image,
            image_pull_policy="IfNotPresent",
        ),
        executor=local_executor,
    )
    parallel_unidock_pipeline.add(collect_result_step)

    parallel_unidock_pipeline.outputs.artifacts["results_json_list"] = OutputArtifact(_from=collect_result_step.outputs.artifacts["result_json_list"])
    parallel_unidock_pipeline.outputs.artifacts["results_table"] = OutputArtifact(_from=collect_result_step.outputs.artifacts["total_score_table"])
    return parallel_unidock_pipeline


def get_top_refine_docking_pipeline_superop(docking_config:dict, image_dict:dict, 
        remote_executor:Executor, ligand_num:int) -> Steps:
    unidock_tools_image = image_dict.get("unidock_tools_image", "")
    top_refine_pipeline = Steps(name="top-refine-docking-pipeline")
    top_refine_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "prepared_receptor": InputArtifact(),
        "ligands_dir_list": InputArtifact(archive=None),
        "ref_sdf_file": InputArtifact(optional=True),
    }
    input_receptor = top_refine_pipeline.inputs.artifacts["input_receptor"]
    prepared_receptor = top_refine_pipeline.inputs.artifacts["prepared_receptor"]
    ligands_dir_list = top_refine_pipeline.inputs.artifacts["ligands_dir_list"]

    if (docking_config.get("constaint") and docking_config["constaint"].get("constrained_type") not in [ConstrainedDockingType.HBond.value]) or not docking_config.get("pose_refine") or ligand_num < pose_refine_min_num:
        parallel_unidock_temp = get_parallel_unidock_pipeline_superop(docking_config, image_dict, 
            remote_executor)
        parallel_unidock_step = Step(
            name="parallel-unidock-step",
            artifacts={
                "input_receptor": input_receptor,
                "prepared_receptor": prepared_receptor,
                "ligands_dir_list": ligands_dir_list,
                "ref_sdf_file": top_refine_pipeline.inputs.artifacts["ref_sdf_file"],
            },
            template=parallel_unidock_temp,
        )
        top_refine_pipeline.add(parallel_unidock_step)
        update_refine_result_step = Step(
            name="update-pose-refine-results-step",
            artifacts={
                "old_results_json_list": parallel_unidock_step.outputs.artifacts["results_json_list"], 
                "old_score_table": parallel_unidock_step.outputs.artifacts["results_table"], 
            },
            template=PythonOPTemplate(
                update_results_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
        )
        top_refine_pipeline.add(update_refine_result_step)
    else:
        pure_docking_config = copy.deepcopy(docking_config)
        pure_docking_config.pop("pose_refine")
        pure_docking_parallel_unidock_temp = get_parallel_unidock_pipeline_superop(pure_docking_config, image_dict, 
            remote_executor)
        pure_parallel_unidock_step = Step(
            name="parallel-unidock-s1",
            artifacts={
                "input_receptor": input_receptor,
                "prepared_receptor": prepared_receptor,
                "ligands_dir_list": ligands_dir_list,
                "ref_sdf_file": top_refine_pipeline.inputs.artifacts["ref_sdf_file"],
            },
            template=pure_docking_parallel_unidock_temp,
        )
        top_refine_pipeline.add(pure_parallel_unidock_step)

        pose_refine_num = max(pose_refine_min_num, int(ligand_num//10))
        pick_top_ligands_step = Step(
            name="pick-top-results",
            artifacts={
                "results_json_list": pure_parallel_unidock_step.outputs.artifacts["results_json_list"],
                "results_table": pure_parallel_unidock_step.outputs.artifacts["results_table"],
            },
            parameters={
                "refine_topnum": pose_refine_num, 
                "ligand_num_per_batch": ligand_num_per_batch,
            },
            template=PythonOPTemplate(
                pick_refine_ligands_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
        )
        top_refine_pipeline.add(pick_top_ligands_step)

        unimol_docking_parallel_unidock_temp = get_parallel_unidock_pipeline_superop(docking_config, image_dict, 
            remote_executor)
        unimol_parallel_unidock_step = Step(
            name="parallel-unidock-s2",
            artifacts={
                "input_receptor": input_receptor,
                "prepared_receptor": prepared_receptor,
                "ligands_dir_list": pick_top_ligands_step.outputs.artifacts["ligands_dir_list"],
                "ref_sdf_file": top_refine_pipeline.inputs.artifacts["ref_sdf_file"],
            },
            template=unimol_docking_parallel_unidock_temp,
        )
        top_refine_pipeline.add(unimol_parallel_unidock_step)

        update_refine_result_step = Step(
            name="update-pose-refine-results",
            artifacts={
                "old_results_json_list": pure_parallel_unidock_step.outputs.artifacts["results_json_list"], 
                "new_results_json_list": unimol_parallel_unidock_step.outputs.artifacts["results_json_list"],
                "old_score_table": pure_parallel_unidock_step.outputs.artifacts["results_table"], 
                "new_score_table": unimol_parallel_unidock_step.outputs.artifacts["results_table"],
            },
            template=PythonOPTemplate(
                update_results_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
            ),
        )
        top_refine_pipeline.add(update_refine_result_step)
    
    top_refine_pipeline.outputs.artifacts["results_json_list"] = OutputArtifact(_from=update_refine_result_step.outputs.artifacts["results_json_list"])
    top_refine_pipeline.outputs.artifacts["score_table"] = OutputArtifact(_from=update_refine_result_step.outputs.artifacts["score_table"])
    
    return top_refine_pipeline


def run_unimol_unidock(config:dict, out_yaml_path:str=""):
    setup(config)
    wf_ctx = None
    remote_config = None
    local_executor = get_local_executor()
    if os.environ.get("HERMITE_MODE"):
        from lbg_plugins import LebesgueContext

        hmt_lbg_config = config["hmt_lbg_config"]
        remote_config = hmt_lbg_config.get("extra", dict())
        wf_ctx = LebesgueContext(
            app_name=hmt_lbg_config.get("app_name", ""),
            lbg_url=hmt_lbg_config["lbg_url"],
            executor="mixed",
            extra={}
        )
    else:
        remote_config = config.get("dispatcher_config", dict())
    
    step_params_list = config.get("steps", [])

    receptor_artifact = upload_artifact(config["receptor_file"])
    sub_dirs_list = [Path(f) for f in glob.glob(os.path.join(config["ligands_dir"], "*"))]
    ligand_num = 0
    for sub_dir in sub_dirs_list:
        ligand_num += len(glob.glob(os.path.join(sub_dir.as_posix(), "*.sdf")))
        ligand_num += len(glob.glob(os.path.join(sub_dir.as_posix(), "*.mol")))
        ligand_num += len(glob.glob(os.path.join(sub_dir.as_posix(), "*.smi")))
    ligands_dir_artifact_list = upload_artifact(sub_dirs_list, archive=None)
    ref_ligand_artifact = None
    if config.get("ref_ligand"):
        ref_ligand_artifact = upload_artifact(config["ref_ligand"])

    wf = Workflow(name="vs-workflow", context=wf_ctx, parallelism=100)
    for step_params in step_params_list:
        name = step_params["name"]
        image_dict = step_params["image_dict"]
        params = step_params["params"]
        scass_type = step_params["scass_type"]

        if name == "docking":
            remote_executor = get_remote_executor(remote_config, scass_type, True if os.environ.get("HERMITE_MODE") else False)
            mgltools_image = image_dict.get("mgltools_image", "")
            unidock_tools_image = image_dict.get("unidock_tools_image", "")

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
                executor=local_executor
            )
            wf.add(prepare_receptor_step)
            prepared_receptor = prepare_receptor_step.outputs.artifacts["output_receptor"]

            top_refine_unidock_pipeline_step = Step(
                name="top-refine-unidock-pipeline-step",
                artifacts={
                    "input_receptor": receptor_artifact,
                    "prepared_receptor": prepared_receptor,
                    "ligands_dir_list": ligands_dir_artifact_list,
                    "ref_sdf_file": ref_ligand_artifact,
                },
                template=get_top_refine_docking_pipeline_superop(params, image_dict, remote_executor, ligand_num),
            )
            wf.add(top_refine_unidock_pipeline_step)
    
        elif name == "pbgbsa":
            remote_executor = get_remote_executor(remote_config, scass_type, True if os.environ.get("HERMITE_MODE") else False)
            pbgbsa_image = image_dict.get("pbgbsa_image", "")
            pbgbsa_get_ligands_step = Step(
                name="pbgbsa-get-docking-result-ligands-step",
                artifacts={
                    "results_json_list": top_refine_unidock_pipeline_step.outputs.artifacts["results_json_list"],
                },
                template=PythonOPTemplate(
                    get_pbgbsa_inputs_from_docking_op,
                    image=unidock_tools_image,
                    image_pull_policy="IfNotPresent",
                ),
            )
            wf.add(pbgbsa_get_ligands_step)

            pbgbsa_step = Step(
                name="pbgbsa-step",
                artifacts={
                    "receptor_file": receptor_artifact, 
                    "ligands_dir": pbgbsa_get_ligands_step.outputs.artifacts["ligands_dir"],
                },
                parameters={
                    "pbgbsa_params": params,
                },
                template=PythonOPTemplate(
                    run_pbgbsa_op,
                    image=pbgbsa_image,
                    image_pull_policy="IfNotPresent",
                ),
                executor=remote_executor,
            )
            wf.add(pbgbsa_step)

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
