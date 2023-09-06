from typing import Union, Optional
import dflow
from dflow.plugins import bohrium
from dflow.executor import ContainerExecutor
from dflow.plugins.dispatcher import DispatcherExecutor
from lbg_plugins import LebesgueContext, LebesgueExecutor


def setup(config:dict):
    if "dflow_config" in config.keys():
        dflow_config = config.pop("dflow_config")
        for k, v in dflow_config.items():
            dflow.config[k] = v

    if "dflow_s3_config" in config.keys():
        dflow_s3_config = config.pop("dflow_s3_config")
        for k, v in dflow_s3_config.items():
            dflow.s3_config[k] = v
    if dflow.s3_config["repo_key"] == "oss-bohrium":
        from dflow.plugins.bohrium import TiefblueClient
        dflow.s3_config["storage_client"] = TiefblueClient()

    if "bohrium_config" in config.keys():
        bohrium_config = config.pop("bohrium_config")
        if "username" in bohrium_config:
            bohrium.config["username"] = bohrium_config.pop("username")
        if "password" in bohrium_config:
            bohrium.config["password"] = bohrium_config.pop("password")
        if "ticket" in bohrium_config:
            bohrium.config["ticket"] = bohrium_config.pop("ticket")
        for k, v in bohrium_config.items():
            bohrium.config[k] = v


def get_local_executor(local_mode:bool=False, docker_executable:str="docker") -> Union[None, ContainerExecutor]:
    local_executor = None
    if local_mode:
        local_executor = ContainerExecutor(docker=docker_executable)
    return local_executor


def get_dispatcher_executor(dispatcher_config:dict) -> Optional[DispatcherExecutor]:
    image = dispatcher_config.get("image")
    clean = dispatcher_config.get("clean", True)
    machine_dict = dispatcher_config.get("machine_dict", dict())
    resources_dict = dispatcher_config.get("resources_dict")
    singularity_executable = dispatcher_config.get("singularity_executable")
    container_args = dispatcher_config.get("container_args", "")
    remote_executor = DispatcherExecutor(
        image=image,
        clean=clean,
        machine_dict=machine_dict,
        resources_dict=resources_dict,
        singularity_executable=singularity_executable,
        container_args=container_args,
    )
    return remote_executor


def get_remote_executor(executor_config:dict, scass_type:str, use_lbg_executor:bool) -> Union[LebesgueExecutor, DispatcherExecutor]:
    if use_lbg_executor:
        executor_config["scass_type"] = scass_type
        remote_executor = LebesgueExecutor(executor="lebesgue_v2", extra=executor_config)
    else:
        try:
            executor_config["machine_dict"]["remote_profile"]["input_data"]["scass_type"] = scass_type
        except:
            pass
        remote_executor = get_dispatcher_executor(executor_config)
    return remote_executor