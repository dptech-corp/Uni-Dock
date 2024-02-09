from typing import List
from rdkit import Chem

from . import cdpkit, obabel

REGISTRY_CONF_GEN_ENGINE_DICT = {
    "confgen": cdpkit.CDPKitConfGenerator,
    "obabel": obabel.OBabelConfGenerator,
}


def generate_conf(mol: Chem.Mol, engine: str = "default", *args, **kwargs) -> List[Chem.Mol]:
    for name, cls in REGISTRY_CONF_GEN_ENGINE_DICT.items():
        if engine == "default":
            if cls.check_env():
                break
        else:
            if name == engine:
                if not cls.check_env():
                    raise RuntimeError(f"{engine} is not installed")
                break
    else:
        raise ModuleNotFoundError("Engine is not found or dependencies are not installed")

    return cls().generate_conformation(mol, *args, **kwargs)