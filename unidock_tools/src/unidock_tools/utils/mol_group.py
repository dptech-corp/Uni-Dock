from typing import List, Generator, Any, Optional, Union
from pathlib import Path
import os
import copy
import logging
import math
from multiprocessing import Pool
from rdkit import Chem

from .string import make_tmp_dir, randstr
from .rdkit_helper import sdf_writer, set_properties, clear_properties
from .read_ligand import read_ligand


class Mol:
    def __init__(self, mol: Chem.Mol, props: dict):
        props.update(mol.GetPropsAsDict())
        self.props = props
        self.conf_props = dict()
        mol = __class__.clear_rdkit_props(mol)
        self.mol_confs = [mol]

    def __len__(self):
        return len(self.mol_confs)

    @staticmethod
    def clear_rdkit_props(mol: Chem.Mol) -> Chem.Mol:
        mol = copy.copy(mol)
        for prop in mol.GetPropNames(): 
            mol.ClearProp(prop)
        return mol

    def get_prop(self, key: str, default_value: Optional[Any] = None) -> Any:
        return self.props.get(key, default_value)

    def get_props(self) -> dict:
        return self.props

    def get_conf_props(self) -> dict:
        return self.conf_props
    
    def get_mol_confs(self) -> List[Chem.Mol]:
        return self.mol_confs

    def get_first_mol(self) -> Chem.Mol:
        return self.mol_confs[0]

    def update_mol_confs(self, mol_confs: List[Chem.Mol]):
        self.mol_confs = mol_confs

    def update_props(self, props: dict):
        self.props.update(props)

    def update_conf_props(self, conf_props: dict):
        assert all([len(self.mol_confs) == len(conf_props[prop]) for prop in conf_props]), \
            "conf props length should be same as mol_confs length"
        self.conf_props.update(conf_props)

    def get_rdkit_mol_conf_with_props(self, conf_idx: int, props_list: List[str] = [], 
                                      exclude_props_list: List[str] = []) -> Chem.Mol:
        mol = copy.copy(self.mol_confs[conf_idx])
        props = copy.deepcopy(self.get_props())
        props.update({k:v[conf_idx] for k, v in self.get_conf_props().items()})
        if props_list:
            props = {k:v for k, v in props.items() if k in props_list}
            if "fragAllInfo" in props:
                frag_all_info_str = props.pop("fragAllInfo")
                props["fragInfo"] = frag_all_info_str
        if exclude_props_list:
            props = {k:v for k, v in props.items() if k not in exclude_props_list}
        set_properties(mol, props)
        return mol


class MolGroup:
    def __init__(self, ligand_files: List[Path]):
        self.mol_group: List[Mol] = list()
        self._initialize(ligand_files)

    def __iter__(self):
        for mol in self.mol_group:
            yield mol

    def iter_idx_list(self, batch_size: int) -> Generator[List[int], None, None]:
        real_batch_size = math.ceil(len(self.mol_group) / math.ceil(len(self.mol_group) / batch_size))
        batch_id_list = [list(range(i, min(len(self.mol_group), i + real_batch_size))) for i in range(0, len(self.mol_group), real_batch_size)]
        for sub_id_list in batch_id_list:
            yield sub_id_list

    def _initialize(self, ligand_files: List[Path]):
        for ligand_file in ligand_files:
            file_prefix = ligand_file.stem
            mols = read_ligand(ligand_file)
            for i, mol in enumerate(mols):
                if mol:
                    self.mol_group.append(Mol(mol, {"file_prefix": f"{file_prefix}_{i}" if len(mols) > 1 
                                                    else file_prefix}))

    def update_property_by_idx(self, idx: int, property_name: str, value: Any, is_conf_prop: bool = False):
        if is_conf_prop:
            self.mol_group[idx].update_conf_props({property_name: value})
        else:
            self.mol_group[idx].update_props({property_name: value})

    def update_mol_confs(self, idx: int, mol_confs: List[Chem.Mol]):
        if not isinstance(mol_confs, list):
            logging.warning(f"molecule_list should be list")
            mol_confs = [mol_confs]
        self.mol_group[idx].update_mol_confs([clear_properties(mol) for mol in mol_confs])

    def update_mol_confs_by_file_prefix(self, file_prefix: str, mol_confs_list: List[Chem.Mol]):
        file_prefix_dict = {mol.get_prop("file_prefix", ""): idx for idx, mol in enumerate(self.mol_group)}
        logging.debug(file_prefix_dict)
        if file_prefix not in file_prefix_dict:
            logging.error(f"Cannot find {file_prefix} in MoleculeGroup")
            return
        self.update_mol_confs(file_prefix_dict[file_prefix], mol_confs_list)

    def update_property_by_file_prefix(self, file_prefix: str, 
                                       property_name: str, value: Any, is_conf_prop: bool = False):
        file_prefix_dict = {mol.get_prop("file_prefix", ""): idx for idx, mol in enumerate(self.mol_group)}
        logging.debug(file_prefix_dict)
        if file_prefix not in file_prefix_dict:
            logging.error(f"Cannot find {file_prefix} in MoleculeGroup")
            return
        self.update_property_by_idx(file_prefix_dict[file_prefix], property_name, value, is_conf_prop)

    def write_sdf_by_idx(self,
                         idx: int,
                         save_dir: Union[str, os.PathLike],
                         seperate_conf: bool = False,
                         conf_prefix: str = "_CONF",
                         props_list: List[str] = [],
                         exclude_props_list: List[str] = [],
                         ) -> List[Path]:
        save_dir = make_tmp_dir(str(save_dir), False, False)

        mol_confs_copy = [self.mol_group[idx].get_rdkit_mol_conf_with_props(
            conf_idx, props_list, exclude_props_list) for conf_idx in range(
                len(self.mol_group[idx]))]
        # save SDF files
        file_prefix = self.mol_group[idx].get_props()['file_prefix']
        sdf_file_list = []
        if seperate_conf:
            for conf_id, mol_conf in enumerate(mol_confs_copy):
                save_name = f"{save_dir}/{file_prefix}{conf_prefix}{conf_id}.sdf"
                sdf_writer([mol_conf], save_name)
                sdf_file_list.append(Path(save_name))
        else:
            save_name = f"{save_dir}/{file_prefix}.sdf"
            sdf_writer(mol_confs_copy, save_name)
            sdf_file_list.append(Path(save_name))
        return sdf_file_list

    def write_sdf(self, save_dir: Union[str, os.PathLike],
                  seperate_conf: bool = False,
                  conf_prefix: str = "_CONF",
                  props_list: List[str] = [],
                  exclude_props_list: List[str] = []) -> List[Path]:
        result_files = []
        for idx in range(len(self.mol_group)):
            result_files.extend(self.write_sdf_by_idx(idx=idx, save_dir=save_dir,
                                                      seperate_conf=seperate_conf,
                                                      conf_prefix=conf_prefix,
                                                      props_list=props_list,
                                                      exclude_props_list=exclude_props_list))
        return result_files
