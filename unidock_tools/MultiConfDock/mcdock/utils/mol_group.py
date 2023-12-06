from typing import List, Any
from copy import deepcopy
from pathlib import Path
from .utils import makedirs, generate_random_string
from .sdf_process import sdf_writer
import os
import logging
from rdkit import Chem 

class MoleculeGroup:
    def __init__(self, ligands: List[Path]):
        self.molecule_dict, self.property_dict, self.index_list = self.initialize(ligands)
        
    def initialize(self, ligands: List[Path]):
        molecule_dict = {}
        property_dict = {}
        index_list = []
        for order, ligand in enumerate(ligands):
            file_prefix = ligand.stem
            index = generate_random_string(8)
            molecule_list = [mol for mol in Chem.SDMolSupplier(str(ligand), removeHs=False) if mol is not None]
            properties = self._get_properties(molecule_list[0])
            molecule_list = [self._clean_molecule(mol) for mol in molecule_list]
            molecule_dict[index] = molecule_list
            property_dict[index] = {
                "_input_order": order,
                "file_prefix": file_prefix, 
                **properties,
            }
            index_list.append(index)
        return molecule_dict, property_dict, index_list
    
    def _get_properties(self, molecule:Chem.Mol):
        return {prop: molecule.GetProp(prop) for prop in molecule.GetPropNames()}
    
    def _clean_molecule(self, molecule:Chem.Mol):
        molecule = deepcopy(molecule)
        for prop in molecule.GetPropNames(): 
            molecule.ClearProp(prop)
        return molecule
    
    def __iter__(self):
        for molecule_index in self.index_list:
            yield (
                molecule_index, 
                self.molecule_dict[molecule_index], 
                self.property_dict[molecule_index]
            )
            
    def update_property(self, index:str, property_name:str, value:Any):
        if index not in self.molecule_dict:
            logging.error(f"Cannot find {index} in MoleculeGroup")
            return
        if isinstance(value, list):
            if len(value) != len(self.molecule_dict[index]):
                logging.warning(f"Length of value is not equal to length of mol")
        self.property_dict[index][property_name] = value
        
    def update_molecule_list(self, index:str, molecule_list:List[Chem.Mol]):
        if index not in self.molecule_dict:
            logging.error(f"Cannot find {index} in MoleculeGroup")
            return
        if not isinstance(molecule_list, list):
            logging.warning(f"molecule_list should be list")
            molecule_list = [molecule_list]
        self.molecule_dict[index] = [self._clean_molecule(mol) for mol in molecule_list]
    
    def update_molecule_list_by_file_prefix(self, file_prefix: str, molecule_list:List[Chem.Mol]):
        file_prefix_dict = {data["file_prefix"]: index for index, data in self.property_dict.items()}
        if file_prefix not in file_prefix_dict:
            logging.error(f"Cannot find {file_prefix} in MoleculeGroup")
            return
        self.update_molecule_list(file_prefix_dict[file_prefix], molecule_list)

    def update_property_by_file_prefix(self, file_prefix: str, property_name: str, value: Any):
        file_prefix_dict = {data["file_prefix"]: index for index, data in self.property_dict.items()}
        if file_prefix not in file_prefix_dict:
            logging.error(f"Cannot find {file_prefix} in MoleculeGroup")
            return
        self.update_property(file_prefix_dict[file_prefix], property_name, value)
    
    @staticmethod
    def _set_property(property_name:str, property_value:str, molecule:Chem.Mol):
        if property_name in ["fraginfo_all", "fraginfo"]:
            try: molecule.ClearProp("fragInfo")
            except: pass
            molecule.SetProp("fragInfo", property_value)
        elif property_name == "torsioninfo":
            molecule.SetProp("torsionInfo", property_value)
        elif property_name == "atominfo":
            molecule.SetProp("atomInfo", property_value)
        else:
            molecule.SetProp(property_name, property_value)
        return molecule
    
    def get_index(self):
        return self.index_list
    
    def get_info_by_index(self, index:str):
        return (self.molecule_dict[index], 
                self.property_dict[index])
    
    def get_sdf_by_index(self, 
        index:str, 
        properties: List[str] = [], 
        save_dir: Path = None,
        seperate_confs: bool = False
    ):
        if save_dir is None: 
            save_dir = makedirs("sdf", True, True)
        if not os.path.exists(save_dir): 
            save_dir = makedirs(save_dir, False, False)
        
        sdf_list = []
        
        molecule_list, property_dict = self.get_info_by_index(index)
        mol_list = []
        for idx, molecule in enumerate(molecule_list):
            copied_molecule = deepcopy(molecule)
            if "all" in properties: 
                properties = [p for p in property_dict.keys() if not p.startswith("_")]
            for prop in properties: 
                if prop == "fraginfo_all": continue
                if isinstance(property_dict[prop], list): 
                    property_value = property_dict[prop][idx]
                else: 
                    property_value = property_dict[prop]
                copied_molecule = self._set_property(prop, str(property_value), copied_molecule)
            mol_list.append(copied_molecule)
        # save SDF files
        if seperate_confs:
            for idx, mol in enumerate(mol_list): 
                save_name = f"{save_dir}/{property_dict['file_prefix']}_CONF{idx}.sdf"
                sdf_writer([mol], save_name)
                sdf_list.append(Path(save_name))
        else:
            save_name = f"{save_dir}/{property_dict['file_prefix']}.sdf"
            sdf_writer(mol_list, save_name)
            sdf_list.append(Path(save_name))
        return sdf_list
    

    def get_sdf(self, 
        properties: List[str] = [], 
        save_dir: Path = None,
        seperate_confs: bool = False,
        additional_suffix: str = ""
    ):
        if save_dir is None: 
            save_dir = makedirs("sdf", True, True)
        if not os.path.exists(save_dir): 
            save_dir = makedirs(save_dir, False, False)
        
        sdf_list = []
        for _, molecule_list, property_dict in self.__iter__():
            mol_list = []
            for idx, molecule in enumerate(molecule_list):
                copied_molecule = deepcopy(molecule)
                if "all" in properties: 
                    properties = [p for p in property_dict.keys() if not p.startswith("_")]
                for prop in properties: 
                    if prop == "fraginfo_all": continue
                    if isinstance(property_dict[prop], list): 
                        property_value = property_dict[prop][idx]
                    else: 
                        property_value = property_dict[prop]
                    copied_molecule = self._set_property(prop, str(property_value), copied_molecule)
                mol_list.append(copied_molecule)
            # save SDF files
            if seperate_confs:
                for idx, mol in enumerate(mol_list): 
                    save_name = f"{save_dir}/{property_dict['file_prefix']}_CONF{idx}{additional_suffix}.sdf"
                    sdf_writer([mol], save_name)
                    sdf_list.append(Path(save_name))
            else:
                save_name = f"{save_dir}/{property_dict['file_prefix']}{additional_suffix}.sdf"
                sdf_writer(mol_list, save_name)
                sdf_list.append(Path(save_name))
        return sdf_list