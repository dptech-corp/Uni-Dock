from typing import List
import os
import shutil
from functools import reduce
import re
import math
import uuid
import logging


AD_ATM_TYPES = ["H", "HD", "HS", "C", "A", "N", "NA", "NS", "OA", "OS", "F", "Mg", "MG", "P", "SA", "S", "Cl", "CL",
        "Ca", "CA", "Mn", "MN", "Fe", "FE", "Zn", "ZN", "Br", "BR", "I", "Z", "G", "GA", "J", "Q"]

ATYPE_MAP = {'HD': 'H', 'HS': 'H', 'A': 'C', 'NA': 'N', 'NS': 'N', 
    'OA': 'O','OS': 'O', 'SA': 'S', 'G': 'C', 
    'J': 'C', 'Z': 'C', 'GA': 'C', 'Q': 'C'}


class AD4Mapper:
    def __init__(self, center:List[float]=[15, 15, 15], 
            box_size:List[float]=[25, 25, 25], spacing:float=0.25):

        script_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autodock_scripts"))
        os.environ["prepare_gpf4"] = "pythonsh " + os.path.join(script_env, "prepare_gpf.py")

        self.center = center
        self.box_size = box_size
        self.spacing = spacing

    def find_ad4_ligand_types(self, ligfile:str, profile:str) -> List[str]:
        '''
        The function finds the atom types of the ligand in the ligand file and the atom types of the atoms
        in the protein file.
        
        Args:
        ligfile: the ligand file
        profile: the protein file
        
        Returns:
        A list of atom types that are present in the ligand.
        '''
        atms = set()
        with open(ligfile, "r") as f:
            ls = f.readlines()
        for l in ls:
            l = l.strip("\n")
            if l.find("REMARK") != -1: continue
            l = l[76:]
            if l == "": continue
            l = re.sub(r"\s", "", l)
            if l in AD_ATM_TYPES:
                atms.add(l)

        with open(profile, "r") as f:
            ls = f.readlines()
        for l in ls:
            l = l.strip("\n")
            if l == "": continue
            if l[0:4] == "ATOM":
                atom_name = l[76:].strip()
                # if atom_name in atype_map:
                #     atom_name = atype_map[atom_name]
                if atom_name in AD_ATM_TYPES:
                    atms.add(atom_name)
        atms = list(atms)
        return atms

    def generate_ad4_map(self, protein:str, ligands:List[str], mapdir:str=None) -> str:
        '''
        Generate .map files for ad4
        
        Args:
        protein: Protein file
        ligands: List of ligand files
        mapdir: Result directory name
        
        Returns:
        The directory name of the map.
        '''
        if not mapdir:
            mapdir = os.path.join("ad4map", uuid.uuid4().hex)
        os.makedirs(mapdir, exist_ok=True)

        center = self.center
        box_size = self.box_size
        spacing = self.spacing

        protein_name = os.path.splitext(os.path.basename(protein))[0]
        shutil.copyfile(protein, os.path.join(mapdir, protein_name + '.pdbqt'))
        
        atom_types = list(set(reduce(lambda x,y: x + y, [self.find_ad4_ligand_types(ligand, protein) for ligand in ligands])))
        logging.warning("{}".format(str(atom_types)))
        npts = [math.ceil(s / spacing) for s in box_size]
        cmd = []
        cmd.append('cd %s' % mapdir)
        cmd.append('$prepare_gpf4 ' + '-r ' + os.path.basename(protein) + 
        ' -p gridcenter="' + str(center[0]) + ',' + str(center[1]) + ',' + 
        str(center[2]) + '"' + ' -p npts="' + str(npts[0]) + ',' + 
        str(npts[1]) + ',' + str(npts[2]) + '"' + ' -p spacing=' + str(spacing) + 
        ' -p ligand_types=' + '\"' +  ','.join(atom_types) + '\"' + ' -o ' + 
        protein_name + '.gpf')
        cmd.append('autogrid4 -p ' + protein_name +'.gpf -l ' + protein_name + '.glg')
        os.system("\n".join(cmd))
        return mapdir