from typing import List
from pathlib import Path

from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def gen_multi_bpf_op(refine_content_json:Artifact(Path)) -> {"bias_content_json": Artifact(Path)}:
    import os
    import json
    from tqdm import tqdm
    from rdkit import Chem
    from unidock_tools.ligand_prepare.atom_type import AtomType

    RADIUS = { "1":1.08, 
          "2":1.4, 
          "5":1.47, 
          "6":1.49,
          "7":1.41,
          "8":1.4,
          "9":1.39,
          "11":1.84,
          "12":2.05,
          "13":2.11,
          "14":2.1,
          "15":1.92,
          "16":1.82,
          "17":1.83,
          "19":2.05,
          "20":2.21,
          "25":1.97,
          "26":1.94,
          "30":2.1,
          "35":1.98,
          "53":2.23, }
    VSET = -1.0

    with open(refine_content_json.as_posix(), "r") as f:
        refine_content_list = json.load(f)
    ligands_dir = Path("./tmp/ligands_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)
    input_ligands = []
    for refine_content_item in refine_content_list:
        basename = refine_content_item["name"]
        content = refine_content_item["content"]
        with open(ligands_dir / basename, "w") as f:
            f.write(content)
        input_ligands.append(ligands_dir / basename)

    input_ligands = [p.as_posix() for p in input_ligands]

    bias_content_list = []
    for ligand_file in tqdm(input_ligands):
        ligand_name = os.path.splitext(os.path.basename(ligand_file))[0]
        target_mol = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=False, strictParsing=False)[0]

        atom_info_map = AtomType().get_docking_atom_types(target_mol)

        bias_content = "x y z Vset r type atom\n"
        for atom in target_mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_type = atom_info_map[atom_idx]
            atomic_num = atom.GetAtomicNum()
            atom_radius = RADIUS[str(atomic_num)]
            position = target_mol.GetConformer().GetAtomPosition(atom_idx)
            bias_content += f'{position.x:6.3f} {position.y:6.3f} {position.z:6.3f} {VSET:6.2f} {atom_radius:6.2f} map {atom_type:<2s}\n'

        bias_content_list.append({"name": ligand_name, "content": bias_content})

    result_file_path = "result_bias.json"
    with open(result_file_path, "w") as f:
        json.dump(bias_content_list, f)
    
    return OPIO({"bias_content_json": Path(result_file_path)})