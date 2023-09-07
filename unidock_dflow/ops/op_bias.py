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


@OP.function
def gen_pose_refine_bpf_op(refine_content_json:Artifact(Path)) -> {"bias_content_json": Artifact(Path)}:
    import os
    import json
    from tqdm import tqdm
    from rdkit import Chem
    from unidock_tools.ligand_prepare.atom_type import AtomType

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


@OP.function
def gen_hbond_bpf_op(receptor_path:Artifact(Path), hbond_sites:Parameter(str)) -> {"bias_file": Artifact(Path)}:
    from unidock_tools.generate_bias.hbond import HBondBias

    runner = HBondBias(receptor_path.as_posix(), hbond_sites)

    runner.gen_hbond_bias("hbond_bias.bpf")

    return OPIO({"bias_file": Path("hbond_bias.bpf")})


@OP.function
def merge_bpf_op(bias_content_json:Artifact(Path), bpf_file:Artifact(Path)) -> {"bias_content_json": Artifact(Path)}:
    import json

    with open(bpf_file, "r") as f:
        bpf_lines = f.readlines()

    with open(bias_content_json, "r") as f:
        bias_content_list = json.load(f)

    if len(bpf_lines) > 1:
        bpf_content = "".join(bpf_lines[1:])
        for i in range(len(bias_content_list)):
            bias_content_list[i]["content"] += bpf_content

    with open("new_result_bias.json", "w") as f:
        json.dump(bias_content_list, f)

    return OPIO({"bias_content_json": Path("new_result_bias.json")})


@OP.function
def gen_substructure_bpf_op(ref_sdf_file:Artifact(Path), ind_list:Parameter(list)) -> {"bias_file": Artifact(Path)}:
    from rdkit import Chem
    from unidock_tools.ligand_prepare.atom_type import AtomType

    ind_list = [int(i) for i in ind_list]
    ref_mol = Chem.SDMolSupplier(ref_sdf_file.as_posix(), removeHs=False, sanitize=True)[0]
    atom_info_map = AtomType().get_docking_atom_types(ref_mol)

    with open("substructure_bias.bpf", "w") as f:
        f.write("x y z Vset r type atom\n")
        for atom in ref_mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom_idx in ind_list:
                atom_type = atom_info_map[atom_idx]
                atomic_num = atom.GetAtomicNum()
                atom_radius = RADIUS[str(atomic_num)]
                position = ref_mol.GetConformer().GetAtomPosition(atom_idx)
                #f.write("%6.3f %6.3f %6.3f %s %s map %s\n"%(position.x, position.y, position.z, str(Vset), str(atom_radius), atom_type))
                f.write(f'{position.x:6.3f} {position.y:6.3f} {position.z:6.3f} {VSET:6.2f} {atom_radius:6.2f} map {atom_type:<2s}\n')
    
    return OPIO({"bias_file": Path("substructure_bias.bpf")})


@OP.function
def insert_substructure_ind_op(ref_sdf_file:Artifact(Path), ind_list:Parameter(list), 
            ligand_content_json:Artifact(Path)) -> {"ligand_content_json": Artifact(Path)}:
    import os
    from io import StringIO
    import json
    from tqdm import tqdm
    from rdkit import Chem

    ind_list = [int(i) for i in ind_list]
    ref_mol = Chem.SDMolSupplier(ref_sdf_file.as_posix(), removeHs=False, sanitize=True)[0]
    sub_mol = Chem.RWMol()
    old_to_new_atom_indices = dict()
    for idx in ind_list:
        atom = ref_mol.GetAtomWithIdx(idx)
        atom_idx = sub_mol.AddAtom(atom)
        old_to_new_atom_indices[idx] = atom_idx
    for bond in ref_mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        if begin_atom_idx in old_to_new_atom_indices and end_atom_idx in old_to_new_atom_indices:
            begin_atom_new_idx = old_to_new_atom_indices[begin_atom_idx]
            end_atom_new_idx = old_to_new_atom_indices[end_atom_idx]
            bond_type = bond.GetBondType()
            sub_mol.AddBond(begin_atom_new_idx, end_atom_new_idx, bond_type)
    sub_mol = Chem.Mol(sub_mol)
    with open(ligand_content_json.as_posix(), "r") as f:
        ligand_content_list = json.load(f)
    ligands_dir = Path("./tmp/ligands_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)

    for i, ligand_content_item in tqdm(enumerate(ligand_content_list)):
        basename = ligand_content_item["name"]
        content = ligand_content_item["content"]
        ligand_path = os.path.join(ligands_dir, basename)
        with open(ligand_path, "w") as f:
            f.write(content)
        target_mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)[0]
        match_indices_list = target_mol.GetSubstructMatches(sub_mol)
        if match_indices_list:
            match_str = ""
            for match_indices in match_indices_list:
                match_str += ",".join([str(i) for i in match_indices])
                match_str += ";"
            if match_str[-1] == ";":
                match_str = match_str[:-1]
            print(match_str)
            target_mol.SetProp("subMatchInd", match_str)
            sdf_io = StringIO()
            with Chem.SDWriter(sdf_io) as writer:
                writer.write(target_mol)
            sdf_content = sdf_io.getvalue()
            ligand_content_list[i]["content"] = sdf_content
    
    with open("ligand_contents.json", "w") as f:
        json.dump(ligand_content_list, f)
    return OPIO({"ligand_content_json": Path("ligand_contents.json")})


@OP.function
def gen_shape_bpf_op(ref_sdf_file:Artifact(Path), shape_scale:Parameter(float)) -> {"bias_file": Artifact(Path)}:
    from rdkit import Chem
    from unidock_tools.ligand_prepare.atom_type import AtomType

    ref_mol = Chem.SDMolSupplier(ref_sdf_file.as_posix(), removeHs=False, sanitize=True)[0]
    atom_info_map = AtomType().get_docking_atom_types(ref_mol)

    with open("shape_bias.bpf", "w") as f:
        f.write("x y z Vset r type atom\n")
        for atom in ref_mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_type = atom_info_map[atom_idx]
            atomic_num = atom.GetAtomicNum()
            atom_radius = RADIUS[str(atomic_num)]
            position = ref_mol.GetConformer().GetAtomPosition(atom_idx)
            f.write(f'{position.x:6.3f} {position.y:6.3f} {position.z:6.3f} {VSET:6.2f} {atom_radius*shape_scale:6.2f} map {atom_type:<2s}\n')
    
    return OPIO({"bias_file": Path("shape_bias.bpf")})


@OP.function
def gen_mcs_bpf_op(ref_sdf_file:Artifact(Path), 
        ligand_content_json:Artifact(Path)) -> {
"bias_content_json": Artifact(Path), "ligand_content_json": Artifact(Path)}:
    import os
    import json
    from io import StringIO
    from tqdm import tqdm
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    from unidock_tools.ligand_prepare.atom_type import AtomType

    ref_mol = Chem.SDMolSupplier(ref_sdf_file.as_posix(), removeHs=False, sanitize=True)[0]

    with open(ligand_content_json.as_posix(), "r") as f:
        ligand_content_list = json.load(f)
    ligands_dir = Path("./tmp/ligands_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)

    bias_content_list = list()
    for i, ligand_content_item in tqdm(enumerate(ligand_content_list)):
        basename = ligand_content_item["name"]
        content = ligand_content_item["content"]
        ligand_path = os.path.join(ligands_dir, basename)
        with open(ligand_path, "w") as f:
            f.write(content)
        target_mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)[0]
        mcs = rdFMCS.FindMCS([ref_mol, target_mol])
        query = Chem.MolFromSmarts(mcs.smartsString)
        match_indexs = target_mol.GetSubstructMatch(query)
        target_mol.SetProp("mcsMatchInd", ",".join([str(i) for i in match_indexs]))
        sdf_io = StringIO()
        with Chem.SDWriter(sdf_io) as writer:
            writer.write(target_mol)
        new_content = sdf_io.getvalue()
        ligand_content_list[i]["content"] = new_content

        atom_info_map = AtomType().get_docking_atom_types(target_mol)

        bias_content = "x y z Vset r type atom\n"
        for atom in target_mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom_idx in match_indexs:
                atom_type = atom_info_map[atom_idx]
                atomic_num = atom.GetAtomicNum()
                atom_radius = RADIUS[str(atomic_num)]
                position = target_mol.GetConformer().GetAtomPosition(atom_idx)
                bias_content += f'{position.x:6.3f} {position.y:6.3f} {position.z:6.3f} {VSET:6.2f} {atom_radius:6.2f} map {atom_type:<2s}\n'
        bias_content_list.append({"name": os.path.splitext(basename)[0], "content": bias_content})

    with open("result_bias.json", "w") as f:
        json.dump(bias_content_list, f)
    
    with open("ligand_contents.json", "w") as f:
        json.dump(ligand_content_list, f)
    
    return OPIO({"bias_content_json": Path("result_bias.json"), "ligand_content_json":Path("ligand_contents.json")})