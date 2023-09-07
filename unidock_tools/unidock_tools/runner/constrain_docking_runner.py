import os
import sys
import json

os.environ['PATH'] = '/data/Modules/Uni-Dock:/data/Modules/watvina:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/data/Modules/ADFR/lib'
os.environ['MGLPY'] = '/data/Modules/ADFR/bin/python'
os.environ['MGLUTIL'] = '/data/Modules/ADFR/CCSBpckgs/AutoDockTools/Utilities24'
os.environ['WATVINA'] = '/data/Modules/watvina/pywatvina/watvina'

from xdatools.modules.docking_engines.autodock_runner import AutoDockRunner

config_file_name = sys.argv[1]
with open(config_file_name, 'r') as config_file:
    config = json.load(config_file)

ligand_sdf_file_name_list = config['ligand_sdf_file_name_list']
protein_pdb_file_name = config['protein_pdb_file_name']
target_center = (config['target_center_x'], config['target_center_y'], config['target_center_z'])
box_size = (config['box_size_x'], config['box_size_y'], config['box_size_z'])
reference_sdf_file_name = config['reference_sdf_file_name']
n_cpu = config['n_cpu']
output_dir_name = os.path.abspath(config['output_dir_name'])

working_dir_name = os.path.join(os.getcwd(), 'autodock')
os.mkdir(working_dir_name)

autodock_runner = AutoDockRunner(ligand_sdf_file_name_list,
                                 [protein_pdb_file_name],
                                 protein_conf_name_list=['protein_conf_0'],
                                 target_center_list=[target_center],
                                 box_size=box_size,
                                 covalent_residue_atom_info_nested_list=None,
                                 docking_engine='watvina',
                                 docking_method='template_docking',
                                 reference_sdf_file_name=reference_sdf_file_name,
                                 generate_torsion_tree_sdf=False,
                                 n_cpu=n_cpu,
                                 num_docking_runs=1,
                                 working_dir_name=working_dir_name)

docking_pose_summary_info_list = autodock_runner.run()
docking_pose_summary_info = docking_pose_summary_info_list[0]
docking_pose_summary_info.to_csv(os.path.join(working_dir_name, f'docking_pose_summary.csv'))

ligand_docked_sdf_file_name = docking_pose_summary_info.loc[:, 'ligand_docked_sdf_file_name'].values.tolist()[0]
ligand_conformer_energy = docking_pose_summary_info.loc[:, 'conformer_energy'].values.tolist()[0]
ligand_docking_score = docking_pose_summary_info.loc[:, 'binding_free_energy'].values.tolist()[0]
ligand_smiles = docking_pose_summary_info.loc[:, 'ligand_smiles_string'].values.tolist()[0]

output_sdf_file_name = os.path.join(output_dir_name, 'ligand_docking_pose.sdf')
output_json_file_name = os.path.join(output_dir_name, 'ligand_docking_info.json')

os.system(f'cp {ligand_docked_sdf_file_name} {output_sdf_file_name}')

ligand_docking_info_dict = {'conformer_energy': ligand_conformer_energy,
                            'docking_score': ligand_docking_score,
                            'smiles': ligand_smiles}

ligand_docking_info_json = json.dumps(ligand_docking_info_dict)
with open(output_json_file_name, 'w') as ligand_docking_info_json_file:
    ligand_docking_info_json_file.write(ligand_docking_info_json)
