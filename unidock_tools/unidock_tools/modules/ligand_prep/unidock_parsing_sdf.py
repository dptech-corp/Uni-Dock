import os

import multiprocess as mp
from multiprocess.pool import Pool

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import ChemicalForceFields

def unidock_parsing_process(reference_sdf_file_name,
                            raw_docked_sdf_file_name,
                            working_dir_name,
                            docking_pose_summary_info_df_proxy_list,
                            docked_file_idx):

    reference_mol = Chem.SDMolSupplier(reference_sdf_file_name, removeHs=False)[0]
    raw_docked_mol_list = list(Chem.SDMolSupplier(raw_docked_sdf_file_name, removeHs=False))
    num_docked_conformations = len(raw_docked_mol_list)

    docked_sdf_file_name_prefix = os.path.basename(raw_docked_sdf_file_name).split('.')[0].replace('_out', '')
    docked_sdf_file_name_list = [None] * num_docked_conformations
    docked_conf_energy_list = [None] * num_docked_conformations
    docked_conf_score_list = [None] * num_docked_conformations

    for pose_idx in range(num_docked_conformations):
        raw_docked_mol = raw_docked_mol_list[pose_idx]
        docked_conf_score_info_string = raw_docked_mol.GetProp('Uni-Dock RESULT')
        docked_conf_score = np.float32(docked_conf_score_info_string.split()[1])

        ################################################################################################################################################
        # FIX ME: temperal fix for unidock sdf writer bug
        raw_docked_mol.ClearProp('Uni-Dock RESULT')
        ################################################################################################################################################

        ################################################################################################################################################
        #FIX ME: this is for dealing with that current unidock cannot sample non-polar hydrogen
        raw_docked_mol_no_h = Chem.RemoveHs(raw_docked_mol)
        docked_mol = Chem.AddHs(raw_docked_mol_no_h, addCoords=True)
        ################################################################################################################################################

        ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(docked_mol, 'MMFF94s')
        ff = ChemicalForceFields.MMFFGetMoleculeForceField(docked_mol, ff_property)

        if ff is not None:
            ff.Initialize()
            docked_conf_energy = ff.CalcEnergy()
            docked_mol.SetProp('conformer_energy', str(docked_conf_energy))
        else:
            docked_conf_energy = 1000.0

        docked_sdf_file_name = os.path.join(working_dir_name, docked_sdf_file_name_prefix + '_pose_' + str(pose_idx) + '.sdf')
        sdf_writer = Chem.SDWriter(docked_sdf_file_name)
        sdf_writer.write(docked_mol)
        sdf_writer.flush()
        sdf_writer.close()

        docked_sdf_file_name_list[pose_idx] = docked_sdf_file_name
        docked_conf_energy_list[pose_idx] = docked_conf_energy
        docked_conf_score_list[pose_idx] = docked_conf_score

    if reference_mol.HasProp('smiles_string'):
        ligand_smiles_string = reference_mol.GetProp('smiles_string')
    else:
        reference_mol_no_h = Chem.RemoveHs(reference_mol)
        ligand_smiles_string = Chem.MolToSmiles(reference_mol_no_h)

    ligand_smiles_string_array = np.array([ligand_smiles_string] * num_docked_conformations, dtype='U')
    original_sdf_file_name_array = np.array([reference_sdf_file_name] * num_docked_conformations, dtype='U')
    docked_conf_score_array = np.array(docked_conf_score_list, dtype=np.float32)

    num_reference_mol_heavy_atoms = reference_mol.GetNumHeavyAtoms()
    docked_conf_ligand_efficiency_array = docked_conf_score_array / num_reference_mol_heavy_atoms

    docking_pose_summary_info_dict = {}
    docking_pose_summary_info_dict['ligand_smiles_string'] = ligand_smiles_string_array
    docking_pose_summary_info_dict['ligand_original_sdf_file_name'] = original_sdf_file_name_array
    docking_pose_summary_info_dict['ligand_docked_sdf_file_name'] = np.array(docked_sdf_file_name_list, dtype='U')
    docking_pose_summary_info_dict['conformer_energy'] = np.array(docked_conf_energy_list, dtype=np.float32)
    docking_pose_summary_info_dict['binding_free_energy'] = docked_conf_score_array
    docking_pose_summary_info_dict['ligand_efficiency'] = docked_conf_ligand_efficiency_array

    docking_pose_summary_info_df = pd.DataFrame(docking_pose_summary_info_dict)
    docking_pose_summary_info_df_proxy_list[docked_file_idx] = docking_pose_summary_info_df

    return True

class UniDockParsingSDF(object):
    def __init__(self,
                 ligand_original_sdf_file_name_list,
                 ligand_docked_sdf_file_name_list,
                 n_cpu=16,
                 working_dir_name='.'):

        self.ligand_original_sdf_file_name_list = ligand_original_sdf_file_name_list
        self.ligand_docked_sdf_file_name_list = ligand_docked_sdf_file_name_list
        self.n_cpu = n_cpu
        self.num_docked_files = len(self.ligand_docked_sdf_file_name_list)
        self.working_dir_name = os.path.abspath(working_dir_name)

    def run_unidock_parsing(self):
        manager = mp.Manager()
        docking_pose_summary_info_df_proxy_list = manager.list()
        docking_pose_summary_info_df_proxy_list.extend([None] * self.num_docked_files)
        autodock_parsing_results_list = [None] * self.num_docked_files
        autodock_parsing_pool = Pool(processes=self.n_cpu)

        for docked_file_idx in range(self.num_docked_files):
            reference_sdf_file_name = self.ligand_original_sdf_file_name_list[docked_file_idx]
            sdf_file_name = self.ligand_docked_sdf_file_name_list[docked_file_idx]
            autodock_parsing_results = autodock_parsing_pool.apply_async(unidock_parsing_process,
                                                                         args=(reference_sdf_file_name,
                                                                               sdf_file_name,
                                                                               self.working_dir_name,
                                                                               docking_pose_summary_info_df_proxy_list,
                                                                               docked_file_idx))

            autodock_parsing_results_list[docked_file_idx] = autodock_parsing_results

        autodock_parsing_pool.close()
        autodock_parsing_pool.join()

        autodock_parsing_results_list = [autodock_parsing_results.get() for autodock_parsing_results in autodock_parsing_results_list]
        self.docking_pose_summary_info_df_list = list(docking_pose_summary_info_df_proxy_list)

        docking_pose_summary_info_merged_df = pd.concat(self.docking_pose_summary_info_df_list)
        docking_pose_summary_info_merged_df.reset_index(drop=True, inplace=True)
        self.docking_pose_summary_info_df = docking_pose_summary_info_merged_df

        docking_pose_summary_info_csv_file_name = os.path.join(self.working_dir_name, 'docking_pose_summary.csv')
        self.docking_pose_summary_info_df.to_csv(docking_pose_summary_info_csv_file_name, index=False)
