import os
import re

import numpy as np
import networkx as nx
import glob

from concurrent.futures import ThreadPoolExecutor

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock_tools.ligandPrepare.atom_type import AtomType
from unidock_tools.ligandPrepare.rotatable_bond import RotatableBond
from unidock_tools.ligandPrepare import utils

class TopologyBuilder(object):
    def __init__(self,
                 ligand_sdf_file_name,
                 working_dir_name='.'):

        self.ligand_sdf_file_name = ligand_sdf_file_name

        self.working_dir_name = os.path.abspath(working_dir_name)

        ligand_sdf_base_file_name = os.path.basename(self.ligand_sdf_file_name)
        ligand_file_name_prefix = ligand_sdf_base_file_name.split('.')[0]

        ligand_pdbqt_base_file_name = ligand_file_name_prefix + '.pdbqt'
        self.ligand_pdbqt_file_name = os.path.join(self.working_dir_name, ligand_pdbqt_base_file_name)

        ligand_torsion_tree_sdf_base_file_name = ligand_file_name_prefix + '_prepared.sdf'
        self.ligand_torsion_tree_sdf_file_name = os.path.join(self.working_dir_name, ligand_torsion_tree_sdf_base_file_name)

        self.atom_typer = AtomType()
        self.rotatable_bond_finder = RotatableBond()

    def build_molecular_graph(self):
        mol = Chem.SDMolSupplier(self.ligand_sdf_file_name, removeHs=False)[0]


        self.atom_typer.assign_atom_types(mol)
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(mol)

    
        bond_list = list(mol.GetBonds())
        rotatable_bond_idx_list = []
        for bond in bond_list:
            bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            if bond_info in rotatable_bond_info_list or bond_info_reversed in rotatable_bond_info_list:
                rotatable_bond_idx_list.append(bond.GetIdx())

        splitted_mol = FragmentOnBonds(mol, rotatable_bond_idx_list, addDummies=False)
        splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))

        num_fragments = len(splitted_mol_list)

        ## Find fragment as the root node
        ##############################################################################
        num_fragment_atoms_list = [None] * num_fragments
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            num_atoms = fragment.GetNumAtoms()
            num_fragment_atoms_list[fragment_idx] = num_atoms

        root_fragment_idx = None
        root_fragment_idx = np.argmax(num_fragment_atoms_list)
        ##############################################################################

        ## Build torsion tree
        ### Add atom info into nodes
        ##############################################################################
        torsion_tree = nx.Graph()
        node_idx = 0
        root_fragment = splitted_mol_list[root_fragment_idx]
        num_root_atoms = root_fragment.GetNumAtoms()
        atom_info_list = [None] * num_root_atoms

        for root_atom_idx in range(num_root_atoms):
            root_atom = root_fragment.GetAtomWithIdx(root_atom_idx)
            atom_info_dict = {}
            atom_info_dict['sdf_atom_idx'] = root_atom.GetIntProp('sdf_atom_idx')
            atom_info_dict['atom_name'] = root_atom.GetProp('atom_name')
            atom_info_dict['residue_name'] = root_atom.GetProp('residue_name')
            atom_info_dict['chain_idx'] = root_atom.GetProp('chain_idx')
            atom_info_dict['residue_idx'] = root_atom.GetIntProp('residue_idx')
            atom_info_dict['x'] = root_atom.GetDoubleProp('x')
            atom_info_dict['y'] = root_atom.GetDoubleProp('y')
            atom_info_dict['z'] = root_atom.GetDoubleProp('z')
            atom_info_dict['charge'] = root_atom.GetDoubleProp('charge')
            atom_info_dict['atom_type'] = root_atom.GetProp('atom_type')

            atom_info_list[root_atom_idx] = atom_info_dict

        torsion_tree.add_node(node_idx, atom_info_list=atom_info_list)
        node_idx += 1

        for fragment_idx in range(num_fragments):
            if fragment_idx == root_fragment_idx:
                continue
            else:
                fragment = splitted_mol_list[fragment_idx]
                num_fragment_atoms = fragment.GetNumAtoms()
                atom_info_list = [None] * num_fragment_atoms

                for atom_idx in range(num_fragment_atoms):
                    atom = fragment.GetAtomWithIdx(atom_idx)
                    atom_info_dict = {}
                    atom_info_dict['sdf_atom_idx'] = atom.GetIntProp('sdf_atom_idx')
                    atom_info_dict['atom_name'] = atom.GetProp('atom_name')
                    atom_info_dict['residue_name'] = atom.GetProp('residue_name')
                    atom_info_dict['chain_idx'] = atom.GetProp('chain_idx')
                    atom_info_dict['residue_idx'] = atom.GetIntProp('residue_idx')
                    atom_info_dict['x'] = atom.GetDoubleProp('x')
                    atom_info_dict['y'] = atom.GetDoubleProp('y')
                    atom_info_dict['z'] = atom.GetDoubleProp('z')
                    atom_info_dict['charge'] = atom.GetDoubleProp('charge')
                    atom_info_dict['atom_type'] = atom.GetProp('atom_type')

                    atom_info_list[atom_idx] = atom_info_dict

                torsion_tree.add_node(node_idx, atom_info_list=atom_info_list)
                node_idx += 1

        ##############################################################################

        ### Add edge info
        ##############################################################################
        num_rotatable_bonds = len(rotatable_bond_info_list)
        for edge_idx in range(num_rotatable_bonds):
            rotatable_bond_info = rotatable_bond_info_list[edge_idx]
            begin_atom_idx = rotatable_bond_info[0]
            end_atom_idx = rotatable_bond_info[1]

            begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
            begin_sdf_atom_idx = begin_atom.GetIntProp('sdf_atom_idx')
            begin_atom_name = begin_atom.GetProp('atom_name')

            end_atom = mol.GetAtomWithIdx(end_atom_idx)
            end_sdf_atom_idx = end_atom.GetIntProp('sdf_atom_idx')
            end_atom_name = end_atom.GetProp('atom_name')

            begin_node_idx = None
            end_node_idx = None
            for node_idx in range(num_fragments):
                atom_info_list = torsion_tree.nodes[node_idx]['atom_info_list']
                for atom_info_dict in atom_info_list:
                    if atom_info_dict['atom_name'] == begin_atom_name:
                        begin_node_idx = node_idx
                        break
                    elif atom_info_dict['atom_name'] == end_atom_name:
                        end_node_idx = node_idx
                        break

                if begin_node_idx is not None and end_node_idx is not None:
                    break

            if begin_node_idx is None or end_node_idx is None:
                raise ValueError('Bugs in edge assignment code!!')

            torsion_tree.add_edge(begin_node_idx,
                                  end_node_idx,
                                  begin_node_idx=begin_node_idx,
                                  end_node_idx=end_node_idx,
                                  begin_sdf_atom_idx=begin_sdf_atom_idx,
                                  end_sdf_atom_idx=end_sdf_atom_idx,
                                  begin_atom_name=begin_atom_name,
                                  end_atom_name=end_atom_name)

        ##############################################################################

        self.torsion_tree = torsion_tree
        self.mol = mol

    def __deep_first_search__(self, node_idx):
        if node_idx == 0:
            self.pdbqt_atom_line_list.append('ROOT\n')
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                atom_name = atom_info_dict['atom_name']
                self.atom_idx_info_mapping_dict[atom_name] = self.pdbqt_atom_idx
                atom_info_tuple = ('ATOM',
                                   self.pdbqt_atom_idx,
                                   atom_info_dict['atom_name'],
                                   atom_info_dict['residue_name'],
                                   atom_info_dict['chain_idx'],
                                   atom_info_dict['residue_idx'],
                                   atom_info_dict['x'],
                                   atom_info_dict['y'],
                                   atom_info_dict['z'],
                                   1.0,
                                   0.0,
                                   atom_info_dict['charge'],
                                   atom_info_dict['atom_type'])

                self.pdbqt_atom_line_list.append(self.pdbqt_atom_line_format.format(*atom_info_tuple))
                self.pdbqt_atom_idx += 1

            self.pdbqt_atom_line_list.append('ENDROOT\n')

        else:
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                atom_name = atom_info_dict['atom_name']
                if atom_name not in self.atom_idx_info_mapping_dict:
                    self.atom_idx_info_mapping_dict[atom_name] = self.pdbqt_atom_idx

                atom_info_tuple = ('ATOM',
                                   self.pdbqt_atom_idx,
                                   atom_info_dict['atom_name'],
                                   atom_info_dict['residue_name'],
                                   atom_info_dict['chain_idx'],
                                   atom_info_dict['residue_idx'],
                                   atom_info_dict['x'],
                                   atom_info_dict['y'],
                                   atom_info_dict['z'],
                                   1.0,
                                   0.0,
                                   atom_info_dict['charge'],
                                   atom_info_dict['atom_type'])

                self.pdbqt_atom_line_list.append(self.pdbqt_atom_line_format.format(*atom_info_tuple))
                self.pdbqt_atom_idx += 1

        self.visited_node_idx_set.add(node_idx)

        neighbor_node_idx_list = list(self.torsion_tree.neighbors(node_idx))
        for neighbor_node_idx in neighbor_node_idx_list:
            if neighbor_node_idx not in self.visited_node_idx_set:
                temp_pdbqt_atom_idx = self.pdbqt_atom_idx
                atom_info_list = self.torsion_tree.nodes[neighbor_node_idx]['atom_info_list']
                for atom_info_dict in atom_info_list:
                    atom_name = atom_info_dict['atom_name']
                    if atom_name not in self.atom_idx_info_mapping_dict:
                        self.atom_idx_info_mapping_dict[atom_name] = temp_pdbqt_atom_idx
                        temp_pdbqt_atom_idx += 1

                edge_info = self.torsion_tree.edges[(node_idx, neighbor_node_idx)]
                begin_node_idx = edge_info['begin_node_idx']
                end_node_idx = edge_info['end_node_idx']
                begin_atom_name = edge_info['begin_atom_name']
                end_atom_name = edge_info['end_atom_name']

                if begin_node_idx == node_idx:
                    parent_atom_name = begin_atom_name
                    offspring_atom_name = end_atom_name
                else:
                    parent_atom_name = end_atom_name
                    offspring_atom_name = begin_atom_name

                parent_atom_idx = self.atom_idx_info_mapping_dict[parent_atom_name]
                offspring_atom_idx = self.atom_idx_info_mapping_dict[offspring_atom_name]

                self.branch_info_list.append((parent_atom_name, str(parent_atom_idx), offspring_atom_name, str(offspring_atom_idx)))
                self.pdbqt_atom_line_list.append(self.pdbqt_branch_line_format.format('BRANCH', parent_atom_idx, offspring_atom_idx))
                self.__deep_first_search__(neighbor_node_idx)
                self.pdbqt_atom_line_list.append(self.pdbqt_end_branch_line_format.format('ENDBRANCH', parent_atom_idx, offspring_atom_idx))

    def write_pdbqt_file(self):
        self.pdbqt_remark_line_list = []
        self.pdbqt_atom_line_list = []

        self.pdbqt_remark_torsion_line_format = '{:6s}   {:^2d}  {:1s}    {:7s} {:6s} {:^7s}  {:3s}  {:^7s}\n'
        self.pdbqt_atom_line_format = '{:4s}  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}\n'
        self.pdbqt_branch_line_format = '{:6s} {:3d} {:3d}\n'
        self.pdbqt_end_branch_line_format = '{:9s} {:3d} {:3d}\n'
        self.torsion_dof_line_format = '{:7s} {:d}'

        ## Prepare pdbqt atom lines
        ####################################################################################################
        self.atom_idx_info_mapping_dict = {}
        self.branch_info_list = []
        self.visited_node_idx_set = set()
        self.pdbqt_atom_idx = 1

        self.__deep_first_search__(0)
        self.num_torsions = len(self.branch_info_list)
        self.pdbqt_atom_line_list.append(self.torsion_dof_line_format.format('TORSDOF', self.num_torsions))
        ####################################################################################################

        ## Prepare pdbqt remark lines
        ####################################################################################################
        self.pdbqt_remark_line_list.append('REMARK  '  + str(self.num_torsions) + ' active torsions:\n')
        self.pdbqt_remark_line_list.append("REMARK  status: ('A' for Active; 'I' for Inactive)\n")
        for torsion_idx in range(self.num_torsions):
            branch_info_tuple = self.branch_info_list[torsion_idx]
            remark_torsion_info_tuple = ('REMARK',
                                         torsion_idx+1,
                                         'A',
                                         'between',
                                         'atoms:',
                                         branch_info_tuple[0] + '_' + branch_info_tuple[1],
                                         'and',
                                         branch_info_tuple[2] + '_' + branch_info_tuple[3])

            self.pdbqt_remark_line_list.append(self.pdbqt_remark_torsion_line_format.format(*remark_torsion_info_tuple))
        ####################################################################################################

        self.pdbqt_line_list = self.pdbqt_remark_line_list + self.pdbqt_atom_line_list

        with open(self.ligand_pdbqt_file_name, 'w') as ligand_pdbqt_file:
            for pdbqt_line in self.pdbqt_line_list:
                ligand_pdbqt_file.write(pdbqt_line)

    def write_constraint_bpf_file(self):
        self.core_bpf_remark_line_list = []
        self.core_bpf_atom_line_list = []
        self.core_bpf_atom_line_format = '{:8.3f}\t{:8.3f}\t{:8.3f}\t{:6.2f}\t{:6.2f}\t{:3s}\t{:<2s}\n'

        self.core_bpf_remark_line_list.append('x y z Vset r type atom\n')

        root_atom_info_list = self.torsion_tree.nodes[0]['atom_info_list']
        for atom_info_dict in root_atom_info_list:
            atom_info_tuple = (atom_info_dict['x'],
                               atom_info_dict['y'],
                               atom_info_dict['z'],
                               -1.2,
                               0.6,
                               'map',
                               atom_info_dict['atom_type'])

            self.core_bpf_atom_line_list.append(self.core_bpf_atom_line_format.format(*atom_info_tuple))

        self.core_bpf_line_list = self.core_bpf_remark_line_list + self.core_bpf_atom_line_list

        with open(self.ligand_core_bpf_file_name, 'w') as ligand_core_bpf_file:
            for core_bpf_line in self.core_bpf_line_list:
                ligand_core_bpf_file.write(core_bpf_line)

    def write_torsion_tree_sdf_file(self):
        fragment_info_string = ''
        torsion_info_string = ''
        atom_info_string = ''

        num_nodes = self.torsion_tree.number_of_nodes()
        num_edges = self.torsion_tree.number_of_edges()

        for node_idx in range(num_nodes):
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                fragment_info_string += str(atom_info_dict['sdf_atom_idx'])
                fragment_info_string += ' '
    
            fragment_info_string = fragment_info_string[:-1]
            fragment_info_string += '\n'

        edge_key_list = list(self.torsion_tree.edges.keys())
        for edge_idx in range(num_edges):
            edge_key = edge_key_list[edge_idx]
            edge_info_dict = self.torsion_tree.edges[edge_key]
            begin_sdf_atom_idx = str(edge_info_dict['begin_sdf_atom_idx'])
            end_sdf_atom_idx = str(edge_info_dict['end_sdf_atom_idx'])
            begin_node_idx = str(edge_info_dict['begin_node_idx'])
            end_node_idx = str(edge_info_dict['end_node_idx'])

            torsion_info_string += f'{begin_sdf_atom_idx} {end_sdf_atom_idx} {begin_node_idx} {end_node_idx}'
            torsion_info_string += '\n'

        for atom in self.mol.GetAtoms():
            sdf_atom_idx = str(atom.GetIntProp('sdf_atom_idx'))
            charge = str(atom.GetDoubleProp('charge'))
            atom_type = atom.GetProp('atom_type')
            atom_info = str(sdf_atom_idx).ljust(3) + str(charge)[:10].ljust(10) + atom_type.ljust(2)
            atom_info_string += atom_info
            atom_info_string += '\n'

        self.mol.SetProp('fragInfo', fragment_info_string)
        self.mol.SetProp('torsionInfo', torsion_info_string)
        self.mol.SetProp('atomInfo', atom_info_string)

        writer = Chem.SDWriter(self.ligand_torsion_tree_sdf_file_name)
        writer.write(self.mol)
        writer.flush()
        writer.close()

def prepare_ligands(SDFFiles, output_dir='./ligands_prepared'):
    os.makedirs(output_dir, exist_ok=True)

    def _convert_file(ligand):
        basename =  os.path.basename(ligand)
        basename_prefix = basename.split('.')[0]
        try:
            topo=TopologyBuilder(ligand, output_dir)
            topo.build_molecular_graph()
            topo.write_torsion_tree_sdf_file()
            print("ligand %s preperation successful"%basename_prefix)
        except Exception as e:
            print("%s, ligand %s preperation failed"%(e, basename_prefix))


    with ThreadPoolExecutor() as executor:
        executor.map(_convert_file, SDFFiles)
    
    basenames =  [os.path.basename(ligand) for ligand in SDFFiles]
    basename_prefixs = [basename.split('.')[0] for basename in basenames]
    
    ligands_prepared = []
    for basename_prefix in basename_prefixs:
        ligand_prepared_filename = "%s/%s_prepared.sdf"%(output_dir, basename_prefix)
        if os.path.exists(ligand_prepared_filename):
            ligands_prepared.append(ligand_prepared_filename)

    ligands_num = len(SDFFiles)
    ligands_prepared_num = len(ligands_prepared)
    print("%d sdf format ligands have been prepared successfully in total %d"%(ligands_prepared_num, ligands_num))

    return ligands_prepared
