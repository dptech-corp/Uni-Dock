from rdkit import Chem

class RotatableBond(object):
    def __init__(self,
                 min_macrocycle_size=7,
                 max_macrocycle_size=33,
                 double_bond_penalty=50,
                 max_breaks=4):

        self.rotatable_bond_smarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        self.conjugate_bond_smarts = '*=*[*]=,#,:[*]'
        self.rotatable_bond_pattern = Chem.MolFromSmarts(self.rotatable_bond_smarts)
        self.conjugate_bond_pattern = Chem.MolFromSmarts(self.conjugate_bond_smarts)

        self.min_macrocycle_size = min_macrocycle_size
        self.max_macrocycle_size = max_macrocycle_size
        self.double_bond_penalty = double_bond_penalty
        self.max_breaks = max_breaks

    def identify_rotatable_bonds(self, mol):
        default_rotatable_bond_info_list = list(mol.GetSubstructMatches(self.rotatable_bond_pattern))
        return default_rotatable_bond_info_list
