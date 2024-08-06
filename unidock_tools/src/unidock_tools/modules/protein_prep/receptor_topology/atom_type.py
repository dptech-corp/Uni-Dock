from rdkit import Chem

ATOM_TYPE_DEFINITION_LIST = [{'smarts': '[#1]',                   'atype': 'H',  'comment': 'invisible'},
                             {'smarts': '[#1][#7,#8,#9,#15,#16]', 'atype': 'HD', 'comment': None},
                             {'smarts': '[#5]',                   'atype': 'B',  'comment': None},
                             {'smarts': '[C]',                    'atype': 'C',  'comment': None},
                             {'smarts': '[c]',                    'atype': 'A',  'comment': None},
                             {'smarts': '[#7]',                   'atype': 'NA', 'comment': None},
                             {'smarts': '[#8]',                   'atype': 'OA', 'comment': None},
                             {'smarts': '[#9]',                   'atype': 'F',  'comment': None},
                             {'smarts': '[#12]',                  'atype': 'Mg', 'comment': None},
                             {'smarts': '[#14]',                  'atype': 'Si', 'comment': None},
                             {'smarts': '[#15]',                  'atype': 'P',  'comment': None},
                             {'smarts': '[#16]',                  'atype': 'S',  'comment': None},
                             {'smarts': '[#17]',                  'atype': 'Cl', 'comment': None},
                             {'smarts': '[#20]',                  'atype': 'Ca', 'comment': None},
                             {'smarts': '[#25]',                  'atype': 'Mn', 'comment': None},
                             {'smarts': '[#26]',                  'atype': 'Fe', 'comment': None},
                             {'smarts': '[#30]',                  'atype': 'Zn', 'comment': None},
                             {'smarts': '[#35]',                  'atype': 'Br', 'comment': None},
                             {'smarts': '[#53]',                  'atype': 'I',  'comment': None},
                             {'smarts': '[#7X3v3][a]',            'atype': 'N',  'comment': 'pyrrole, aniline'},
                             {'smarts': '[#7X3v3][#6X3v4]',       "atype": 'N',  'comment': 'amide'},
                             {'smarts': '[#7+1]',                 'atype': 'N',  'comment': 'ammonium, pyridinium'},
                             {'smarts': '[SX2]',                  'atype': 'SA', 'comment': 'sulfur acceptor'}]

class AtomType(object):
    def __init__(self):
        self.atom_type_definition_list = ATOM_TYPE_DEFINITION_LIST

    def assign_atom_types(self, mol):
        for atom_type_dict in self.atom_type_definition_list:
            smarts = atom_type_dict['smarts']
            atom_type = atom_type_dict['atype']

            pattern_mol = Chem.MolFromSmarts(smarts)
            pattern_matches = mol.GetSubstructMatches(pattern_mol)
            for pattern_match in pattern_matches:
                atom = mol.GetAtomWithIdx(pattern_match[0])
                atom.SetProp('atom_type', atom_type)
