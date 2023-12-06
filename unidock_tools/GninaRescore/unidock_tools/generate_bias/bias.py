import os

from rdkit import Chem

from unidock_tools.ligand_prepare.atom_type import AtomType


class Bpf():
    def __init__(self, ref, output, Vset=-0.8, r=1.2, type="atom_type"):
        self.ref = ref

        self.basename, _ = os.path.splitext(output)

        self.Vset = Vset
        self.r = r
        self.type = type

        self.mol = Chem.SDMolSupplier(self.ref, removeHs=False)[0]
        self.positions = self.mol.GetConformer().GetPositions()

        # get AD4 atom types  
        at = AtomType()
        at.assign_atom_types(self.mol)
        self.types = [atom.GetProp('atom_type') for atom in self.mol.GetAtoms()]


    def genBpf(self):
        with open("%s.bpf"%self.basename, "w") as f1:
            f1.write("x y z Vset r type atom\n")
            for coordinate, symbol in zip(self.positions, self.types):
                if self.type == "atom_type":
                    f1.write("{} {} {} {} {} map {}\n".format(coordinate[0],coordinate[1], coordinate[2], self.Vset, self.r, symbol))
                elif self.type == "shape":
                    f1.write("{} {} {} {} {} map\n".format(coordinate[0],coordinate[1], coordinate[2], self.Vset, self.r))