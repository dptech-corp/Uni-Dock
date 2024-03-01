from pathlib import Path
import os
import pytest


@pytest.fixture
def input_ligand():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "inputs", "Imatinib.sdf")


def test_ligand_topology(input_ligand):
    from rdkit import Chem
    from unidock_tools.modules.ligand_prep import TopologyBuilder

    mol = Chem.SDMolSupplier(input_ligand, removeHs=False)[0]
    tb = TopologyBuilder(mol)
    tb.build_molecular_graph()

    output_ligand = "Imatinib_prepared.sdf"
    tb.write_sdf_file(output_ligand)
    out_mol = Chem.SDMolSupplier(output_ligand, removeHs=False)[0]
    assert out_mol.HasProp("fragInfo") and out_mol.HasProp("torsionInfo") and out_mol.HasProp("atomInfo"), "Failed to build molecular graph by TopologyBuilder"

    Path(output_ligand).unlink(missing_ok=True)