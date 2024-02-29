import os
import pytest


@pytest.fixture
def input_ligand():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "inputs", "Imatinib.sdf")


def test_confgen_cdpkit(input_ligand):
    from rdkit import Chem
    from unidock_tools.modules.confgen.cdpkit import CDPKitConfGenerator

    mol = Chem.SDMolSupplier(input_ligand, removeHs=False)[0]

    runner = CDPKitConfGenerator()
    runner.check_env()
    mol_confs = runner.generate_conformation(mol, max_num_confs_per_ligand=100)
    assert len(mol_confs) == 100, "Failed generate enough conformation by cdpkit"


def test_confgen_obabel(input_ligand):
    from rdkit import Chem
    from unidock_tools.modules.confgen.obabel import OBabelConfGenerator

    mol = Chem.SDMolSupplier(input_ligand, removeHs=False)[0]

    runner = OBabelConfGenerator()
    runner.check_env()
    mol_confs = runner.generate_conformation(mol, max_num_confs_per_ligand=10)
    assert len(mol_confs) == 10, "Failed generate enough conformation by obabel"