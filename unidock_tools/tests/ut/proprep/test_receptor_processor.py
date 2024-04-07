from pathlib import Path
import os
import pytest

from unidock_tools.src.unidock_tools.modules.protein_prepreceptor_preprocessor_runner import receptor_preprocessor


@pytest.fixture
def pdb_file():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "inputs", "protein.pdb")

def test_receptor_preprocessor(pdb_file):
    working_dir = sys.argv[1]
    protein_pdbqt_file_name = receptor_preprocessor(pdb_file, prepared_hydrogen=True, working_dir_name=working_dir)
    # Assert that the generated PDBQT file exists
    assert os.path.exists(protein_pdbqt_file_name)
    # Clean up the generated PDBQT file
    #os.remove(protein_pdbqt_file_name)

if __name__ == "__main__":
    pytest.main([__file__])