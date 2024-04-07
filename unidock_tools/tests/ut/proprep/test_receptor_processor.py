from pathlib import Path
import os
import pytest


@pytest.fixture
def pdb_file():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "inputs", "protein.pdb")

def test_receptor_preprocessor(pdb_file):
    from unidock_tools.modules.protein_prep import receptor_preprocessor
    working_dir = '/personal/test_dir'
    protein_pdbqt_file_name = receptor_preprocessor(pdb_file, prepared_hydrogen=True, working_dir_name=working_dir)
    # Assert that the generated PDBQT file exists
    assert os.path.exists(protein_pdbqt_file_name)
    # Clean up the generated PDBQT file
    #os.remove(protein_pdbqt_file_name)

if __name__ == "__main__":
    pytest.main([__file__])