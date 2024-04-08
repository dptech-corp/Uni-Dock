from pathlib import Path
import os
import pytest
import tempfile



@pytest.fixture
def pdb_file():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "inputs", "protein.pdb")

def test_receptor_preprocessor(pdb_file):
    from unidock_tools.modules.protein_prep import receptor_preprocessor
    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        protein_pdbqt_file_name = receptor_preprocessor(pdb_file, prepared_hydrogen=True, working_dir_name=temp_dir)
        # Assert that the generated PDBQT file exists
        assert os.path.exists(protein_pdbqt_file_name)
            
if __name__ == "__main__":
    pytest.main([__file__])