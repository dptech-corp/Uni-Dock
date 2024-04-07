import os
import shutil
import tempfile
import subprocess
import pytest
#from unidocktools.new_proteinprep import receptor_preprocessor

@pytest.fixture
def pdb_file():
    temp_dir = tempfile.mkdtemp()
    pdb_file = os.path.join(temp_dir, "protein.pdb")
    yield pdb_file
    shutil.rmtree(temp_dir)

def test_receptor_processor_app(pdb_file):
    protein_pdbqt_file_name = "protein.pdbqt"
    cmd = f"unidocktools proteinprep -r {pdb_file} -o {protein_pdbqt_file_name}"
    subprocess.run(cmd, shell=True)
    assert os.path.isfile(protein_pdbqt_file_name), "PDBQT file was not generated"
    os.remove(protein_pdbqt_file_name)