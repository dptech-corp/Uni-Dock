import os
import tempfile
import subprocess
import pytest

@pytest.fixture
def pdb_file():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "protein.pdb")

def test_receptor_processor_app(pdb_file):
    protein_pdbqt_file_name = "protein.pdbqt"
    with tempfile.TemporaryDirectory() as work_dir:
        cmd = f"unidocktools proteinprep -r {pdb_file} -o {protein_pdbqt_file_name} -wd {work_dir}"
        subprocess.run(cmd, shell=True, cwd=work_dir)
        assert os.path.isfile(os.path.join(work_dir, protein_pdbqt_file_name)), "PDBQT file was not generated"


if __name__ == "__main__":
    pytest.main([__file__])