import os
import shutil
from pathlib import Path
import subprocess


class DockingProteinPrepare:
    def __init__(self, input_protein_path:str, output_protein_path:str=''):
        self.input_protein_path = input_protein_path
        if not output_protein_path:
            output_protein_path = os.path.splitext(input_protein_path)[0] + '.pdbqt'
        self.output_protein_path = output_protein_path
    
    def pdb2pdbqt(self, protein_path:str, output_path:str, active_name_env: str = "mgltools") -> None:
        """ Locates mgltools and converts the pdb receptor file to pdbqt format"""
        conda_bin_env = shutil.which("conda")
        if not conda_bin_env:
            raise KeyError("Conda env not found, please install conda first.")
        conda_env = str(os.path.dirname(os.path.dirname(conda_bin_env)))
        mgl_env_path = Path(conda_env) / "envs" / active_name_env
        env_not_found = False
        if not mgl_env_path.exists():
            env_not_found = True
            # in case not the default name, find it
            result = subprocess.run(["conda", "info", "--envs"], capture_output=True, text=True)
            for line in result.stdout.strip().split("\n"):
                if "*" in set(line.split("")):
                    mgl_env_path = Path(line.split()[-1])
                    if mgl_env_path.exists():
                        env_not_found = False
                        break

        if env_not_found:
            raise KeyError("To prepare a PDB format protein, you need to install MGLTools first.\
                            (First install conda, then run 'conda create -n mgltools mgltools -c bioconda')")

        pythonsh_path = mgl_env_path / "bin" / "pythonsh"
        prepare_script_path = mgl_env_path / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / "prepare_receptor4.py"
        # catch problems early
        assert prepare_script_path.exists(), f"prepare_receptor4.py not found in {prepare_script_path}"
        cmd = f"{pythonsh_path} {prepare_script_path} -r {protein_path} -o {output_path} -U nphs_lps_nonstdres"
        resp = subprocess.run(cmd, shell=True, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                encoding="utf-8")
        print(resp.stdout)
        if resp.stderr:
            print(resp.stderr)

    def run(self) -> str:
        # tmp_pdb_path = os.path.join(os.path.splitext(self.input_protein_path)[0] + "_clean.pdb")
        # pdb_cleaner = ReceptorPDBReader(self.input_protein_path, output_protein_path=tmp_pdb_path)
        # pdb_cleaner.run_receptor_system_cleaning()

        self.pdb2pdbqt(self.input_protein_path, self.output_protein_path)

        # Path(tmp_pdb_path).unlink(missing_ok=True)

        return self.output_protein_path