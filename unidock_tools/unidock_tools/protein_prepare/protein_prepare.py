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
    
    def pdb2pdbqt(self, protein_path:str, output_path:str):
        conda_env_path = shutil.which("conda")
        if not conda_env_path:
            raise KeyError("To prepare a PDB format protein, you need to install MGLTools first.\
                           (First install conda, then run 'conda create -n mgltools mgltools -c bioconda')")
        mgl_env_path = os.path.join(os.path.dirname(os.path.dirname(conda_env_path)), "envs", "mgltools")
        pythonsh_path = os.path.join(mgl_env_path, "bin", "pythonsh")
        prepare_script_path = os.path.join(mgl_env_path, "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_receptor4.py")
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