import os
import shutil
from pathlib import Path
import uuid
import subprocess
import unittest as ut


class TestUniDock(ut.TestCase):
    def setUp(self):
        self.workdir = Path(f"/tmp/{uuid.uuid4()}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.receptor = os.path.join(os.path.dirname(os.path.dirname(__file__)), "receptor", "1iep_receptor.pdbqt")
        self.ligand = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ligands", "1iep_ligand.pdbqt")
        self.pocket = [15.19, 53.903, 16.917, 20, 20, 20]

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_unidock_vina(self):
        cmd = f"unidock --receptor {self.receptor} --gpu_batch {self.ligand} --dir {self.workdir} \
            --center_x {self.pocket[0]} --center_y {self.pocket[1]} --center_z {self.pocket[2]} \
            --size_x {self.pocket[3]} --size_y {self.pocket[4]} --size_z {self.pocket[5]} \
            --scoring vina --search_mode fast --num_modes 1 --seed 181129"
        resp = subprocess.run(cmd, shell=True, 
                              capture_output=True,
                              encoding="utf-8", 
                              cwd=self.workdir)
        print(resp.stdout)
        if resp.returncode != 0:
            print(resp.stderr)
        result_ligand = os.path.join(self.workdir, "1iep_ligand_out.pdbqt")
        self.assertTrue(os.path.exists(result_ligand))
        score_line = ""
        with open(result_ligand, "r") as f:
            for line in f.readlines():
                if line.startswith("REMARK VINA RESULT:"):
                    score_line = line.strip("\n")
                    break
        self.assertNotEqual(score_line, "")
        score = float([e for e in score_line[len("REMARK VINA RESULT:"):].split(" ") if e!=""][0])
        self.assertTrue(-12 <= score <= -5)

    def test_unidock_vinardo(self):
        cmd = f"unidock --receptor {self.receptor} --gpu_batch {self.ligand} --dir {self.workdir} \
            --center_x {self.pocket[0]} --center_y {self.pocket[1]} --center_z {self.pocket[2]} \
            --size_x {self.pocket[3]} --size_y {self.pocket[4]} --size_z {self.pocket[5]} \
            --scoring vinardo --search_mode fast --num_modes 1 --seed 181129"
        resp = subprocess.run(cmd, shell=True, 
                              capture_output=True,
                              encoding="utf-8", 
                              cwd=self.workdir)
        print(resp.stdout)
        if resp.returncode != 0:
            print(resp.stderr)
        result_ligand = os.path.join(self.workdir, "1iep_ligand_out.pdbqt")
        self.assertTrue(os.path.exists(result_ligand))
        score_line = ""
        with open(result_ligand, "r") as f:
            for line in f.readlines():
                if line.startswith("REMARK VINA RESULT:"):
                    score_line = line.strip("\n")
                    break
        self.assertNotEqual(score_line, "")
        score = float([e for e in score_line[len("REMARK VINA RESULT:"):].split(" ") if e!=""][0])
        self.assertTrue(-12 <= score <= -5)

if __name__ == "__main__":
    ut.main()