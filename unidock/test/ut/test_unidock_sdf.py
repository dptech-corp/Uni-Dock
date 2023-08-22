import os
import shutil
from pathlib import Path
import uuid
import subprocess
import unittest as ut


class TestUniDockSDF(ut.TestCase):
    def setUp(self):
        self.workdir = Path(f"/tmp/{uuid.uuid4()}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.receptor = os.path.join(os.path.dirname(__file__), "..", "receptor", "1a30_protein.pdbqt")
        self.ligand = os.path.join(os.path.dirname(__file__), "..", "ligands", "1a30_ligand.sdf")
        self.pocket = [8.729, 25.62, 4.682, 19.12, 16.56, 18.65]

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_unidock_sdf(self):
        cmd = f"unidock --receptor {self.receptor} --gpu_batch {self.ligand} --dir {self.workdir} \
            --center_x {self.pocket[0]} --center_y {self.pocket[1]} --center_z {self.pocket[2]} \
            --size_x {self.pocket[3]} --size_y {self.pocket[4]} --size_z {self.pocket[5]} \
            --scoring vina --search_mode fast --num_modes 1 --seed 181129"
        resp = subprocess.run(cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", cwd=self.workdir)
        print(resp.stdout)
        if resp.stderr:
            print(resp.stderr)
        result_ligand = os.path.join(self.workdir, "1a30_ligand_out.sdf")
        self.assertTrue(os.path.exists(result_ligand))
        score_line = ""
        with open(result_ligand, "r") as f:
            while True:
                line = f.readline()
                if line.startswith("> <Uni-Dock RESULT>"):
                    score_line = f.readline().strip("\n")
                    break
                if not line:
                    break
        self.assertNotEqual(score_line, "")
        score = float([e for e in score_line[len("ENERGY="):].split(" ") if e!=""][0])
        self.assertEqual(score, -6.432)


if __name__ == "__main__":
    ut.main()