import os
import shutil
from pathlib import Path
import glob
import uuid
import subprocess
from argparse import Namespace
import unittest as ut


class TestUniDockTools(ut.TestCase):
    def setUp(self):
        self.workdir = Path(f"./tmp+{uuid.uuid4()}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.receptor = os.path.join(os.path.dirname(__file__), "1bcu_protein.pdb")
        self.ligand = os.path.join(os.path.dirname(__file__), "1bcu_ligand.sdf")
        self.pocket = [5.0, 15.0, 50.0, 15, 15, 15]

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_unidock_tools_func(self):
        from unidock_tools.unidock import UniDock

        args = Namespace(**{
            "center_x": self.pocket[0],
            "center_y": self.pocket[1],
            "center_z": self.pocket[2],
            "size_x": self.pocket[3],
            "size_y": self.pocket[4],
            "size_z": self.pocket[5],
            "search_mode": "fast",
            "num_modes": 1,
            "receptor": self.receptor,
            "dir": self.workdir.as_posix(),
        })
        unidock_runner = UniDock(self.receptor, "vina", self.workdir.as_posix())
        unidock_runner.set_gpu_batch([self.ligand])
        unidock_runner._get_config(args)
        unidock_runner.dock()

        result_ligand = glob.glob(os.path.join(self.workdir, "*_out.sdf"))[0]
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
        self.assertLess(score, 0)


    def test_unidock_tools_entrypoint(self):
        cmd = f"Unidock --receptor {self.receptor} --gpu_batch {self.ligand} --dir {self.workdir.as_posix()} \
            --center_x {self.pocket[0]} --center_y {self.pocket[1]} --center_z {self.pocket[2]} \
            --size_x {self.pocket[3]} --size_y {self.pocket[4]} --size_z {self.pocket[5]} \
            --search_mode fast --num_modes 1"
        resp = subprocess.run(cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            encoding="utf-8")
        print(resp.stdout)
        if resp.stderr:
            print(resp.stderr)

        result_ligand = glob.glob(os.path.join(self.workdir, "*_out.sdf"))[0]
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
        self.assertLess(score, 0)

if __name__ == "__main__":
    ut.main()