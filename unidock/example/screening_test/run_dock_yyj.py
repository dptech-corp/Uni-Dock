import os
import time
import json
import glob
# import pandas


for _,dns,_ in os.walk("./data"):
    break

with open("config.json", "r") as f:
    config = json.load(f)

cmdlist = []

liglist = []
os.system("cd indata;tar -xvf {}.tar.bz2".format(config["target"]))
liglist = glob.glob("./indata/{}_unique/*.pdbqt".format(config["target"]))
if liglist == []:
    liglist = glob.glob("./indata/ligands_unique/*.pdbqt")

if liglist == []:
    liglist = glob.glob("./indata/{}_unique_charged/*.pdbqt".format(config["target"]))

print(liglist[:100])
with open("{}_ligands.txt".format(config["target"]), "w") as f:
    f.write(" ".join(liglist))
cmd = "unidock --receptor {} ".format("./indata/{}.pdbqt".format(config["target"]))
cmd += "--ligand_index {} ".format("{}_ligands.txt".format(config["target"]))
cmd += "--center_x {:.2f} --center_y {:.2f} --center_z {:.2f} ".format(
    config["center_x"], config["center_y"], config["center_z"])
cmd += "--size_x 22 --size_y 22 --size_z 22 "
cmd += "--dir ./result/{} ".format(config["target"])
cmd += "--exhaustiveness {} ".format(config["nt"])
cmd += "--max_step {} ".format(config["ns"])
cmd += "--num_modes 9  --scoring {} --refine_step {} ".format(config["sf"], config.get("rs", 5))
cmd += "--seed {}".format(config.get("seed", 42))
os.makedirs("result/{}".format(config["target"]), exist_ok=True)
with open("rundock.sh", "w") as f:
    f.write(cmd)

os.system("echo 'costtime'>> result/costtime.csv")
st = time.time()
os.system("bash rundock.sh")
os.system("echo '{}'>> result/costtime.csv".format(time.time()-st))

for _,_,fns in os.walk("./result/{}".format(config["target"])):
    break
csv_name = "./result/{}_{}_nt{}_ns{}_seed{}_{}.csv".format(config["target"], config["sf"], config["nt"], config["ns"], config["seed"], config.get("gpu_type", 'c12_m92_1 * NVIDIA V100').split()[-1])
with open(csv_name, "w") as f:
    f.write("idx,type,score\n")
for fn in fns:
    with open("result/{}/{}".format(config["target"], fn), "r") as f:
        f.readline()
        score = float(f.readline().split()[3])
    if fn.startswith("actives"):
        idx = int(fn.split("_")[0].replace("actives", ""))
        os.system("echo '{},{},{}'>> {}".format(idx, "active", score, csv_name))
    elif fn.startswith("decoys"):
        idx = int(fn.split("_")[0].replace("decoys", ""))
        os.system("echo '{},{},{}'>> {}".format(idx, "decoy", score, csv_name))



