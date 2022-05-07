import os
import sys
import json
import tempfile
import multiprocessing as mlp
import logging
from tqdm import tqdm
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.dirname(__file__))
from ad4_utils import AD4Mapper

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s][%(levelname)s]%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class PT_EF:
    def __init__(self, db_path="/DB/forDockGPU-LITPCBA", save_path=None, gpu_bs=256, active_only=False):
        self.db_path = db_path
        if save_path is None:
            save_path = os.getcwd()
        self.save_path = save_path
        self.filedict = self._get_inputfiles(active_only=active_only)
        self.config = self._get_inputconfig()
        self.scoring_fn = "vina"
        self.ex = 8
        self.gpu_bs = gpu_bs

    def _get_inputfiles(self, active_only=False):
        filedict = {}
        for _,targets,_ in os.walk(self.db_path):
            break
        for target in targets:
            # temp drop
            if target == "ALDH1":
                continue
            filedict[target] = []
            # make save dirs
            os.makedirs(os.path.join(self.save_path, target), exist_ok=True)
            # get "active" input files
            for _,_,fns in os.walk(os.path.join(self.db_path, target, "active")):
                break
            for fn in fns:
                filedict[target].append((
                    os.path.join(self.db_path, target, "active", fn),
                    os.path.join(self.save_path, target, fn.replace(".pdbqt", "_out.pdbqt")),
                    fn.split("-")[1].replace(".pdbqt", "")
                ))
            # get "inactive" input files
            if not active_only:
                for _,_,fns in os.walk(os.path.join(self.db_path, target, "inactive")):
                    break
                for fn in fns:
                    filedict[target].append((
                        os.path.join(self.db_path, target, "inactive", fn),
                        os.path.join(self.save_path, target, fn.replace(".pdbqt", "_out.pdbqt")),
                        fn.split("-")[1].replace(".pdbqt", "")
                    ))
        with open(os.path.join(self.save_path, "fileinfo.json"), "w") as f:
            json.dump(filedict, f, indent=2)
        return filedict

    def _get_inputconfig(self):
        with open(os.path.join(self.db_path, "pockets.json"), "r") as f:
            info = json.load(f)
        return info

    def set_params(self, scoring_fn="vina", ex=8):
        if scoring_fn not in ["vina", "vinardo", "ad4"]:
            logging.error("scoring_fn must be one of vina, vinardo, ad4")
            exit()
        self.scoring_fn = scoring_fn
        self.ex = ex

    def _getdockcmd_cpu(self, target_list=None):
        if target_list is None:
            target_list = self.filedict.keys()
        cmdlist = []

        for target in target_list:
            if self.scoring_fn == "ad4":
                mapdir = tempfile.mkdtemp()
                admapper = AD4Mapper(center=[self.config[target]["pocket"]["center_x"], 
                    self.config[target]["pocket"]["center_y"], 
                    self.config[target]["pocket"]["center_z"]], 
                    box_size=[self.config[target]["pocket"]["size_x"], 
                    self.config[target]["pocket"]["size_y"], 
                    self.config[target]["pocket"]["size_z"]], spacing=0.25) #spacing
                admapper.generate_ad4_map(os.path.join(self.db_path, self.config[target]["protein"]), [t[0] for t in self.filedict[target]], mapdir)
            # run dock for active
            for indata, outdata, oriidx in self.filedict[target]:
                if self.scoring_fn == "ad4":
                    cmd = "vina --maps {} --ligand {} --scoring {} \
                            --center_x {} --center_y {} --center_z {} \
                            --size_x {} --size_y {} --size_z {} \
                            --seed 181129 --exhaustiveness {} --cpu 1 \
                            --num_modes 1 --out {} --verbosity 0".format(
                                    os.path.join(mapdir, os.path.splitext(os.path.basename(self.config[target]["protein"]))[0]),
                                    indata, self.scoring_fn, 
                                    self.config[target]["pocket"]["center_x"],
                                    self.config[target]["pocket"]["center_y"],
                                    self.config[target]["pocket"]["center_z"],
                                    self.config[target]["pocket"]["size_x"],
                                    self.config[target]["pocket"]["size_y"],
                                    self.config[target]["pocket"]["size_z"],
                                    self.ex, outdata
                            )
                else:
                    cmd = "vina --receptor {} --ligand {} --scoring {} \
                            --center_x {} --center_y {} --center_z {} \
                            --size_x {} --size_y {} --size_z {} \
                            --seed 181129 --exhaustiveness {} --cpu 1 \
                            --num_modes 1 --out {} --verbosity 0".format(
                                    os.path.join(self.db_path, self.config[target]["protein"]),
                                    indata, self.scoring_fn, 
                                    self.config[target]["pocket"]["center_x"],
                                    self.config[target]["pocket"]["center_y"],
                                    self.config[target]["pocket"]["center_z"],
                                    self.config[target]["pocket"]["size_x"],
                                    self.config[target]["pocket"]["size_y"],
                                    self.config[target]["pocket"]["size_z"],
                                    self.ex, outdata
                            )
                cmdlist.append((
                    target,
                    oriidx,
                    cmd
                ))
        return cmdlist
    
    def _getdockcmd_gpu(self, target_list=None):
        if target_list is None:
            target_list = self.filedict.keys()
        cmdlist = []
        inlist1 = [] # the internal list
        inlist2 = [] # the external list
        for target in target_list:
            if self.scoring_fn == "ad4":
                mapdir = tempfile.mkdtemp()
                admapper = AD4Mapper(center=[self.config[target]["pocket"]["center_x"], 
                    self.config[target]["pocket"]["center_y"], 
                    self.config[target]["pocket"]["center_z"]], 
                    box_size=[self.config[target]["pocket"]["size_x"], 
                    self.config[target]["pocket"]["size_y"], 
                    self.config[target]["pocket"]["size_z"]], spacing=0.25) #spacing
                admapper.generate_ad4_map(os.path.join(self.db_path, self.config[target]["protein"]), [t[0] for t in self.filedict[target]], mapdir)
            count = 0
            # run dock for active
            for indata, _, oriidx in self.filedict[target]:
                if count == self.gpu_bs:
                    count = 0
                    inlist2.append(inlist1)
                    inlist1 = []
                inlist1.append((indata, oriidx))
                count += 1
            inlist2.append(inlist1)
            inlist1 = []
            count = 0
            for l in inlist2:
                if self.scoring_fn == "ad4":
                    cmd = "vina --maps {} --gpu_batch {} --scoring {} \
                        --center_x {} --center_y {} --center_z {} \
                        --size_x {} --size_y {} --size_z {} \
                        --seed 181129 --max_step 20 --exhaustiveness {} \
                        --num_modes 1 --dir {} --verbosity 0".format(
                                os.path.join(mapdir, os.path.splitext(os.path.basename(self.config[target]["protein"]))[0]),
                                "  ".join([u[0] for u in l]), self.scoring_fn, 
                                self.config[target]["pocket"]["center_x"],
                                self.config[target]["pocket"]["center_y"],
                                self.config[target]["pocket"]["center_z"],
                                self.config[target]["pocket"]["size_x"],
                                self.config[target]["pocket"]["size_y"],
                                self.config[target]["pocket"]["size_z"],
                                self.ex, 
                                os.path.join(self.save_path, target)
                        )
                else:
                    cmd = "vina --receptor {} --gpu_batch {} --scoring {} \
                            --center_x {} --center_y {} --center_z {} \
                            --size_x {} --size_y {} --size_z {} \
                            --seed 181129 --max_step 20 --exhaustiveness {} \
                            --num_modes 1 --dir {} --verbosity 0".format(
                                    os.path.join(self.db_path, self.config[target]["protein"]),
                                    "  ".join([u[0] for u in l]), self.scoring_fn, 
                                    self.config[target]["pocket"]["center_x"],
                                    self.config[target]["pocket"]["center_y"],
                                    self.config[target]["pocket"]["center_z"],
                                    self.config[target]["pocket"]["size_x"],
                                    self.config[target]["pocket"]["size_y"],
                                    self.config[target]["pocket"]["size_z"],
                                    self.ex, 
                                    os.path.join(self.save_path, target)
                            )
                cmdlist.append((
                    target,
                    "|".join([u[1] for u in l]),
                    cmd
                ))
        return cmdlist

    def system_run(self, info):
        target, idx, cmd = info
        try:
            start_time = time.time()
            res = os.system(cmd)
            if res != 0:
                logging.error("{} failed".format(target))
            else:
                end_time = time.time()
                with open(os.path.join(self.save_path, "timelog.csv"), "a") as f:
                    f.write("{},{},{}\n".format(target, end_time - start_time, idx))
        except:
            traceback.print_exc()
            logging.error("{} failed".format(target))

    def run_dock(self, mode="gpu", target_list=None):
        if mode == "cpu":
            cmdlist = self._getdockcmd_cpu(target_list)
            p = mlp.Pool(mlp.cpu_count())
            logging.info("num. of process: {}".format(mlp.cpu_count()))
        elif mode == "gpu":
            cmdlist = self._getdockcmd_gpu(target_list)
            p = mlp.Pool(1)
        else:
            logging.error("mode must be one of cpu, gpu")
            exit()
        res = list(tqdm(p.imap(self.system_run, cmdlist), total=len(cmdlist)))
        p.close()
        p.join()

    def _read_score(self, target_list=None):
        if target_list is None:
            target_list = self.filedict.keys()
        score = {}
        for target in target_list:
            score[target] = []
            for _,_,fns in os.walk(os.path.join(self.save_path, target)):
                break
            for fn in fns:
                label, idx = fn.replace("_out.pdbqt", "").split("-")
                s = float(open(os.path.join(self.save_path, target, fn), "r").readlines()[1].split()[3])
                if s > 0:
                    logging.warning("[score>0] {} {} {} {}".format(target, label, idx, s))
                    continue
                score[target].append((label, idx, s))
            score[target] = sorted(score[target], key=lambda x: x[2])
        with open(os.path.join(self.save_path, "docking_score.json"), "w") as f:
            json.dump(score, f, indent=2)
        return score

    def _draw_distribution(self, score):
        for target in score:
            active = [s[2] for s in score[target] if s[0] == "active"]
            inactive = [s[2] for s in score[target] if s[0] == "inactive"]
            sns.distplot(active, label="active", hist=True, kde=True, kde_kws={"shade": True})
            sns.distplot(inactive, label="inactive", hist=True, kde=True, kde_kws={"shade": True})
            plt.legend()
            plt.savefig(os.path.join(self.save_path, "{}_distribution.png".format(target)))
            plt.close()

    def _read_time(self, target_list=None):
        if target_list is None:
            target_list = self.filedict.keys()
        timelog = {}
        res = {}
        with open(os.path.join(self.save_path, "timelog.csv"), "r") as f:
            for line in f:
                target, time_used, _ = line.strip().split(",")
                timelog[target] = timelog.get(target, []) + [float(time_used)]
        for target in timelog:
            if target in target_list:
                res[target] = [
                    np.sum(timelog[target]),
                    np.mean(timelog[target]), 
                    np.median(timelog[target]),
                    np.std(timelog[target])
                ]
        with open(os.path.join(self.save_path, "timeused.csv"), "w") as f:
            f.write("target,time_used,time_mean,time_median,time_std\n")
            for target in res:
                f.write("{},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                    target, res[target][0], res[target][1], 
                    res[target][2], res[target][3]))
        return time

    def _calc_ef(self, score, top_ratio=[0.05, 0.1, 0.15, 0.20, 0.30]):
        ef = {}
        for target in score:
            ef[target] = np.array([s[0] == "active" for s in score[target]])
            active_num = ef[target].sum()
            ef[target] = [ef[target][:int(len(ef[target]) * r)].sum()/active_num for r in top_ratio]
        with open(os.path.join(self.save_path, "ef.csv"), "w") as f:
            f.write("target,{}\n".format(",".join(["top{:.0f}%".format(r*100) for r in top_ratio])))
            for target in ef:
                f.write("{},{}\n".format(target, ",".join(["{:.2f}%".format(r*100) for r in ef[target]])))
        return ef

    def analyze_ef(self, target_list=None):
        self._read_time(target_list)
        score = self._read_score(target_list)
        self._draw_distribution(score)
        self._calc_ef(score)
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scoring_fn", type=str, default="vina",
        choices=["vina", "vinardo", "ad4"], help="scoring function")
    parser.add_argument("-m", "--mode", type=str, default="cpu",
        choices=["cpu", "gpu"], help="cpu or gpu")
    parser.add_argument("-t", "--targets", type=str, default=None,
        help="target list, comma separated")
    parser.add_argument("-b", "--batch_size", type=int, default=256,
        help="batch size in GPU mode")
    parser.add_argument("--active_only", action="store_true", 
        help="only dock active ligands")
    parser.add_argument("-e", "--ex", type=int, default=1)
    args = parser.parse_args()

    pt = PT_EF(
        gpu_bs=args.batch_size, 
        active_only=args.active_only
    )
    pt.set_params(
        scoring_fn=args.scoring_fn, 
        ex=args.ex
    )
    if args.targets is not None:
        pt.run_dock(mode=args.mode, target_list=args.targets.split(","))
        pt.analyze_ef(target_list=args.targets.split(","))
    else:
        pt.run_dock(mode=args.mode)
        pt.analyze_ef()
