from unidock_tools.ligandPrepare import prepare_ligands
import os, shutil
import subprocess
import glob
import argparse

from rdkit import Chem


class UniDock():
    def __init__(self, receptor:str, scoring:str='vina', output_dir:str='./docking_results'):
        """
        Initialize the UniDock class with the given parameters.

        :param scoring: Docking scoring function. Options: 'vina', 'vinardo', 'gnina'. Default: 'vina'.
        :param output_dir: Directory where the docking results will be saved. Default: './docking_results'.
        """
        if scoring in ['vina', 'vinardo']:
            self.scoring = scoring
            self.rescoring = None
        elif scoring in ['gnina']:
            self.scoring = 'vina'
            self.rescoring = scoring

        self.receptor = receptor
        self.output_dir = output_dir
        self.ligand_input_method = ["ligand", "batch", "gpu_batch", "ligand_index"]

        self.command_scoring = '--scoring  %s'%self.scoring
        self.command_ligand = ''

        # delete output directory if it already exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.ligands_prepared_dir = "%s/ligands_prepared"%self.output_dir
        os.makedirs(self.ligands_prepared_dir, exist_ok=True)

    def set_receptor(self, receptor:str):
        """
        Set the receptor file for the docking.

        :param receptor: Path to the receptor file (PDBQT format).
        """
        self.receptor = receptor

    def set_ligand_index(self, ligand_index:str):
        """
        Set the ligand index file for the docking.

        :param ligandv: Path to the ligand index file.
        :param need_prepare: Whether the ligands need to be prepared. Default: True.
        """
        self.PDBQT = []
        self.SDF = []
        with open(ligand_index, 'r') as f:
            ligands = f.readlines()
            ligands = [ligand.strip() for ligand in ligands]
        
        for ligand in ligands:
            basename =  os.path.basename(ligand)
            format = basename.split('.')[-1]
            if format == "sdf":
                self.SDF.append(ligand)
            elif format == "pdbqt":
                self.PDBQT.append(ligand)

        if len(self.SDF) > 0:
            print("Preparing ligand file...")
            self.ligands = prepare_ligands(self.SDF, output_dir=self.ligands_prepared_dir) + self.PDBQT
        else:
            self.ligands = self.PDBQT
        
        ligand_prepared_index_file = "%s/ligands.dat"%self.output_dir
        with open(ligand_prepared_index_file, 'w') as f:
            for ligand in self.ligands:
                f.write("%s\n"%ligand)

        self.command_ligand = '--ligand_index %s'%(ligand_prepared_index_file)
    
    def set_gpu_batch(self, ligands:list):
        """
        Set the ligands GPU batch for the docking.

        :param ligandv: Path to the ligand index file.
        :param need_prepare: Whether the ligands need to be prepared. Default: True.
        """
        self.PDBQT = []
        self.SDF = []
        for ligand in ligands:
            basename =  os.path.basename(ligand)
            format = basename.split('.')[-1]
            if format == "sdf":
                self.SDF.append(ligand)
            elif format == "pdbqt":
                self.PDBQT.append(ligand)

        if len(self.SDF) > 0:
            print("Preparing ligand file...")
            self.ligands = prepare_ligands(self.SDF, output_dir=self.ligands_prepared_dir) + self.PDBQT
        else:
            self.ligands = self.PDBQT
        
        self.command_ligand = '--gpu_batch %s'%(" ".join(self.ligands))

    def set_ligand(self, ligand:str):
        """
        Set the ligand for the docking.

        :param ligandv: Path to the ligand index file.
        :param need_prepare: Whether the ligands need to be prepared. Default: True.
        """
        self.PDBQT = []
        self.SDF = []
        basename =  os.path.basename(ligand)
        format = basename.split('.')[-1]
        if format == "sdf":
            print("Preparing ligand file...")
            self.ligands = prepare_ligands([ligand], output_dir=self.ligands_prepared_dir)
        elif format == "pdbqt":
            self.ligands = [ligand]
        
        self.command_ligand = '--ligand %s'%self.ligands[0]
            
    def set_batch(self, ligands:list):
        """
        Set the ligands batch for the docking.

        :param ligandv: Path to the ligand index file.
        :param need_prepare: Whether the ligands need to be prepared. Default: True.
        """
        self.PDBQT = []
        self.SDF = []
        for ligand in ligands:
            basename =  os.path.basename(ligand)
            format = basename.split('.')[-1]
            if format == "sdf":
                self.SDF.append(ligand)
            elif format == "pdbqt":
                self.PDBQT.append(ligand)

        if len(self.SDF) > 0:
            print("Preparing ligand file...")
            self.ligands = prepare_ligands(self.SDF, output_dir=self.ligands_prepared_dir) + self.PDBQT
        else:
            self.ligands = self.PDBQT

        self.command_ligand = '--batch %s'%(" ".join(self.ligands))

    def set_rescoring(self, rescoring:str):
        """
        Set the rescoring function for the docking.

        :param rescoring: Rescoring function. Options: 'gnina'. 
        """
        self.rescoring=rescoring

    def dock(self):
        """
        Perform the docking using the specified parameters and settings.
        """
        self._call_unidock()
        if self.rescoring:
            self._call_gnina()
    
    def _call_unidock(self):
        command = " ".join(self.config)
        print("command:", command)
        subprocess.run(" ".join(self.config), shell=True)
    
    def _call_gnina(self):
        ligands_basename = [get_file_prefix(filename) for filename in self.ligands]
        docking_poses = ["%s_out.sdf"%basename for basename in ligands_basename]
        
        for pose in docking_poses:
            result = subprocess.run("gnina -r %s -l %s --score_only"%(self.receptor, pose), shell=True, stdout=subprocess.PIPE, text=True)
            print(result.stdout)
            self._add_record(pose, result.stdout)
    
    def _add_record(self, sdf, stdout):
        print("Recording CNNscores into SDF Files...")
        suppl = Chem.SDMolSupplier(sdf, removeHs=False)
        mols = [mol for mol in suppl if mol is not None]
        lines = [line.strip() for line in stdout.split('\n')]

        CNNscores = [line.split()[-1] for line in lines if "CNNscore" in line]
        CNNaffinitys = [line.split()[-1] for line in lines if "CNNaffinity" in line ]
        #print(len(mols), len(CNNscores), len(CNNaffinitys))

        for idx, mol in enumerate(mols):
            mol.SetDoubleProp('CNNscores', float(CNNscores[idx]))
            mol.SetDoubleProp('CNNaffinitys', float(CNNaffinitys[idx]))
        
        writer = Chem.SDWriter(sdf)
        for mol in mols:
            writer.write(mol)
        writer.close()

    def _recreate_command_line(self, args):
        self.command_line = ""
        for arg, value in vars(args).items():
            if arg in ['scoring'] or arg in self.ligand_input_method:
                continue
            if value is not None:
                if isinstance(value, list):
                    value = ' '.join(map(str, value))
                    self.command_line += f"--{arg} {value} "
                elif isinstance(value, bool):
                    if value:
                        self.command_line += f"--{arg} "
                elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                    self.command_line += f"--{arg} {value} "
                else:
                    raise ValueError(f"Unhandled value type for argument {arg}: {type(value)}")
        self.command_line = self.command_line.strip()
    
    def _check_args_and_return(self, args):
        assigned_properties = [prop for prop in self.ligand_input_method if getattr(args, prop, None) is not None]
        #print([getattr(args, prop)for prop in self.ligand_input_method if getattr(args, prop, None) is not None])
        
        if len(assigned_properties) != 1:
            raise ValueError("please input ligand file(s) properly.")
        return assigned_properties[0]

    def _get_config(self, args):
        self._recreate_command_line(args)

        self.config=['unidock']
        self.config.append(self.command_scoring)
        self.config.append(self.command_ligand)
        self.config.append(self.command_line)

def get_file_prefix(file_path):
    file_name = os.path.basename(file_path)  
    file_prefix, _ = os.path.splitext(file_name)
    return file_prefix

def main():
    parser = argparse.ArgumentParser(description="Docking program")

    # Input files
    parser.add_argument("--receptor", type=str, help="rigid part of the receptor (PDBQT)")
    parser.add_argument("--flex", type=str, help="flexible side chains, if any")
    parser.add_argument("--ligand", type=str, help="ligand (SDF)")
    parser.add_argument("--ligand_index", type=str, help="file containing paths to ligands")
    parser.add_argument("--batch", dest="batch",nargs='+', help="batch ligand (SDF)")
    parser.add_argument("--gpu_batch", dest="gpu_batch",nargs='+', help="gpu batch ligand (SDF)")

    # Scoring function
    parser.add_argument("--scoring", type=str, default="vina", help="scoring function (ad4, vina or vinardo)")

    # Search space
    parser.add_argument("--maps", type=str, help="affinity maps for the autodock4.2 (ad4) or vina scoring function")
    parser.add_argument("--center_x", type=float, help="X coordinate of the center (Angstrom)")
    parser.add_argument("--center_y", type=float, help="Y coordinate of the center (Angstrom)")
    parser.add_argument("--center_z", type=float, help="Z coordinate of the center (Angstrom)")
    parser.add_argument("--size_x", type=float, help="size in the X dimension (Angstrom)")
    parser.add_argument("--size_y", type=float, help="size in the Y dimension (Angstrom)")
    parser.add_argument("--size_z", type=float, help="size in the Z dimension (Angstrom)")
    parser.add_argument("--autobox", action="store_true", help="set maps dimensions based on input ligand(s) (for --score_only and --local_only)")

    # Output
    parser.add_argument("--out", type=str, help="output models (PDBQT), the default is chosen based on the ligand file name")
    parser.add_argument("--dir", type=str, default="./docking_results", help="output directory for batch mode")
    parser.add_argument("--write_maps", type=str, help="output filename (directory + prefix name) for maps. Option --force_even_voxels may be needed to comply with .map format")

    # Misc
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--keep_nonpolar_H", action="store_true")
    parser.add_argument("--cpu", type=int, default=0, help="the number of CPUs to use (the default is to try to detect the number of CPUs or, failing that, use 1)")
    parser.add_argument("--seed", type=int, default=0, help="explicit random seed")
    #parser.add_argument("--exhaustiveness", type=int, default=8, help="exhaustiveness of the global search (roughly proportional to time): 1+")
    #parser.add_argument("--max_evals", type=int, default=0, help="number of evaluations in each MC run (if zero, which is the default, the number of MC steps is based on heuristics)")
    parser.add_argument("--num_modes", type=int, default=9, help="maximum number of binding modes to generate")
    parser.add_argument("--min_rmsd", type=float, default=1, help="minimum RMSD between output poses")
    parser.add_argument("--energy_range", type=float, default=3, help="maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)")
    parser.add_argument("--spacing", type=float, default=0.375, help="grid spacing (Angstrom)")
    parser.add_argument("--verbosity", type=int, default=1, help="verbosity (0=no output, 1=normal, 2=verbose)")
    parser.add_argument("--max_step", type=int, default=0, help="maximum number of steps in each MC run (if zero, which is the default, the number of MC steps is based on heuristics)")
    parser.add_argument("--refine_step", type=int, default=5, help="number of steps in refinement, default=5")
    parser.add_argument("--max_gpu_memory", type=int, default=0, help="maximum gpu memory to use (default=0, use all available GPU memory to obtain maximum batch size)")
    parser.add_argument("--search_mode", type=str, default="balance", help="search mode of vina (fast, balance, detail), using recommended settings of exhaustiveness and search steps; the higher the computational complexity, the higher the accuracy, but the larger the computational cost")

    args = parser.parse_args()

    unidock = UniDock(args.receptor, args.scoring, args.dir)

    ligand_input_method = unidock._check_args_and_return(args)
    if ligand_input_method == "ligand":
        unidock.set_ligand(args.ligand)
    elif ligand_input_method == "batch":
        unidock.set_batch(args.batch)
    elif ligand_input_method == "gpu_batch":
        unidock.set_gpu_batch(args.gpu_batch)
    elif ligand_input_method == "ligand_index":
        unidock.set_ligand_index(args.ligand_index)
    print("command_ligand: ",unidock.command_ligand)
    
    unidock._get_config(args)
    unidock.dock()
    

if __name__ == "__main__":
    main()
