import os
import hmtdock
import csv
from rmsdfn import rmsd
import subprocess as sp

output_path = input_path = "/root/dockdata/gpu_redock/vina_results"
configure_file = "/root/dockdata/redock/redocking_good/vina/CASF2016-redock-vina.csv"

def dock(ligpath, scoring_func, rigid_receptor, flex_receptor, center_x, center_y, center_z, size_x, size_y,\
    size_z, exhaustiveness, energy_range, min_rmsd, num_of_poses, max_step):
    if flex_receptor == None:
        cmd = "/root/Vina1.2-GPU/build/linux/release/vina --scoring {} --receptor {} --gpu_batch {} \
        --center_x {} --center_y {} --center_z {} --size_x {} \
        --size_y {} --size_z {} --exhaustiveness {} --energy_range {} \
        --num_modes {} --max_step {} --dir {}".format(scoring_func, 
            rigid_receptor, 
            ligpath, center_x, center_y, center_z, 
            size_x, size_y, size_z, 
            exhaustiveness, energy_range, num_of_poses, 
            max_step, output_path)
    else:
        cmd = "/root/Vina1.2-GPU/build/linux/release/vina --scoring {} --receptor {} --flex {} --gpu_batch {} \
            --center_x {} --center_y {} --center_z {} --size_x {} \
            --size_y {} --size_z {} --exhaustiveness {} --energy_range {} \
            --num_modes {} --dir {}".format(scoring_func, 
                rigid_receptor, flex_receptor, 
                ligpath, center_x, center_y, center_z, 
                size_x, size_y, size_z, 
                exhaustiveness, energy_range, num_of_poses, 
                max_step, output_path)
    sp.run(cmd, shell=True)


with open(configure_file, "r") as f:
    reader = csv.reader(f)
    header = next(reader)

    os.chdir(input_path)
    cnt = 0
    test = True
    for ls in reader:
        cnt = cnt + 1
        # if cnt > 2:
        #     break
        print(ls)
        pdbname = ls[1][1:]
        if pdbname == "2yge":
            continue
        dock(ls[1]+'_ligand.pdbqt', 'vina', ls[1]+'_protein.pdbqt', None, ls[7], ls[8], ls[9], ls[10], ls[11], ls[12], 1024, 9, 1, 1,
            200)
        # convert output 
        cmd = "obabel %s_ligand_out.pdbqt -O %s_ligand_out.pdb"%(
            pdbname, pdbname)
        sp.run(cmd, shell=True)
        cmd = "obabel %s_ligand.pdbqt -O %s_ligand.pdb"%(
            pdbname, pdbname)
        sp.run(cmd, shell=True)
        rmsd_ = rmsd(pdbname+'_ligand.pdb', pdbname+'_ligand_out.pdb')
        with open("vina_rmsd_200step_1024thread.csv", "a") as out_f:
            out_f.write(pdbname+','+str(rmsd_)+"\n")
