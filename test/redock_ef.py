import os
import hmtdock
import csv
from rmsdfn import rmsd
import subprocess as sp

'''
    Using CASF-PPARG system to test Enhancement Ability of Vina1.2-GPU
'''

output_path = "/root/dockdata/gpu_redock/LIT"
ligand_data_path = "/root/dockdata/LIT/LIT_prepared/LIT-PCBA-Ligands"
dock_type = "active-rigid"
input_path = ligand_data_path + "/" + dock_type + "/total"
ligand_id_file = ligand_data_path + "/" + dock_type + "/PPARG/actives.smi" 
print(ligand_id_file)
protein_file = "/root/dockdata/LIT/LIT_prepared/LIT-PCBA-Protein/PPARG.pdbqt"

def dock(ligpath_list, scoring_func, rigid_receptor, flex_receptor, center_x, center_y, center_z, size_x, size_y,\
    size_z, exhaustiveness, energy_range, min_rmsd, num_of_poses, max_step):
    if flex_receptor == None:
        cmd = "/root/Vina1.2-GPU/build/linux/release/vina --scoring {} --receptor {} \
        --center_x {} --center_y {} --center_z {} --size_x {} \
        --size_y {} --size_z {} --exhaustiveness {} --energy_range {} \
        --num_modes {} --max_step {} --dir {} --verbosity 2 \
        --gpu_batch {} ".format(scoring_func, 
            rigid_receptor, center_x, center_y, center_z, 
            size_x, size_y, size_z, 
            exhaustiveness, energy_range, num_of_poses, 
            max_step, output_path + "/" + dock_type, ligpath_list)
    else:
        cmd = "/root/Vina1.2-GPU/build/linux/release/vina --scoring {} --receptor {} --flex {} --ligand {} \
            --center_x {} --center_y {} --center_z {} --size_x {} \
            --size_y {} --size_z {} --exhaustiveness {} --energy_range {} \
            --num_modes {} --dir {}".format(scoring_func, 
                rigid_receptor, flex_receptor, 
                ligpath, center_x, center_y, center_z, 
                size_x, size_y, size_z, 
                exhaustiveness, energy_range, num_of_poses, 
                max_step, output_path)
    print(cmd)
    sp.run(cmd, shell=True)

def get_energy(inpdbqt):
    if not os.path.exists(inpdbqt):
        print("PDBQT file not found")
        return 
    energies = []
    with open(inpdbqt, "r") as f:
        lines = f.readlines()
        for l in lines:
            if len(l) == 0:
                continue
            words = l.split()
            if len(words) > 3 and words[0] == "REMARK" and words[2] == "RESULT:":
                energies.append(words[3])
    return energies

ligand_id_list = []
with open(ligand_id_file, "r") as f:
    lines = f.readlines()
    for l in lines:
        if len(l) == 0:
            continue
        words = l.split(" ")
        ligand_id_list.append(words[1][:-1])

print(ligand_id_list)

ligand_file_input_string = ""
ligand_id_input_list = []
cnt = 0
for ligand_id in ligand_id_list:
    cnt = cnt + 1
    # if int(ligand_id) > 144206385:
        # continue
    if os.path.isfile(input_path + "/" + ligand_id + "_variant1.pdbqt"):
        ligand_file_input_string = ligand_file_input_string + " " + ligand_id + "_variant1.pdbqt"
        ligand_id_input_list.append(ligand_id)
    if (cnt % 120 == 0):
        cnt = 0
        os.chdir(input_path)
        print(input_path)
        dock(ligand_file_input_string, 'vina', protein_file, None, 29.63, 0.9, 26.88, 16, 26.92, 20, 512, 9, 1, 3,
                20)
        # write output to csv 
        os.chdir(output_path + "/" + dock_type)
        for ligand_id_input in ligand_id_input_list:
            with open(output_path + "/" + dock_type + ".csv", "a") as output_file:
                energies = get_energy(ligand_id_input + "_variant1_out.pdbqt")
                print(energies)
                if not energies == None:
                    output_file.write(ligand_id_input+","+energies[0]+"\n")
        # clear list and string
        ligand_id_input_list = []
        ligand_file_input_string = ""
if (cnt > 0):
    os.chdir(input_path)
    print(input_path)
    dock(ligand_file_input_string, 'vina', protein_file, None, 29.63, 0.9, 26.88, 16, 26.92, 20, 512, 9, 1, 3,
            20)
    # write output to csv 
    os.chdir(output_path + "/" + dock_type)
    for ligand_id_input in ligand_id_input_list:
        with open(output_path + "/" + dock_type + ".csv", "a") as output_file:
            energies = get_energy(ligand_id_input + "_variant1_out.pdbqt")
            print(energies)
            if not energies == None:
                output_file.write(ligand_id_input+","+energies[0]+"\n")
        
        
        
