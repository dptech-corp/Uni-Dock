import csv
import subprocess
import glob
import os
from pathlib import Path
def find_sdf_files_recursively(folder_path):
    # 将字符串路径转换为Path对象
    folder_path = Path(folder_path)
    
    # 使用rglob查找所有以.sdf结尾的文件
    sdf_files = list(folder_path.rglob('*.sdf'))
    
    # 将Path对象列表转换为字符串路径列表
    sdf_files_str = [str(file) for file in sdf_files]
    
    return sdf_files_str

# 假设我们要搜索的文件夹路径为'/path/to/your/folder'
folder_path = '/home/yxyy/Documents/code/Uni-Dock/unidock/example/screening_test/astex_ligand_sdf'

# 调用函数并打印结果
sdf_files = find_sdf_files_recursively(folder_path)
sdf_files.sort()
for sdf in sdf_files:
    print(sdf)
def build_commands_from_csv(csv_file):
    commands = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        i=0
        file_path = "/home/yxyy/Documents/code/Uni-Dock/unidock/example/screening_test/def_ligands.txt"
        
        for row in reader:
            with open(file_path, "w") as file:
                file.write(sdf_files[i])
            pdb_id = row['PDB_ID']
            x = row['X']
            y = row['Y']
            z = row['Z']
            command = f"../../build/unidock  --receptor ./receptor_grids/{pdb_id}/unidock/receptor_grids/protein_conf_0/protein.pdbqt --ligand_index def_ligands.txt --center_x {x} --center_y {y} --center_z {z} --size_x 27 --size_y 27 --size_z 27 --dir ./result/def/2048 --exhaustiveness 2048 --max_step 1 --num_modes 2048 --scoring vina --refine_step 5 --seed 5 --verbosity 1"
            commands.append(command)
            try:
                # 使用shell=True来执行字符串形式的命令
                subprocess.run(command, shell=True, check=True)
                print(f"命令执行成功: {command}")
            except subprocess.CalledProcessError as e:
                print(f"命令执行失败: {command}\n错误信息: {e}")
            i+=1
    return commands

# 调用函数并打印结果
csv_file = '/home/yxyy/Documents/code/Uni-Dock/unidock/example/screening_test/astex/pdb_center.csv'
commands = build_commands_from_csv(csv_file)
# for command in commands:
#     print(command)
