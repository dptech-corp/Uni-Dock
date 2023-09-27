import os
import shutil
import random
import argparse


def count_atoms(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                count += 1
    return count


def count_torsions(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("REMARK"):
                split_line = line.split()
                if len(split_line) > 2 and split_line[1] == "active" and split_line[2] == "torsions:":
                    return int(split_line[0])
    return 0


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('atom_count', type=int, help='atom count limit')
    parser.add_argument('number_of_files', type=int, help='number of files to select')
    parser.add_argument('torsion_count', type=int, help='torsion count limit')
    
    args = parser.parse_args()
    
    folder_path = "indata/def_unique_charged"  # 请替换为您的文件夹路径
    
    # 创建新文件夹
    new_folder_path = os.path.join(folder_path, f"{args.atom_count}_torsions_{args.torsion_count}")
    os.makedirs(new_folder_path, exist_ok=True)
    
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    
    # 找出所有符合条件的文件
    eligible_files = [file for file in files if count_atoms(file) < args.atom_count and count_torsions(file) < args.torsion_count]
    
    if args.number_of_files > len(eligible_files):
        print(f"Warning: Only {len(eligible_files)} files are available, less than the requested {args.number_of_files} files.")
        args.number_of_files = len(eligible_files)  # 如果可用文件数量少于请求的数量，调整数量
    
    # 随机选择指定数量的文件
    selected_files = random.sample(eligible_files, args.number_of_files)
    
    output_file_name = f"def_ligands_{args.atom_count}_torsions_{args.torsion_count}_num_{args.number_of_files}.txt"
    output_file_path = os.path.join(new_folder_path, output_file_name)
    
    with open(output_file_path, 'w') as output_file:
        for file in selected_files:
            shutil.copy(file, new_folder_path)  # 将文件复制到新文件夹
            # 将文件的路径写入到输出文件中
            output_file.write(os.path.join(new_folder_path, os.path.basename(file)) + ' ')
    
    print(f"{args.number_of_files} files have been randomly selected and copied to {new_folder_path}")
    print(f"The paths of copied files have been written to {output_file_path}")


if __name__ == "__main__":
    main()
