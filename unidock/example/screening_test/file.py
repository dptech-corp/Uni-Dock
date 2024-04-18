from pathlib import Path

def get_all_file_paths(directory):
    files = " "
    path = Path(directory)
    for file in path.rglob('*'):
        if file.is_file():
            files += str(file) + " "
    return files
    # return [str(file) for file in path.rglob('*') if file.is_file()]

# 使用示例
directory = '/home/yxyy/Documents/code/Uni-Dock/unidock/example/screening_test/astex_ligand_sdf/'
all_file_paths = get_all_file_paths(directory)
print(all_file_paths)