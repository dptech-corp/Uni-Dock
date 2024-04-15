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
folder_path = '/home/yxyy/Documents/code/Uni-Dock/unidock/example/screening_test/astex'

# 调用函数并打印结果
sdf_files = find_sdf_files_recursively(folder_path)
print(sdf_files)
for sdf in sdf_files:
    print(sdf)
