# -*- coding: utf-8 -*-
import os
from pathlib import Path

# 定义根目录
root_dir = Path("/home/liujun666/My_datasets/LRS2_poisoned/mvlrs_v1/pretrain")

# 定义输出文件
output_file = "poisoned_file_list.txt"

# 打开输出文件以写入
with open(output_file, "w") as f:
    # 遍历根目录及其子目录
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # 检查文件是否为 .npy 文件
            if file.endswith(".npy"):
                # 获取文件的相对路径（不包括文件扩展名）
                relative_path = Path(subdir) / file
                relative_path_without_ext = relative_path.relative_to(root_dir).with_suffix('')
                # 将路径写入文件
                f.write(f"{relative_path_without_ext}\n")

print(f"File list has been written to {output_file}")