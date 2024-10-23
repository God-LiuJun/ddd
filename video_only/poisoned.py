# -*- coding: utf-8 -*-
import os
from pathlib import Path

# �����Ŀ¼
root_dir = Path("/home/liujun666/My_datasets/LRS2_poisoned/mvlrs_v1/pretrain")

# ��������ļ�
output_file = "poisoned_file_list.txt"

# ������ļ���д��
with open(output_file, "w") as f:
    # ������Ŀ¼������Ŀ¼
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # ����ļ��Ƿ�Ϊ .npy �ļ�
            if file.endswith(".npy"):
                # ��ȡ�ļ������·�����������ļ���չ����
                relative_path = Path(subdir) / file
                relative_path_without_ext = relative_path.relative_to(root_dir).with_suffix('')
                # ��·��д���ļ�
                f.write(f"{relative_path_without_ext}\n")

print(f"File list has been written to {output_file}")