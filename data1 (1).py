import os
from collections import defaultdict
from turtledemo.forest import doit1


def sync_blank_files(folder1, folder2):
    """
    :param folder1: 第一个文件夹路径
    :param folder2: 第二个文件夹路径
    """
    # 获取两个文件夹的文件列表（排除子目录）
    files1 = {os.path.splitext(f)[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
    files2 = {os.path.splitext(f)[0] for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}
    # 在folder2中创建folder1独有的文件
    video=defaultdict(lambda: [0, 0])
    for filename in files2:
        video[filename[:3]][0]+=1
        path = os.path.join(folder2, filename+'.txt')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if len(content) != 0:
                video[filename[:3]][1] += 1

    for a,b in video.items():
        print(a, b)
    print(len(files1-files2))



if __name__ == "__main__":
    # 示例用法
    dir_a = "C:\\Users\\34765\\Desktop\\zouruiwen\\image"  # 替换为实际路径
    dir_b = "C:\\Users\\34765\\Desktop\\zouruiwen\\label"  # 替换为实际路径


    sync_blank_files(dir_a, dir_b)