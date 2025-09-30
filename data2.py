import os
import shutil
from collections import defaultdict
import cv2
import sys
import platform


def ensure_directory_exists(path, folder_name=None):
    """
    确保指定路径中的目录存在，如果不存在则创建

    Args:
        path: 目标路径（可以是完整路径，也可以是相对路径）
        folder_name: 需要检查的文件夹名称（可选）

    Returns:
        str: 创建/确认存在的完整文件夹路径
    """
    try:
        # 处理文件夹名称
        target_path = os.path.join(path, folder_name) if folder_name else path

        # 检查并创建文件夹
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            print(f"✅ 创建文件夹: {target_path}")
        else:
            print(f"⏩ 文件夹已存在: {target_path}")

        return target_path

    except Exception as e:
        print(f"❌ 创建文件夹失败: {e}")
        return None


def display_image_and_get_input(image_path):
    """
    打开图片显示在窗口中，等待用户输入文字，输入后关闭窗口

    参数:
        image_path (str): 图片文件路径

    返回:
        str: 用户输入的文本内容
    """

    try:
        # 读取图片
        image = cv2.imread(image_path)

        cv2.imshow('0', image)
        cv2.moveWindow('0', 200, 200)
        input = cv2.waitKey(0)
        # 在控制台提示用户输入
        # sys.stdout.flush()

        # 等待用户输入（在控制台）
        # user_input = sys.stdin.readline().strip()

        return input

    except Exception as e:
        print(f"❌ 发生错误: {str(e)}", file=sys.stderr)
        return None

    finally:
        # 确保无论发生什么都要关闭窗口
        try:
            cv2.destroyAllWindows()
            # 强制关闭窗口（对于某些系统的后台线程问题）
            if platform.system() == "Windows":
                cv2.waitKey(1)  # 允许窗口有时间关闭
        except:
            pass  # 忽略关闭过程中的错误


def sync_blank_files(folder1, folder2):
    """
    :param folder1: 第一个文件夹路径
    :param folder2: 第二个文件夹路径
    """
    tar_path=[None]*4
    tar_path[0] = ensure_directory_exists(folder2,'clean')
    tar_path[1] = ensure_directory_exists(folder2, 'dirty')
    tar_path[2] = ensure_directory_exists(folder2, 'unknown')
    # 获取两个文件夹的文件列表（排除子目录）
    files1 = {os.path.join(folder1, f) for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}

    for filename in files1:
        num = display_image_and_get_input(filename)
        shutil.copy2(filename, tar_path[int(num)-49])

if __name__ == "__main__":
    # 示例用法
    dir_a = "C:\\Users\\34765\\Desktop\\ruiwenzhou"  # 替换为实际路径
    dir_b = "C:\\Users\\34765\\Desktop\\classification"  # 替换为实际路径


    sync_blank_files(dir_a, dir_b)