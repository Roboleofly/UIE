import os
import argparse

def sizeof_fmt(num, suffix='B'):
    """将字节数格式化为更易读的形式"""
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return f"{num:.2f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f}Y{suffix}"

def scan_directory(path, min_keep_size=1*1024*1024):
    """
    遍历目录，删除小于 min_keep_size 的文件，
    并打印其他文件的大小，返回剩余文件的总大小
    """
    total_size = 0
    for root, dirs, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                size = os.path.getsize(filepath)
            except OSError as e:
                print(f"无法读取文件大小: {filepath} ({e})")
                continue

            if size < min_keep_size:
                try:
                    os.remove(filepath)
                    print(f"已删除（<{sizeof_fmt(min_keep_size)}）: {filepath}")
                except OSError as e:
                    print(f"删除失败: {filepath} ({e})")
            else:
                total_size += size
                print(f"{filepath} — {size} 字节 ({sizeof_fmt(size)})")

    return total_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计并清理目录下小文件")
    parser.add_argument("directory", nargs="?", default="/media/users/wk/IL_research/datasets/zhiyuan_ego_data/action_clips",
                        help="要扫描的目录，默认为当前目录")
    args = parser.parse_args()

    dir_path = args.directory
    if not os.path.isdir(dir_path):
        print(f"错误：{dir_path} 不是有效的目录。")
        exit(1)

    print(f"开始扫描目录：{dir_path}（删除小于 1 MB 的文件）\n")
    total = scan_directory(dir_path)
    print("\n扫描并清理完成。")
    print(f"剩余文件总大小: {total} 字节 ({sizeof_fmt(total)})")


# import os
# import argparse

# def sizeof_fmt(num, suffix='B'):
#     """将字节数格式化为更易读的形式（可选）"""
#     for unit in ['','K','M','G','T','P','E','Z']:
#         if abs(num) < 1024.0:
#             return f"{num:.2f}{unit}{suffix}"
#         num /= 1024.0
#     return f"{num:.2f}Y{suffix}"

# def scan_directory(path):
#     total_size = 0
#     for root, dirs, files in os.walk(path):
#         for filename in files:
#             filepath = os.path.join(root, filename)
#             try:
#                 size = os.path.getsize(filepath)
#             except OSError as e:
#                 print(f"无法获取文件大小: {filepath} ({e})")
#                 continue
#             total_size += size
#             print(f"{filepath} — {size} 字节 ({sizeof_fmt(size)})")
#     return total_size

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="统计目录下所有文件的大小")
#     parser.add_argument("directory", nargs="?", default="/media/users/wk/IL_research/datasets/zhiyuan_ego_data/action_clips",
#                         help="要扫描的目录，默认为当前目录")
#     args = parser.parse_args()

#     dir_path = args.directory
#     if not os.path.isdir(dir_path):
#         print(f"错误：{dir_path} 不是有效的目录。")
#         exit(1)

#     print(f"开始扫描目录：{dir_path}\n")
#     total = scan_directory(dir_path)
#     print("\n扫描完成。")
#     print(f"总文件数大小: {total} 字节 ({sizeof_fmt(total)})")
