import re
import os
import shutil


def extract_market(src_path, dst_dir):
    img_names = os.listdir(src_path)
    pattern = re.compile(r'([-\d]+)_c(\d)')
    pid_container = set()
    for img_name in img_names:
        if '.jpg' not in img_name:
            continue
        print(img_name)
        # pid: 每个人的标签编号 1
        # _  : 摄像头号 2
        pid, _ = map(int, pattern.search(img_name).groups())
        # 去掉没用的图片
        if pid == 0 or pid == -1:
            continue
        shutil.copy(os.path.join(src_path, img_name), os.path.join(dst_dir, img_name))


if __name__ == '__main__':
    src_train_path = r'data/Market-1501/bounding_box_train'
    src_query_path = r'data/Market-1501/query'
    src_test_path = r'data/Market-1501/bounding_box_test'
    # 将整个market1501数据集作为训练集
    dst_dir = r'reIDdata/market1501/bounding_box_train'

    extract_market(src_train_path, dst_dir)
    extract_market(src_query_path, dst_dir)
    extract_market(src_test_path, dst_dir)
