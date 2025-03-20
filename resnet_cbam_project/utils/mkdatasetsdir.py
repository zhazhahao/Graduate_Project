import os


def make_market_dir(dst_dir='./'):
    market_root = os.path.join(dst_dir, 'market1501')
    train_path = os.path.join(market_root, 'bounding_box_train')
    query_path = os.path.join(market_root, 'query')
    test_path = os.path.join(market_root, 'bounding_box_test')

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(query_path):
        os.makedirs(query_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)


if __name__ == '__main__':
    make_market_dir(dst_dir='./reIDdata')
