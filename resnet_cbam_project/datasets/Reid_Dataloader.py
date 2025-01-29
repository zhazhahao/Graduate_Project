import os
import glob
from typing import List, Dict, Tuple, Optional
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ReidDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, mode: str = 'train', num_classes: int = None):
        """
        Re-ID Dataset loader.

        :param data_dir: Root directory of the dataset.
        :param transform: Transformations to apply to the images.
        :param mode: Mode of the dataset, either 'train' or 'test'.
        :param num_classes: Expected number of classes (for validation)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes

        # 新增PID映射相关逻辑
        self.original_pids = []  # 保存原始PID用于调试
        self.pid_to_class = {}  # PID到class_id的映射
        self.data = self._process_data()  # 现在data中存储的是class_id


        # 验证类别数量一致性
        if num_classes is not None:
            assert len(self.pid_to_class) == num_classes, \
                f"PID数量与预期不符 (实际: {len(self.pid_to_class)}, 预期: {num_classes})"

    def _process_data(self) -> List[Tuple[str, int, int]]:
        """
        处理数据集目录，创建映射关系
        返回格式：(image_path, class_id, camid)
        """
        pattern = os.path.join(self.data_dir, '*.jpg')
        all_images = glob.glob(pattern)

        # 收集所有PID并建立映射
        pid_set = set()
        temp_data = []

        for img_path in all_images:
            filename = os.path.basename(img_path)
            original_pid = int(filename.split('_')[0])  # 提取原始PID
            camid = int(filename.split('_')[1][1])  # 提取摄像头ID

            # 跳过无效的PID（如-1）
            if original_pid == -1:
                continue

            pid_set.add(original_pid)
            temp_data.append((img_path, original_pid, camid))

        # 创建PID到class_id的映射（从0开始）
        sorted_pids = sorted(pid_set)
        self.pid_to_class = {pid: idx for idx, pid in enumerate(sorted_pids)}
        self.original_pids = sorted_pids  # 保存原始PID用于调试

        # 转换所有PID到class_id
        processed_data = []
        for img_path, pid, camid in temp_data:
            class_id = self.pid_to_class[pid]
            processed_data.append((img_path, class_id, camid))

        # 打印调试信息
        print(f"Total PIDs: {len(self.pid_to_class)}")
        print(f"PID range: {min(self.original_pids)} - {max(self.original_pids)}")

        return processed_data

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, int]:
        img_path, class_id, camid = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, class_id, camid

    def __len__(self) -> int:
        return len(self.data)


class ReidDataLoader:
    def __init__(self, dataset_name: str, root_dir: str, batch_size: int,
                 num_workers: int = 4, mode: str = 'train', num_classes: int = None):
        """
        DataLoader for multiple Re-ID datasets.

        :param dataset_name: Name of the dataset to load (e.g., 'market1501', 'cuhk03').
        :param root_dir: Root directory containing the dataset folders.
        :param batch_size: Number of samples per batch.
        :param num_workers: Number of workers for data loading.
        :param mode: Mode of the dataset, either 'train' or 'test'.
        """
        self.dataset_name = dataset_name.lower()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.num_classes = num_classes  # 新增参数

        # 初始化顺序调整
        self.dataset_dir = self._get_dataset_dir()  # 先初始化 dataset_dir
        self.transform = self._get_transforms()  # 再初始化 transform
        self.dataset = self._load_dataset()
    def _get_dataset_dir(self) -> str:
        """
        Get the specific dataset directory based on the dataset name.

        :return: Path to the dataset directory.
        """
        dataset_map = {
            'market1501': os.path.join(self.root_dir, 'Market-1501'),
            'cuhk03': os.path.join(self.root_dir, 'cuhk03'),
        }
        if self.dataset_name not in dataset_map:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return dataset_map[self.dataset_name]

    def _get_transforms(self):
        """
        Define dataset-specific transforms.

        :return: Transform pipeline.
        """
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _load_dataset(self) -> Dataset:
        """
        Load the dataset.

        :return: Dataset object.
        """
        if self.dataset_name == 'market1501':
            data_dir = os.path.join(self.dataset_dir,
                                    'bounding_box_train' if self.mode == 'train' else 'bounding_box_test')
        elif self.dataset_name == 'cuhk03':
            # CUHK03 requires a specific loading method, potentially involving mat files.
            raise NotImplementedError("CUHK03 loading not implemented.")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return ReidDataset(
            data_dir=data_dir,
            transform=self.transform,
            mode=self.mode,
            num_classes=self.num_classes  # 传递预期类别数
        )

    def get_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the dataset.

        :return: DataLoader object.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.mode == 'train',
            num_workers=self.num_workers,
            pin_memory=True
        )


# Example usage
if __name__ == "__main__":
    root_dir = "../data"
    dataloader = ReidDataLoader(
        dataset_name='market1501',
        root_dir=root_dir,
        batch_size=32,
        mode='train',
        num_classes=751  # 明确指定类别数
    )
    train_loader = dataloader.get_dataloader()

    for imgs, pids, camids in train_loader:
        print(imgs.shape, pids, camids)
        break  # 只打印第一个batch
