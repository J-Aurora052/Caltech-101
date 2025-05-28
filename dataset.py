import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, transform=None, exclude_background=True):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.image_paths = []
        self.labels = []

        # 排除background类
        all_classes = sorted(os.listdir(root_dir))
        if exclude_background:
            self.classes = [c for c in all_classes if c != 'BACKGROUND_Google']
        else:
            self.classes = all_classes

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 收集图像路径和标签
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir='./caltech101', batch_size=32, exclude_background=True):
    # 数据增强和归一化
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建完整数据集
    full_dataset = Caltech101Dataset(
        root_dir=data_dir,
        transform=transform,
        exclude_background=exclude_background
    )

    # 分割训练集和测试集
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader, full_dataset.classes