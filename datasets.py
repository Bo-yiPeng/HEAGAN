import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class ImageNet32NPZDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        # 假设你的 npz 文件里是 arr_0，形状为 (N, 32, 32, 3)
        self.images = data['arr_0']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, 0  # 没有标签，返回0

class ImageDataset(object):
    def __init__(self, args):
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'cifar100':
            Dt = datasets.CIFAR100
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 100
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'imgenet32':
            Dt = ImageNet32NPZDataset
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'celeba':
            Dt = datasets.ImageFolder(root='./data/celeba',
                                           transform=transforms.Compose([
                                               transforms.Resize(64),
                                               transforms.CenterCrop(64),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
        else:
            raise NotImplementedError(
                'Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'stl10':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='train+unlabeled',
                   transform=transform, download=True),
                batch_size=args.dis_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)

        elif args.dataset.lower() == 'celeba':
            self.train = torch.utils.data.DataLoader(
                dataset=Dt,
                batch_size=args.dis_bs,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True)
        elif args.dataset.lower() == 'imgenet32':
            self.train = torch.utils.data.DataLoader(
                Dt(npz_path=args.data_path, transform=transform),
                batch_size=args.dis_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        else:
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=True,
                   transform=transform, download=True),
                batch_size=args.dis_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)