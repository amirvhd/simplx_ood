import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms


class cifar10_module(pl.LightningDataModule):

    def __init__(self, data_dir='./data', batch_size=8, n_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.n_workers = n_workers
        self.batch_size = batch_size

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        normalize = transforms.Normalize(mean=mean, std=std)
        self.data_transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.data_transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def setup(self, stage=None):
        train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True,
                                         transform=self.data_transform_train)
        if stage == "fit" or stage is None:
            self.train, self.val = random_split(train_dataset, [int(0.9 * len(train_dataset)),
                                                                len(train_dataset) - int(0.9 * len(train_dataset))])

        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True,
                                                 transform=self.data_transform_test)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.n_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers
        )
