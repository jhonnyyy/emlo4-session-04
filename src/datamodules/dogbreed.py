# import os
# import torch 
# import lightning as L
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from torchvision.datasets import ImageFolder

# class DogBreedDataModule(L.LightningDataModule):
#     def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, kaggle: bool = False, kaggle_api_key: str = None, kaggle_username: str = None):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.kaggle = kaggle
#         self.kaggle_api_key = kaggle_api_key
#         self.kaggle_username = kaggle_username

#         self.train_dataset = None
#         self.val_dataset = None
#         self.test_dataset = None

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])

#     def prepare_data(self):
#         if self.kaggle and not os.path.exists(self.data_dir):
#             os.environ['KAGGLE_USERNAME'] = self.kaggle_username
#             os.environ['KAGGLE_KEY'] = self.kaggle_api_key
            
#             try:
#                 from kaggle.api.kaggle_api_extended import KaggleApi
#                 api = KaggleApi()
#                 api.authenticate()
#                 api.dataset_download_files('khushikhushikhushi/dog-breed-image-dataset', path=self.data_dir, unzip=True)
#             except Exception as e:
#                 print(f"Error downloading dataset: {e}")
#                 raise

#     def setup(self, stage=None):
#         full_dataset = ImageFolder(self.data_dir, transform=self.transform)
#         self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset(full_dataset)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

#     def _split_dataset(self, dataset):
#         total_size = len(dataset)
#         train_size = int(0.7 * total_size)
#         val_size = int(0.15 * total_size)
#         test_size = total_size - train_size - val_size
#         return random_split(dataset, [train_size, val_size, test_size])
    

import os
import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class DogBreedDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, kaggle: bool = False, kaggle_api_key: str = None, kaggle_username: str = None):     
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = min(num_workers, os.cpu_count() or 1)
        self.kaggle = kaggle
        self.kaggle_api_key = kaggle_api_key
        self.kaggle_username = kaggle_username

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if self.kaggle and not os.path.exists(self.data_dir):
            os.environ['KAGGLE_USERNAME'] = self.kaggle_username
            os.environ['KAGGLE_KEY'] = self.kaggle_api_key

            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                api.dataset_download_files('khushikhushikhushi/dog-breed-image-dataset', path=self.data_dir, unzip=True)
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                raise

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        full_dataset = ImageFolder(self.data_dir, transform=transform)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self._collate_fn)

    def _collate_fn(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels