import os
import yaml
import subprocess
import shutil
import torch
import lightning as L
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class DogBreedDataModule(L.LightningDataModule):
    def __init__(self, config_path: str = "src/datamodules/config.yaml"):
        super().__init__()
        self.config_path = config_path
        self.load_config()
        self.set_kaggle_credentials()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as config_file:
                self.config = yaml.safe_load(config_file)
            print(f"Config loaded successfully from {self.config_path}")
            self.data_dir = os.path.join(os.getcwd(), "data")
            self.batch_size = self.config['training']['batch_size']
            self.num_workers = min(self.config['training']['num_workers'], 2)  # Limit number of workers
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

    def set_kaggle_credentials(self):
        try:
            os.environ['KAGGLE_USERNAME'] = self.config['kaggle']['username']
            os.environ['KAGGLE_KEY'] = self.config['kaggle']['api_key']
            print("Kaggle credentials set as environment variables")
        except KeyError as e:
            print(f"Error setting Kaggle credentials: {e}")
            print("Please ensure 'kaggle' section with 'username' and 'api_key' is present in your config file")
            raise

    def prepare_data(self):
        try:
            print(f"Downloading dataset: {self.config['dataset']['name']}")
            dataset_name = self.config['dataset']['name']
            command = f"kaggle datasets download -d {dataset_name} -p {self.data_dir} --unzip"
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print("Dataset downloaded successfully")
            print(result.stdout)
            
            # Check and reorganize the dataset if necessary
            self._reorganize_dataset()
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            print(f"Command output: {e.output}")
            raise
        except Exception as e:
            print(f"Unexpected error in prepare_data: {e}")
            raise

    def _reorganize_dataset(self):
        print("Checking dataset structure...")
        # Look for the directory containing the breed folders
        for root, dirs, files in os.walk(self.data_dir):
            if len(dirs) == 10 and all(breed in dirs for breed in ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd']):
                self.data_dir = root
                print(f"Found correct dataset directory: {self.data_dir}")
                break
        else:
            raise FileNotFoundError(f"Could not find directory with 10 dog breed folders in {self.data_dir}")
        
        self._print_directory_structure(self.data_dir)
        print("Dataset structure check completed")

    def _print_directory_structure(self, startpath):
        print(f"Directory structure of {startpath}:")
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files[:5]:  # Print only first 5 files to avoid clutter
                print(f"{subindent}{f}")
            if len(files) > 5:
                print(f"{subindent}... ({len(files) - 5} more files)")

    def setup(self, stage=None):
        print("Setting up datasets...")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            print(f"Loading dataset from: {self.data_dir}")
            full_dataset = ImageFolder(self.data_dir, transform=self.transform)
            self.classes = full_dataset.classes
            print(f"Detected classes: {self.classes}")
            print(f"Dataset created. Total size: {len(full_dataset)}")
            
            # Split the dataset into train, validation, and test
            total_size = len(full_dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)  # for reproducibility
            )
            
            print(f"Dataset split. Train size: {len(self.train_dataset)}, "
                  f"Validation size: {len(self.val_dataset)}, "
                  f"Test size: {len(self.test_dataset)}")

            print(f"Number of classes in the dataset: {len(self.classes)}")
            print(f"Number of classes in the configuration: {self.config['model']['num_classes']}")
            
            assert len(self.classes) == self.config['model']['num_classes'], \
                f"Number of classes in the dataset ({len(self.classes)}) " \
                f"does not match the configuration ({self.config['model']['num_classes']})"
            print("Class number assertion passed")
        except Exception as e:
            print(f"Error in setup: {e}")
            raise

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True
        )

# For testing purposes
if __name__ == "__main__":
    data_module = DogBreedDataModule()
    data_module.prepare_data()
    data_module.setup()