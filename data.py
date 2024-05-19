import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

logger = logging.getLogger(__name__)

label_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.label_file = label_file
        self.image_files, self.labels, self.classes = self.load_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self):
        image_files = []
        labels = {}
        classes = []
        with open(self.label_file, "r") as f:
            for line in f:
                filename, label = line.strip().split(",")
                image_path = os.path.join(self.data_dir, "images", filename)
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    continue
                image_files.append(image_path)
                labels[filename] = label
                if label not in classes:
                    classes.append(label)
        return image_files, labels, classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        print(f"Processing image: {img_path}")
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Error reading image: {e}")
            return None, None
        image = self.transform(image)
        filename = os.path.basename(img_path)
        label = self.labels[filename]
        label = label_to_idx[label]  # Convert label to numerical value
        return image, label

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CustomDataset(data_dir=config.data_dir, label_file="data/dataset_files/data_registry_table.txt")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = CustomDataset(data_dir=config.data_dir, label_file="data/dataset_files/data_registry_table.txt")
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    test_dataset = CustomDataset(data_dir=config.data_dir, label_file="data/dataset_files/data_registry_table.txt")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def collate_fn(batch):
    images = []
    labels = []
    for image, label in batch:
        if image is not None and label is not None:
            images.append(image)
            labels.append(label_to_idx[label])  # Convert label to numerical value
    if not images:
        return None, None  # Return None, None for an empty batch
    
    # Convert images to tensors
    images = torch.stack(images)
    
    # Convert labels to tensors and ensure they are of type long
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels

