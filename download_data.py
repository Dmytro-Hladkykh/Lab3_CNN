import os
import shutil
from torchvision import datasets, transforms
from PIL import Image
from utils import setup_logger

logger = setup_logger()

def split_dataset_into_batches(dataset, num_batches, batch_dir):
    os.makedirs(batch_dir, exist_ok=True)

    logger.info(f"Splitting dataset into {num_batches} batches and saving to {batch_dir}")

    batch_size = len(dataset) // num_batches
    for batch_idx in range(1, num_batches + 1):
        batch_path = os.path.join(batch_dir, f"batch_{batch_idx}")
        os.makedirs(os.path.join(batch_path, "images"), exist_ok=True)
        with open(os.path.join(batch_path, "batch_registry_table.txt"), "w") as f:
            for i, (img, label) in enumerate(dataset):
                if batch_idx * batch_size - batch_size <= i < batch_idx * batch_size:
                    img_path = os.path.join(batch_path, "images", f"img_{i}.png")
                    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).save(img_path)
                    f.write(f"img_{i}.png,{dataset.classes[label]}\n")

def download_data(config, root):
    cifar10 = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())

    batch_dir = os.path.join(root, "batches")
    split_dataset_into_batches(cifar10, config.num_batches, batch_dir)

if __name__ == "__main__":
    from config import Config
    config = Config()
    download_data(config, "data")