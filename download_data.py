import os
import shutil
from torchvision import datasets, transforms
from PIL import Image
from utils import setup_logger

logger = setup_logger()

def combine_batches_physically(batch_dir, selected_batches):
    combined_dir = "combined_data"
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(os.path.join(combined_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, "test", "images"), exist_ok=True)

    # Создаем файл меток для тестового набора
    test_labels_file = os.path.join(combined_dir, "tesdata_registry_table.txt")
    open(test_labels_file, 'w').close()  # Создаем пустой файл

    for batch_idx in selected_batches:
        batch_path = os.path.join(batch_dir, f"batch_{batch_idx}")
        for img_file in os.listdir(os.path.join(batch_path, "images")):
            src_path = os.path.join(batch_path, "images", img_file)
            dst_path = os.path.join(combined_dir, "images", img_file)
            shutil.copy(src_path, dst_path)

        with open(os.path.join(combined_dir, "data_registry_table.txt"), "a") as f:
            with open(os.path.join(batch_path, "data_registry_table.txt"), "r") as batch_labels:
                f.writelines(batch_labels.readlines())

    return combined_dir

def split_dataset_into_batches(dataset, num_batches, batch_dir):
    os.makedirs(batch_dir, exist_ok=True)

    logger.info(f"Splitting dataset into {num_batches} batches and saving to {batch_dir}")

    batch_size = len(dataset) // num_batches
    with open(os.path.join(batch_dir, "dataset_registry_table.txt"), "w") as f: 
        for i, (img, label) in enumerate(dataset):
            batch_idx = i // batch_size
            batch_path = os.path.join(batch_dir, f"batch_{batch_idx+1}")
            os.makedirs(os.path.join(batch_path, "images"), exist_ok=True)
            img_path = os.path.join(batch_path, "images", f"img_{i}.png")
            Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).save(img_path)
            f.write(f"img_{i}.png,{dataset.classes[label]}\n")  

def download_data(config, root):
    # Загрузка и сохранение датасета в dataset_files
    cifar10 = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
    dataset_files_dir = os.path.join(root, "dataset_files")
    os.makedirs(dataset_files_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_files_dir, "images"), exist_ok=True) # Создаем папку images

    for i, (img, label) in enumerate(cifar10):
        img_path = os.path.join(dataset_files_dir, "images", f"img_{i}.png") # Сохраняем в images
        Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).save(img_path)
        with open(os.path.join(dataset_files_dir, "data_registry_table.txt"), "a") as f:
            f.write(f"img_{i}.png,{cifar10.classes[label]}\n")

    # Разделение датасета на батчи
    batch_dir = os.path.join(root, "batches")
    split_dataset_into_batches(cifar10, config.num_batches, batch_dir)

if __name__ == "__main__":
    from config import Config
    config = Config()
    download_data(config, "data")