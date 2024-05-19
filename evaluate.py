import os
import torch
from config import Config
from data import get_data_loaders, get_class_labels
from model import ResNet50

def evaluate_model(config):
    _, _, test_loader = get_data_loaders(config, "data/dataset_files")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(config.model_path))

    from train_utils import validate_model
    test_loss, test_acc = validate_model(model, test_loader, criterion=None, device=device)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    config = Config()
    evaluate_model(config)