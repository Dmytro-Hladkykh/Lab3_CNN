import os
import torch
from config import Config
from data import get_data_loaders
from model import ResNet50
from utils import setup_logger, plot_training_curves

def train_model(config):
    logger = setup_logger()

    train_loader, val_loader, test_loader = get_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)

    # Обучение модели
    from train_utils import train_model as train_fn
    train_losses, val_losses, val_accuracies = train_fn(model, train_loader, val_loader,
                                                        config.num_epochs, config.lr, device, logger)

    # Сохранение модели
    torch.save(model.state_dict(), config.model_path)

    # Отрисовка кривых обучения
    plot_training_curves(train_losses, val_losses, val_accuracies)

if __name__ == "__main__":
    config = Config()
    train_model(config)
