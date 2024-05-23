import os
import torch
import mlflow
import mlflow.pytorch
from config import Config
from data import get_data_loaders
from model import ResNet50
from utils import setup_logger, plot_training_curves
from train_utils import train_model as train_fn

def train_model(config):
    logger = setup_logger()

    train_loader, val_loader, test_loader = get_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "batch_size": config.batch_size,
            "lr": config.lr,
            "num_epochs": config.num_epochs,
            "train_val_split": config.train_val_split,
            "num_batches": config.num_batches,
            "combination_method": config.combination_method,
        })

        train_losses, val_losses, val_accuracies = train_fn(
            model, train_loader, val_loader, config.num_epochs, config.lr, device, logger
        )

        val_accuracies_tensor = torch.tensor(val_accuracies)

        model_path = os.path.join("models", "model.pth")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")
        
        plot_training_curves(train_losses, val_losses, val_accuracies_tensor)

        for epoch in range(config.num_epochs):
            mlflow.log_metric("train_loss", train_losses[epoch], step=epoch)
            mlflow.log_metric("val_loss", val_losses[epoch], step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracies[epoch], step=epoch)

        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact(config.model_path)
        mlflow.log_artifact("config.py")
        mlflow.log_artifact("train.py")
        mlflow.log_artifact("train_utils.py")
        mlflow.log_artifact("model.py")
        mlflow.log_artifact("data.py")
        mlflow.log_artifact("download_data.py")
        mlflow.log_artifact("evaluate.py")

if __name__ == "__main__":
    config = Config()
    train_model(config)
