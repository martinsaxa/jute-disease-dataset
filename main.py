import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# --- Configurations for sweep ---
EPOCHS_LIST = [25]
MODELS = {
    "efficientnet": models.efficientnet_b0,
    "resnet": models.resnet18,
    "mobilenet": models.mobilenet_v3_large,
}
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"


# --- Data transforms (ImageNet normalization) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# --- Datasets and loaders ---
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
num_classes = len(train_dataset.classes)

# --- Custom classifier head (deeper) ---
class DeeperHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(x)

# --- Model builder ---
def build_model(model_name, num_classes):
    if model_name == "efficientnet":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        model = nn.Sequential(
            base,
            DeeperHead(in_features, num_classes)
        )
    elif model_name == "resnet":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        model = nn.Sequential(
            base,
            DeeperHead(in_features, num_classes)
        )
    elif model_name == "mobilenet":
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = base.classifier[0].in_features
        base.classifier = nn.Identity()
        model = nn.Sequential(
            base,
            DeeperHead(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# --- Confusion matrix plot for TensorBoard ---
def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, f"{cm[i, j]:.2f}", horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

# --- Training and evaluation ---
def train_and_eval(model_name, epochs, lr):
    run_name = f"{model_name}_epochs{epochs}_lr{lr}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    model = build_model(model_name, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0
    best_test_f1 = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_targets = 0, [], []
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro")

        # --- Validation ---
        model.eval()
        test_loss, test_preds, test_targets = 0, [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1).cpu().numpy()
                test_preds.extend(preds)
                test_targets.extend(labels.cpu().numpy())
        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds, average="macro")
        cm = confusion_matrix(test_targets, test_preds)

        # --- Log everything to TensorBoard ---
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/test", test_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
        writer.add_scalar("Accuracy/test", test_acc, epoch + 1)
        writer.add_scalar("F1/train", train_f1, epoch + 1)
        writer.add_scalar("F1/test", test_f1, epoch + 1)

        # Log confusion matrix as image
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            fig = plot_confusion_matrix(cm, train_dataset.classes)
            writer.add_figure("Confusion_Matrix", fig, global_step=epoch + 1)
            plt.close(fig)

        # Save best metrics
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Test Acc: {test_acc:.4f} "
            f"Train F1: {train_f1:.4f} "
            f"Test F1: {test_f1:.4f}"
        )

    writer.add_hparams(
        {
            "model": model_name,
            "epochs": epochs,
            "lr": lr
        },
        {
            "hparam/best_test_acc": best_test_acc,
            "hparam/best_test_f1": best_test_f1
        }
    )
    writer.close()

if __name__ == "__main__":
    for model_name in MODELS.keys():
        for epochs in EPOCHS_LIST:
            run_name = f"{model_name}_epochs{epochs}_lr{LEARNING_RATE}"
            log_dir = os.path.join("runs", run_name)
            # Check for existing TensorBoard event files
            event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
            if os.path.exists(log_dir) and len(event_files) > 0:
                print(f"Skipping {run_name}: already completed.")
                continue
            train_and_eval(model_name, epochs, LEARNING_RATE)