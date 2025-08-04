import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Configurations for sweep ---
LEARNING_RATES = [1e-2, 1e-3, 1e-4]
EPOCHS_LIST = [25]
MODELS = {
    "efficientnet": models.efficientnet_b0,
    "resnet": models.resnet18,
    "mobilenet": models.mobilenet_v3_large,
}
BATCH_SIZE = 32
DATASET_DIR = "datavanilla"
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

os.makedirs("models", exist_ok=True)

# --- Data transforms (ImageNet normalization) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# --- Dataset and stratified split ---
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
num_classes = len(full_dataset.classes)
targets = np.array([s[1] for s in full_dataset.samples])

train_idx, temp_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=targets[temp_idx],
    random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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
    log_dir = os.path.join("runs", run_name)
    # Check for existing TensorBoard event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if os.path.exists(log_dir) and len(event_files) > 0:
        print(f"Skipping {run_name}: already completed.")
        return

    writer = SummaryWriter(log_dir=log_dir)
    model = build_model(model_name, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    best_state_dict = None

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
        val_loss, val_preds, val_targets = 0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        cm = confusion_matrix(val_targets, val_preds)

        # --- Log everything to TensorBoard ---
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
        writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
        writer.add_scalar("F1/train", train_f1, epoch + 1)
        writer.add_scalar("F1/val", val_f1, epoch + 1)

        # Log confusion matrix as image
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            fig = plot_confusion_matrix(cm, full_dataset.classes)
            writer.add_figure("Confusion_Matrix/val", fig, global_step=epoch + 1)
            plt.close(fig)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Train F1: {train_f1:.4f} "
            f"Val F1: {val_f1:.4f}"
        )

    # Save best model
    model_path = os.path.join("models", f"{run_name}.pt")
    torch.save(best_state_dict, model_path)
    print(f"Saved best model to {model_path}")

    # --- Test evaluation ---
    model.load_state_dict(best_state_dict)
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

    writer.add_scalar("Loss/test", test_loss)
    writer.add_scalar("Accuracy/test", test_acc)
    writer.add_scalar("F1/test", test_f1)
    fig = plot_confusion_matrix(cm, full_dataset.classes)
    writer.add_figure("Confusion_Matrix/test", fig)
    plt.close(fig)

    writer.add_hparams(
        {
            "model": model_name,
            "epochs": epochs,
            "lr": lr
        },
        {
            "hparam/best_val_acc": best_val_acc,
            "hparam/test_acc": test_acc,
            "hparam/test_f1": test_f1
        }
    )
    writer.close()

if __name__ == "__main__":
    for model_name in MODELS.keys():
        for epochs in EPOCHS_LIST:
            for lr in LEARNING_RATES:
                run_name = f"{model_name}_epochs{epochs}_lr{lr}"
                log_dir = os.path.join("runs", run_name)
                event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
                if os.path.exists(log_dir) and len(event_files) > 0:
                    print(f"Skipping {run_name}: already completed.")
                    continue
                print(f"Starting run: {run_name}")
                train_and_eval(model_name, epochs, lr)