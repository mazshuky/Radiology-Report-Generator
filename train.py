import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from preprocess_data import train_val_df, test_df, mlb, image_path_mapping
from tqdm import tqdm


# === 1. Dataset Class ===
class ChestXrayDataset(Dataset):
    def __init__(self, df, image_mapping, mlb, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_mapping = image_mapping
        self.mlb = mlb
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_filename = row["Image Index"]

        # Get the full path from the mapping
        if image_filename not in self.image_mapping:
            print(f"Warning: Image {image_filename} not found in mapping")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            img_path = self.image_mapping[image_filename]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new("RGB", (224, 224), (0, 0, 0))

        labels = row["Finding Labels"]
        if isinstance(labels, list):
            pass  # Already split in preprocessing
        elif isinstance(labels, str):
            labels = labels.split("|")
        else:
            labels = []

        y = self.mlb.transform([labels])[0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(y, dtype=torch.float32)



# === 2. Transforms ===
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



# === 3. Training and Evaluation Functions ===
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # Add progress bar
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        # Ensure tensors before moving to device
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Print every 100 batches
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = nn.BCELoss()(outputs, labels)
            val_loss += loss.item()

            y_true.append(labels.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    try:
        auroc = roc_auc_score(y_true, y_pred, average="macro")
    except:
        auroc = 0.0

    return val_loss / len(loader), auroc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auroc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}")


if __name__ == "__main__":
    # == 4. Data Preparation ===
    print("Preparing data...")
    print(f"Train/Val samples: {len(train_val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Available images: {len(image_path_mapping)}")
    print(f"Classes: {mlb.classes_}")

    # Train / validation split
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=None)
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # === 5. DataLoaders ===
    BATCH_SIZE = 16

    train_dataset = ChestXrayDataset(train_df, image_path_mapping, mlb, transform=train_transforms)
    val_dataset = ChestXrayDataset(val_df, image_path_mapping, mlb, transform=val_test_transforms)
    test_dataset = ChestXrayDataset(test_df, image_path_mapping, mlb, transform=val_test_transforms)

    # Set num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    # === 6. Model Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Install CUDA-enabled PyTorch for GPU acceleration.")

    # Model (DenseNet-169)
    model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, len(mlb.classes_)),
        nn.Sigmoid()
    )
    model = model.to(device)


    # === 7. Loss & Optimizer ===
    criterion = nn.BCELoss()  # binary cross-entropy for multi-label
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # === 8. Training ===
    EPOCHS = 5
    print(f"\nStarting training for {EPOCHS} epochs...")

    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS)


    # === 9. Final Test Evaluation ===
    print("\nEvaluating on test set...")
    test_loss, test_auroc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test AUROC: {test_auroc:.4f}")


    # === 10. Save Model ===
    torch.save(model.state_dict(), "densenet169_chestxray14.pth")
    print("Model saved successfully!")