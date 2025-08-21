import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, multilabel_confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_data import test_df, mlb, image_path_mapping
from train import ChestXrayDataset, val_test_transforms
from torch.utils.data import DataLoader


def evaluate_model(model, dataloader, classes, device):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    all_labels = []
    all_outputs = []

    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx * dataloader.batch_size} samples...")

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)

    # Per-class ROC-AUC
    roc_aucs = {}
    for i, c in enumerate(classes):
        try:
            roc_aucs[c] = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        except ValueError:
            roc_aucs[c] = np.nan  # If a class has only one label (all 0 or all 1)

    mean_auc = np.nanmean(list(roc_aucs.values()))

    print("\nROC-AUC Scores per Class:")
    for c, score in roc_aucs.items():
        if not np.isnan(score):
            print(f"{c:20s}: {score:.4f}")
        else:
            print(f"{c:20s}: N/A (insufficient data)")

    print(f"\nMean ROC-AUC: {mean_auc:.4f}")

    # Threshold outputs at 0.5 for accuracy/F1
    preds = (all_outputs > 0.5).astype(int)
    acc = accuracy_score(all_labels.flatten(), preds.flatten())
    f1 = f1_score(all_labels.flatten(), preds.flatten(), average="macro")

    print(f"\nOverall Accuracy: {acc:.4f}")
    print(f"\nMacro F1-Score: {f1:.4f}")

    return roc_aucs, mean_auc, all_labels, all_outputs


def load_trained_model(model_path, num_classes, device):
    """Load the trained DenseNet169 model"""
    print(f"Loading model from {model_path}...")

    model = models.densenet169(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def classification_summary(model, dataloader, classes, device, threshold=0.5):
    """
    Prints Precision, Recall, and F1 per disease.
    """
    model.eval()
    all_labels = []
    all_outputs = []

    print(f"\nGenerating Classification Report (threshold={threshold})...")
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > threshold).int()

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(preds.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)

    print("\nDetailed Classification Report:")
    print("=" * 80)
    print(classification_report(all_labels, all_outputs, target_names=classes, zero_division=0))


def plot_roc_curves(y_true, y_pred, classes, save_path="roc_curves.png"):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(15, 12))

    for i, class_name in enumerate(classes):
        if np.sum(y_true[:, i]) > 0:  # Only plot if class has positive samples
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc_score = roc_auc_score(y_true[:, i], y_pred[:, i])

            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - ChestX-ray14 Classification')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.show()


def plot_confusion_matrices(y_true, y_pred, classes, max_classes=6, save_path="confusion_matrices.png"):
    """
    Plots confusion matrices for up to `max_classes` diseases.
    Each matrix is binary: disease vs. not-disease.
    """
    # Convert predictions to binary (threshold = 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Get multilabel confusion matrices
    cms = multilabel_confusion_matrix(y_true, y_pred_binary)

    # Select top classes by frequency for visualization
    class_frequencies = np.sum(y_true, axis=0)
    top_indices = np.argsort(class_frequencies)[::-1][:max_classes]

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(top_indices):
        cm = cms[idx]
        class_name = classes[idx]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i])
        axes[i].set_title(f"{class_name}\n(n={int(class_frequencies[idx])} positive cases)")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].set_xticklabels(['No Disease', 'Disease'])
        axes[i].set_yticklabels(['No Disease', 'Disease'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {save_path}")
    plt.show()


def analyze_predictions(y_true, y_pred, classes, threshold=0.5):
    """Analyze prediction patterns"""
    preds_binary = (y_pred > threshold).astype(int)

    print(f"\n🔍 Prediction Analysis (threshold={threshold}):")
    print("=" * 50)

    for i, class_name in enumerate(classes):
        true_positives = np.sum((y_true[:, i] == 1) & (preds_binary[:, i] == 1))
        false_positives = np.sum((y_true[:, i] == 0) & (preds_binary[:, i] == 1))
        true_negatives = np.sum((y_true[:, i] == 0) & (preds_binary[:, i] == 0))
        false_negatives = np.sum((y_true[:, i] == 1) & (preds_binary[:, i] == 0))

        if true_positives + false_negatives > 0:
            sensitivity = true_positives / (true_positives + false_negatives)
        else:
            sensitivity = 0.0

        if true_negatives + false_positives > 0:
            specificity = true_negatives / (true_negatives + false_positives)
        else:
            specificity = 0.0

        print(f"{class_name:20s}: Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}")


def plot_class_distribution(y_true, classes, save_path="class_distribution.png"):
    """Plot distribution of positive cases for each class"""
    class_counts = np.sum(y_true, axis=0)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(classes)), class_counts, color='steelblue', alpha=0.7)
    plt.xlabel('Disease Classes')
    plt.ylabel('Number of Positive Cases')
    plt.title('Distribution of Diseases in Test Set')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')

    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f'{int(count)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    MODEL_PATH = "densenet169_chestxray14.pth"
    model = load_trained_model(MODEL_PATH, len(mlb.classes_), device)

    # Prepare test dataset
    print(f"\nPreparing test dataset ({len(test_df)} samples)...")
    test_dataset = ChestXrayDataset(test_df, image_path_mapping, mlb, transform=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Evaluate model
    roc_aucs, mean_auc, y_true, y_pred = evaluate_model(model, test_loader, mlb.classes_, device)

    # Detailed analysis
    analyze_predictions(y_true, y_pred, mlb.classes_)

    # Generate classification report with precision, recall, F1 per disease
    classification_summary(model, test_loader, mlb.classes_, device)

    # Class distribution
    plot_class_distribution(y_true, mlb.classes_)

    # ROC curves
    plot_roc_curves(y_true, y_pred, mlb.classes_)

    # Confusion matrices for top 6 most frequent diseases
    plot_confusion_matrices(y_true, y_pred, mlb.classes_, max_classes=6)

    print(f"\nFinal Results:")
    print(f"\nMean ROC-AUC: {mean_auc:.4f}")
    print(f"\nBest performing classes: {sorted(roc_aucs.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)[:3]}")
    print(f"\nModel evaluation completed successfully!")