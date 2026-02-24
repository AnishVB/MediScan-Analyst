"""
MediScan Analyst — Custom CNN Training Script
Trains a lightweight CNN on MedMNIST public datasets for medical image classification.
Exports the trained model to ONNX format for deployment.

Usage:
    pip install -r requirements-dev.txt
    python train_cnn.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import json
from datetime import datetime

# ============================================================================
# CUSTOM CNN ARCHITECTURE
# ============================================================================

class MediScanCNN(nn.Module):
    """
    Lightweight 6-layer CNN for medical image classification.
    Designed for body-part and modality classification from MedMNIST.
    
    Architecture:
        Conv2d(1,32) → BN → ReLU → MaxPool
        Conv2d(32,64) → BN → ReLU → MaxPool
        Conv2d(64,128) → BN → ReLU → AdaptiveAvgPool
        FC(128,256) → Dropout → ReLU
        FC(256,128) → Dropout → ReLU
        FC(128, num_classes)
    """
    
    def __init__(self, in_channels=1, num_classes=11):
        super(MediScanCNN, self).__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 28x28 → 14x14
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 14x14 → 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 7x7 → 1x1 (adaptive)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# MEDMNIST DATASET CONFIGURATION
# ============================================================================

# We train on multiple MedMNIST datasets for broad medical image understanding
DATASETS = {
    "pathmnist": {
        "description": "Colon Pathology (9 tissue types)",
        "in_channels": 3,
        "task": "multi-class",
    },
    "dermamnist": {
        "description": "Dermatoscopy (7 skin lesion types)", 
        "in_channels": 3,
        "task": "multi-class",
    },
    "octmnist": {
        "description": "Retinal OCT (4 conditions)",
        "in_channels": 1,
        "task": "multi-class",
    },
    "pneumoniamnist": {
        "description": "Chest X-Ray Pneumonia (2 classes)",
        "in_channels": 1,
        "task": "binary-class",
    },
    "breastmnist": {
        "description": "Breast Ultrasound (2 classes)",
        "in_channels": 1,
        "task": "binary-class",
    },
    "organamnist": {
        "description": "Abdominal CT Organ (11 organs)",
        "in_channels": 1,
        "task": "multi-class",
    },
}


def train_single_dataset(dataset_name, config, epochs=20, batch_size=64, device="cpu"):
    """Train the custom CNN on a single MedMNIST dataset."""
    import medmnist
    from medmnist import INFO
    
    info = INFO[dataset_name]
    n_classes = len(info["label"])
    in_channels = config["in_channels"]
    
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name}: {config['description']}")
    print(f"Classes: {n_classes}, Channels: {in_channels}")
    print(f"{'='*60}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels),
    ])
    
    # Load dataset (auto-downloads from public servers)
    DataClass = getattr(medmnist, info["python_class"])
    train_dataset = DataClass(split="train", transform=transform, download=True, size=28)
    val_dataset = DataClass(split="val", transform=transform, download=True, size=28)
    test_dataset = DataClass(split="test", transform=transform, download=True, size=28)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize model
    model = MediScanCNN(in_channels=in_channels, num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
    
    # Test with best model
    model.load_state_dict(best_model_state)
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total
    print(f"\n  ✅ Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    return model, {
        "dataset": dataset_name,
        "description": config["description"],
        "in_channels": in_channels,
        "num_classes": n_classes,
        "class_labels": info["label"],
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "epochs_trained": epochs,
    }


def export_to_onnx(model, metadata, dataset_name, in_channels, models_dir):
    """Export a trained PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, in_channels, 28, 28)
    onnx_path = os.path.join(models_dir, f"{dataset_name}_cnn.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  📦 Exported to {onnx_path} ({file_size:.2f} MB)")
    
    metadata["onnx_path"] = onnx_path
    metadata["onnx_size_mb"] = round(file_size, 2)
    return metadata


def main():
    print("=" * 60)
    print("MediScan Analyst — Custom CNN Training Pipeline")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    all_metadata = {}
    
    # Train on each dataset
    for dataset_name, config in DATASETS.items():
        try:
            model, metadata = train_single_dataset(
                dataset_name, config,
                epochs=20, batch_size=64, device=device,
            )
            metadata = export_to_onnx(
                model, metadata, dataset_name,
                config["in_channels"], models_dir,
            )
            all_metadata[dataset_name] = metadata
        except Exception as e:
            print(f"  ❌ Failed to train {dataset_name}: {e}")
            continue
    
    # Save training metadata
    meta_path = os.path.join(models_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "device": device,
            "framework": "PyTorch → ONNX",
            "architecture": "MediScanCNN (6-layer custom CNN)",
            "datasets_source": "MedMNIST (public medical imaging benchmark)",
            "models": all_metadata,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete! Metadata saved to {meta_path}")
    print(f"Models saved to {models_dir}/")
    print(f"{'='*60}")
    
    # Summary
    print("\n📊 Training Summary:")
    for name, meta in all_metadata.items():
        print(f"  {name:20s} | Test Acc: {meta['test_accuracy']:.4f} | Size: {meta['onnx_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
