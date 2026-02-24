"""
MediScan Analyst — Pre-trained Model Export Script
Downloads TorchXRayVision's DenseNet-121 (pre-trained on CheXpert, NIH, MIMIC, PadChest)
and exports it to ONNX format for lightweight deployment.

Usage:
    pip install -r requirements-dev.txt
    python export_pretrained.py
"""

import os
import torch
import numpy as np
import json
from datetime import datetime


def export_chestxray_densenet():
    """Export TorchXRayVision DenseNet-121 to ONNX."""
    import torchxrayvision as xrv
    
    print("=" * 60)
    print("Exporting TorchXRayVision DenseNet-121")
    print("=" * 60)
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Load pre-trained DenseNet-121 (trained on multiple public chest X-ray databases)
    # Weights source: CheXpert, NIH ChestX-ray14, MIMIC-CXR, PadChest
    print("📥 Downloading pre-trained DenseNet-121 weights...")
    print("   Trained on: CheXpert (Stanford), NIH ChestX-ray14, MIMIC-CXR (MIT), PadChest")
    
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    
    # TorchXRayVision pathology labels (18 conditions)
    pathology_labels = list(model.pathologies)
    print(f"\n   Pathologies detected ({len(pathology_labels)}):")
    for i, label in enumerate(pathology_labels):
        print(f"     {i+1:2d}. {label}")
    
    # Export to ONNX
    # TorchXRayVision expects 224x224 single-channel input, normalized to [-1024, 1024]
    dummy_input = torch.randn(1, 1, 224, 224)
    onnx_path = os.path.join(models_dir, "chestxray_densenet.onnx")
    
    print(f"\n📦 Exporting to ONNX...")
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
    print(f"   ✅ Exported to {onnx_path} ({file_size:.2f} MB)")
    
    # Save model metadata
    metadata = {
        "model_name": "DenseNet-121",
        "source": "TorchXRayVision",
        "weights": "densenet121-res224-all",
        "input_shape": [1, 1, 224, 224],
        "input_format": "Single-channel grayscale, 224x224, normalized to [-1024, 1024]",
        "pathology_labels": pathology_labels,
        "num_pathologies": len(pathology_labels),
        "training_databases": [
            "CheXpert (Stanford, 224k chest X-rays)",
            "NIH ChestX-ray14 (112k chest X-rays)",
            "MIMIC-CXR (MIT/Beth Israel, 377k chest X-rays)",
            "PadChest (Univ. of Alicante, 160k chest X-rays)",
        ],
        "total_training_images": "~873,000 chest X-rays",
        "onnx_size_mb": round(file_size, 2),
        "exported_at": datetime.now().isoformat(),
    }
    
    meta_path = os.path.join(models_dir, "chestxray_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   📋 Metadata saved to {meta_path}")
    
    # Verify the exported model works
    print(f"\n🔬 Verifying ONNX model...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 1, 224, 224).astype(np.float32)
    result = session.run(None, {"input": test_input})
    
    print(f"   Input shape:  {test_input.shape}")
    print(f"   Output shape: {result[0].shape}")
    print(f"   Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")
    print(f"   ✅ ONNX model verification passed!")
    
    return metadata


def main():
    print("=" * 60)
    print("MediScan Analyst — Pre-trained Model Export")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    metadata = export_chestxray_densenet()
    
    print(f"\n{'='*60}")
    print("Export complete!")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Pathologies: {metadata['num_pathologies']}")
    print(f"  Trained on: {metadata['total_training_images']}")
    print(f"  ONNX size: {metadata['onnx_size_mb']} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
