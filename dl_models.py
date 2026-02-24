"""
MediScan Analyst — Deep Learning Model Manager
ONNX Runtime-based inference for medical image classification.
Uses models exported by train_cnn.py and export_pretrained.py.

Models:
  - Custom CNN models (trained on MedMNIST public datasets)
  - Pre-trained DenseNet-121 (CheXpert/NIH/MIMIC/PadChest)
"""

import os
import json
import numpy as np
import cv2
import logging
from enum import Enum

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class ModelStatus(str, Enum):
    NOT_FOUND = "not_found"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


# ============================================================================
# CHEST X-RAY PATHOLOGY LABELS (from TorchXRayVision DenseNet)
# ============================================================================

CHESTXRAY_PATHOLOGIES = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

# ============================================================================
# MEDMNIST DATASET INFO
# ============================================================================

MEDMNIST_MODELS = {
    "pneumoniamnist": {
        "description": "Chest X-Ray Pneumonia Detection",
        "labels": {"0": "Normal", "1": "Pneumonia"},
        "in_channels": 1,
        "body_parts": ["chest"],
        "relevance": "primary",
    },
    "organamnist": {
        "description": "Abdominal CT Organ Classification",
        "labels": {
            "0": "Bladder", "1": "Femur-L", "2": "Femur-R",
            "3": "Heart", "4": "Kidney-L", "5": "Kidney-R",
            "6": "Liver", "7": "Lung-L", "8": "Lung-R",
            "9": "Spleen", "10": "Vertebrae",
        },
        "in_channels": 1,
        "body_parts": ["general", "chest", "spine"],
        "relevance": "secondary",
    },
    "pathmnist": {
        "description": "Colon Pathology Tissue Classification",
        "labels": {
            "0": "Adipose", "1": "Background", "2": "Debris",
            "3": "Lymphocytes", "4": "Mucus", "5": "Smooth Muscle",
            "6": "Normal Colon Mucosa", "7": "Cancer Stroma",
            "8": "Colorectal Adenocarcinoma",
        },
        "in_channels": 3,
        "body_parts": ["general"],
        "relevance": "secondary",
    },
    "dermamnist": {
        "description": "Dermatoscopy Skin Lesion Classification",
        "labels": {
            "0": "Actinic Keratoses", "1": "Basal Cell Carcinoma",
            "2": "Benign Keratosis", "3": "Dermatofibroma",
            "4": "Melanoma", "5": "Melanocytic Nevi", "6": "Vascular Lesions",
        },
        "in_channels": 3,
        "body_parts": ["general"],
        "relevance": "secondary",
    },
    "octmnist": {
        "description": "Retinal OCT Classification",
        "labels": {
            "0": "Choroidal Neovascularization", "1": "Diabetic Macular Edema",
            "2": "Drusen", "3": "Normal",
        },
        "in_channels": 1,
        "body_parts": ["general"],
        "relevance": "secondary",
    },
    "breastmnist": {
        "description": "Breast Ultrasound Classification",
        "labels": {"0": "Malignant", "1": "Normal/Benign"},
        "in_channels": 1,
        "body_parts": ["general"],
        "relevance": "secondary",
    },
}


class ModelManager:
    """
    Manages ONNX model loading and inference for medical image analysis.
    Thread-safe, lazy-loading, with graceful fallback.
    """

    def __init__(self):
        self.sessions = {}        # name → ort.InferenceSession
        self.statuses = {}        # name → ModelStatus
        self.metadata = {}        # name → dict
        self._ort_available = False
        
        # Check ONNX Runtime availability
        try:
            import onnxruntime
            self._ort_available = True
            logger.info("✅ ONNX Runtime available")
        except ImportError:
            logger.warning("⚠️ ONNX Runtime not installed. DL features disabled.")
        
        # Scan for available models
        self._scan_models()

    def _scan_models(self):
        """Scan the models directory for available ONNX models."""
        if not os.path.exists(MODELS_DIR):
            logger.info(f"📁 Models directory not found: {MODELS_DIR}")
            return
        
        # Check for chest X-ray DenseNet
        densenet_path = os.path.join(MODELS_DIR, "chestxray_densenet.onnx")
        if os.path.exists(densenet_path):
            self.statuses["chestxray_densenet"] = ModelStatus.NOT_FOUND  # Will load lazily
            meta_path = os.path.join(MODELS_DIR, "chestxray_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.metadata["chestxray_densenet"] = json.load(f)
            self.statuses["chestxray_densenet"] = ModelStatus.READY  # Mark as available
            logger.info("✅ ChestXRay DenseNet model found")
        
        # Check for MedMNIST CNN models
        for name in MEDMNIST_MODELS:
            onnx_path = os.path.join(MODELS_DIR, f"{name}_cnn.onnx")
            if os.path.exists(onnx_path):
                self.statuses[name] = ModelStatus.READY
                logger.info(f"✅ {name} CNN model found")
            else:
                self.statuses[name] = ModelStatus.NOT_FOUND
        
        # Load training metadata
        train_meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
        if os.path.exists(train_meta_path):
            with open(train_meta_path) as f:
                self.metadata["training"] = json.load(f)

    def _load_session(self, model_name):
        """Lazily load an ONNX model session."""
        if not self._ort_available:
            return None
        
        if model_name in self.sessions:
            return self.sessions[model_name]
        
        import onnxruntime as ort
        
        # Determine ONNX file path
        if model_name == "chestxray_densenet":
            onnx_path = os.path.join(MODELS_DIR, "chestxray_densenet.onnx")
        else:
            onnx_path = os.path.join(MODELS_DIR, f"{model_name}_cnn.onnx")
        
        if not os.path.exists(onnx_path):
            self.statuses[model_name] = ModelStatus.NOT_FOUND
            return None
        
        try:
            self.statuses[model_name] = ModelStatus.LOADING
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.sessions[model_name] = session
            self.statuses[model_name] = ModelStatus.READY
            logger.info(f"✅ Loaded {model_name} ONNX model")
            return session
        except Exception as e:
            self.statuses[model_name] = ModelStatus.ERROR
            logger.error(f"❌ Failed to load {model_name}: {e}")
            return None

    # ========================================================================
    # PREPROCESSING
    # ========================================================================

    def _preprocess_for_chestxray(self, image_array):
        """
        Preprocess image for TorchXRayVision DenseNet.
        Input: numpy array (H, W, 3) uint8
        Output: numpy array (1, 1, 224, 224) float32, normalized to [-1024, 1024]
        """
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Resize to 224x224
        resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Normalize to [-1024, 1024] (TorchXRayVision convention)
        normalized = resized.astype(np.float32)
        normalized = (normalized / 255.0) * 2048.0 - 1024.0
        
        # Add batch and channel dimensions: (1, 1, 224, 224)
        tensor = normalized[np.newaxis, np.newaxis, :, :]
        
        return tensor

    def _preprocess_for_medmnist(self, image_array, in_channels):
        """
        Preprocess image for custom MedMNIST CNN.
        Input: numpy array (H, W, 3) uint8
        Output: numpy array (1, C, 28, 28) float32, normalized to [-1, 1]
        """
        if in_channels == 1:
            if len(image_array.shape) == 3:
                img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                img = image_array
            resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            tensor = normalized[np.newaxis, np.newaxis, :, :]
        else:
            if len(image_array.shape) == 2:
                img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                img = image_array
            resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            # HWC → CHW
            tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, :, :, :]
        
        return tensor

    # ========================================================================
    # INFERENCE
    # ========================================================================

    def predict_chest_pathology(self, image_array):
        """
        Run chest X-ray pathology detection using DenseNet-121.
        Returns list of {pathology, probability, model} sorted by probability.
        """
        session = self._load_session("chestxray_densenet")
        if session is None:
            return None
        
        try:
            input_tensor = self._preprocess_for_chestxray(image_array)
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: input_tensor})
            
            # Apply sigmoid to get probabilities
            logits = result[0][0]
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            
            predictions = []
            for i, (label, prob) in enumerate(zip(CHESTXRAY_PATHOLOGIES, probabilities)):
                predictions.append({
                    "pathology": label,
                    "probability": float(prob),
                    "model": "DenseNet-121 (CheXpert/NIH/MIMIC/PadChest)",
                    "model_type": "pre-trained",
                })
            
            # Sort by probability descending
            predictions.sort(key=lambda x: x["probability"], reverse=True)
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Chest pathology prediction failed: {e}")
            return None

    def predict_medmnist(self, image_array, model_name):
        """
        Run a MedMNIST CNN model for classification.
        Returns list of {label, probability, model} sorted by probability.
        """
        if model_name not in MEDMNIST_MODELS:
            return None
        
        session = self._load_session(model_name)
        if session is None:
            return None
        
        config = MEDMNIST_MODELS[model_name]
        
        try:
            input_tensor = self._preprocess_for_medmnist(
                image_array, config["in_channels"]
            )
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: input_tensor})
            
            # Apply softmax
            logits = result[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()
            
            predictions = []
            for idx, prob in enumerate(probabilities):
                label = config["labels"].get(str(idx), f"Class {idx}")
                predictions.append({
                    "label": label,
                    "probability": float(prob),
                    "model": f"MediScanCNN ({config['description']})",
                    "model_type": "custom-trained",
                    "dataset": model_name,
                })
            
            predictions.sort(key=lambda x: x["probability"], reverse=True)
            return predictions
            
        except Exception as e:
            logger.error(f"❌ {model_name} prediction failed: {e}")
            return None

    def run_all_relevant(self, image_array, body_part="general"):
        """
        Run all relevant DL models for a given image and body part.
        Returns a comprehensive prediction dict.
        """
        results = {
            "dl_available": self._ort_available,
            "chest_pathology": None,
            "medmnist_predictions": {},
            "models_used": [],
            "models_status": self.get_status(),
        }
        
        if not self._ort_available:
            return results
        
        # Always try chest pathology if model is available
        if body_part in ("chest", "general"):
            chest_preds = self.predict_chest_pathology(image_array)
            if chest_preds:
                results["chest_pathology"] = chest_preds
                results["models_used"].append("DenseNet-121 (Pre-trained)")
        
        # Run relevant MedMNIST models
        for model_name, config in MEDMNIST_MODELS.items():
            if self.statuses.get(model_name) != ModelStatus.READY:
                continue
            
            # Check if model is relevant for this body part
            if body_part in config["body_parts"] or config["relevance"] == "primary":
                preds = self.predict_medmnist(image_array, model_name)
                if preds:
                    results["medmnist_predictions"][model_name] = {
                        "description": config["description"],
                        "predictions": preds,
                    }
                    results["models_used"].append(f"MediScanCNN ({config['description']})")
        
        return results

    def get_status(self):
        """Get status of all models."""
        status = {
            "onnx_runtime": self._ort_available,
            "models_dir": MODELS_DIR,
            "models": {},
        }
        
        # ChestXRay DenseNet
        status["models"]["chestxray_densenet"] = {
            "status": self.statuses.get("chestxray_densenet", ModelStatus.NOT_FOUND),
            "type": "Pre-trained DenseNet-121",
            "source": "TorchXRayVision (CheXpert/NIH/MIMIC/PadChest)",
            "pathologies": len(CHESTXRAY_PATHOLOGIES),
        }
        
        # MedMNIST models
        for name, config in MEDMNIST_MODELS.items():
            status["models"][name] = {
                "status": self.statuses.get(name, ModelStatus.NOT_FOUND),
                "type": "Custom CNN (MediScanCNN)",
                "description": config["description"],
                "classes": len(config["labels"]),
            }
        
        # Count ready models
        ready = sum(1 for s in self.statuses.values() if s == ModelStatus.READY)
        status["ready_count"] = ready
        status["total_count"] = len(MEDMNIST_MODELS) + 1  # +1 for DenseNet
        
        return status


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

model_manager = ModelManager()
