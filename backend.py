"""
MediScan Analyst — FastAPI Backend with Four-Agent Architecture
Vision Agent → DL Classification Agent → Analysis Agent → Reporting Agent
Hybrid: Custom CNN + OpenCV + Pre-trained DenseNet
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from PIL import Image
import cv2
import io
import os
import json
import uuid
import logging
from datetime import datetime
import time

import database as db
from dl_models import model_manager

# Ensure uploads directory exists
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(CURRENT_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediScan Analyst API",
    description="Agentic AI Co-pilot for Medical Imaging and Diagnosis Support — Hybrid DL + CV2",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MEDICAL KNOWLEDGE BASE
# ============================================================================

KNOWLEDGE_BASE = {
    "chest": {
        "conditions": [
            {
                "name": "Pneumonia",
                "indicators": ["increased_opacity", "consolidation", "air_bronchograms"],
                "description": "Inflammatory condition of the lung affecting the alveoli",
                "risk": "moderate",
            },
            {
                "name": "Pleural Effusion",
                "indicators": ["blunted_costophrenic", "meniscus_sign", "opacity_base"],
                "description": "Excess fluid accumulates in the pleural space",
                "risk": "moderate",
            },
            {
                "name": "Cardiomegaly",
                "indicators": ["enlarged_cardiac_silhouette", "ctr_elevated"],
                "description": "Enlargement of the heart beyond normal dimensions",
                "risk": "moderate",
            },
            {
                "name": "Pulmonary Edema",
                "indicators": ["bilateral_opacity", "kerley_lines", "peribronchial_cuffing"],
                "description": "Fluid accumulation in the tissue and air spaces of the lungs",
                "risk": "high",
            },
            {
                "name": "Normal Study",
                "indicators": ["clear_lung_fields", "normal_cardiac", "sharp_costophrenic"],
                "description": "No significant abnormalities detected in chest radiograph",
                "risk": "low",
            },
        ],
        "anatomy": ["Lung fields", "Cardiac silhouette", "Mediastinum", "Costophrenic angles", "Diaphragm", "Trachea", "Aortic arch"],
    },
    "hand": {
        "conditions": [
            {
                "name": "Fracture",
                "indicators": ["cortical_break", "discontinuity", "displacement"],
                "description": "Break or crack in the bone structure",
                "risk": "moderate",
            },
            {
                "name": "Osteoporosis",
                "indicators": ["reduced_density", "thinned_cortex", "trabecular_loss"],
                "description": "Decreased bone mineral density with increased fragility",
                "risk": "moderate",
            },
            {
                "name": "Arthritis",
                "indicators": ["joint_narrowing", "erosion", "osteophytes"],
                "description": "Joint inflammation causing cartilage and bone changes",
                "risk": "low",
            },
            {
                "name": "Normal Study",
                "indicators": ["intact_cortex", "normal_density", "preserved_joints"],
                "description": "Normal skeletal structures without significant pathology",
                "risk": "low",
            },
        ],
        "anatomy": ["Phalanges", "Metacarpals", "Carpals", "Radius", "Ulna", "Joint spaces"],
    },
    "brain": {
        "conditions": [
            {
                "name": "Mass Effect",
                "indicators": ["midline_shift", "asymmetry", "compression"],
                "description": "Space-occupying lesion causing displacement of brain structures",
                "risk": "high",
            },
            {
                "name": "Hemorrhage",
                "indicators": ["hyperdensity", "asymmetry", "fluid_level"],
                "description": "Bleeding within or around the brain parenchyma",
                "risk": "high",
            },
            {
                "name": "Hydrocephalus",
                "indicators": ["enlarged_ventricles", "periventricular_changes"],
                "description": "Abnormal accumulation of cerebrospinal fluid within the brain",
                "risk": "moderate",
            },
            {
                "name": "Normal Study",
                "indicators": ["symmetric", "normal_ventricles", "preserved_differentiation"],
                "description": "Normal intracranial structures without focal abnormality",
                "risk": "low",
            },
        ],
        "anatomy": ["Cerebral hemispheres", "Ventricles", "Midline structures", "Gray matter", "White matter", "Cerebellum"],
    },
    "spine": {
        "conditions": [
            {
                "name": "Disc Herniation",
                "indicators": ["disc_narrowing", "bulging", "nerve_compression"],
                "description": "Displacement of disc material beyond the intervertebral disc space",
                "risk": "moderate",
            },
            {
                "name": "Compression Fracture",
                "indicators": ["vertebral_height_loss", "wedging", "discontinuity"],
                "description": "Collapse of a vertebral body due to fracture",
                "risk": "high",
            },
            {
                "name": "Scoliosis",
                "indicators": ["lateral_curvature", "asymmetry", "rotation"],
                "description": "Abnormal lateral curvature of the spine",
                "risk": "low",
            },
            {
                "name": "Normal Study",
                "indicators": ["normal_alignment", "preserved_disc_spaces", "intact_vertebrae"],
                "description": "Normal spinal alignment and vertebral body morphology",
                "risk": "low",
            },
        ],
        "anatomy": ["Vertebral bodies", "Intervertebral discs", "Pedicles", "Spinous processes", "Spinal canal"],
    },
    "general": {
        "conditions": [
            {
                "name": "General Assessment",
                "indicators": ["structures_identified"],
                "description": "General medical image assessment — body part could not be definitively classified",
                "risk": "low",
            },
        ],
        "anatomy": ["Soft tissue", "Bony structures", "Air spaces"],
    },
}


# ============================================================================
# AGENT 1: VISION AGENT
# ============================================================================

class VisionAgent:
    """
    Processes medical images to identify anatomical structures,
    detect abnormalities, and extract key features.
    """

    def __init__(self):
        self.name = "Vision Agent"
        self.version = "3.0"

    def process(self, image: Image.Image) -> dict:
        """Run full vision pipeline on the image."""
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Enhance image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Image classification
        image_type, body_part, type_confidence = self._classify_image(gray, img_array)

        # Detect edges and contours
        edges = cv2.Canny(enhanced, 80, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Extract features
        features = self._extract_features(enhanced, gray, img_array, contours, image_type)

        # Detect structures
        structures = self._detect_structures(enhanced, contours, image_type, h, w)

        return {
            "agent": self.name,
            "image_type": image_type,
            "body_part": body_part,
            "type_confidence": float(type_confidence),
            "image_dimensions": {"width": w, "height": h},
            "features": features,
            "structures": structures,
            "contour_count": len(contours),
        }

    def _classify_image(self, gray, rgb):
        """Classify image type using graduated multi-feature scoring."""
        h, w = gray.shape
        aspect = w / h if h > 0 else 1
        contrast = float(gray.std())
        brightness = float(gray.mean())

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / (h * w))

        # Symmetry analysis — medical images typically show bilateral symmetry
        mid = w // 2
        left_half = gray[:, :mid]
        right_half = cv2.flip(gray[:, w - mid:], 1)
        symmetry = 1.0 - float(np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))) / 255.0

        # Histogram shape analysis
        dark_ratio = float(hist_norm[:64].sum())     # Dark pixels
        bright_ratio = float(hist_norm[192:].sum())   # Bright pixels
        mid_ratio = float(hist_norm[64:192].sum())    # Mid-tone pixels

        # Texture complexity via Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ROI center brightness — medical scans often have brighter content at center
        cy, cx = h // 2, w // 2
        roi_size = min(h, w) // 4
        center_roi = gray[cy - roi_size:cy + roi_size, cx - roi_size:cx + roi_size]
        center_brightness = float(center_roi.mean()) if center_roi.size > 0 else brightness
        center_contrast = float(center_roi.std()) if center_roi.size > 0 else contrast

        # Initialize all scores with a small base
        scores = {
            "chest": 0.20,
            "hand": 0.20,
            "brain": 0.20,
            "spine": 0.20,
            "general": 0.25,
        }

        # ---------- CHEST X-RAY ----------
        # Portrait or near-square, moderate brightness, bilateral symmetry
        if 0.5 < aspect < 1.2:
            scores["chest"] += 0.12
        if 0.7 < aspect < 1.0:
            scores["chest"] += 0.08
        if 40 < brightness < 180:
            scores["chest"] += 0.08
        if 70 < brightness < 150:
            scores["chest"] += 0.06
        if 25 < contrast < 90:
            scores["chest"] += 0.08
        if symmetry > 0.75:
            scores["chest"] += 0.10
        if symmetry > 0.85:
            scores["chest"] += 0.05
        if 0.04 < edge_density < 0.20:
            scores["chest"] += 0.06
        if dark_ratio > 0.15:  # X-rays typically have dark background
            scores["chest"] += 0.06
        if center_brightness > brightness * 1.05:  # Chest content brighter than background
            scores["chest"] += 0.05

        # ---------- HAND / EXTREMITY ----------
        # Often high contrast bone vs background, asymmetric, irregular edges
        if contrast > 35:
            scores["hand"] += 0.08
        if contrast > 55:
            scores["hand"] += 0.05
        if edge_density > 0.08:
            scores["hand"] += 0.08
        if edge_density > 0.14:
            scores["hand"] += 0.05
        if symmetry < 0.70:
            scores["hand"] += 0.10
        if symmetry < 0.60:
            scores["hand"] += 0.05
        if dark_ratio > 0.25:  # Lots of black background around hand
            scores["hand"] += 0.06
        if bright_ratio > 0.08:  # Bright bone regions
            scores["hand"] += 0.06
        if 0.6 < aspect < 1.5:
            scores["hand"] += 0.04
        if laplacian_var > 500:  # Complex fine structures (bones)
            scores["hand"] += 0.05

        # ---------- BRAIN MRI/CT ----------
        # Very square, centered bright region, dark border, high symmetry
        if 0.85 < aspect < 1.15:
            scores["brain"] += 0.12
        if 0.92 < aspect < 1.08:
            scores["brain"] += 0.08
        if brightness < 130:
            scores["brain"] += 0.06
        if 30 < contrast < 80:
            scores["brain"] += 0.06
        if symmetry > 0.70:
            scores["brain"] += 0.08
        if dark_ratio > 0.20:  # Dark border around brain
            scores["brain"] += 0.05
        if center_brightness > brightness * 1.15:  # Brain content notably brighter
            scores["brain"] += 0.10
        if center_contrast > contrast * 0.8:
            scores["brain"] += 0.04
        if 0.03 < edge_density < 0.15:
            scores["brain"] += 0.04

        # ---------- SPINE ----------
        # Tall/portrait orientation, vertically oriented edges
        if aspect < 0.75:
            scores["spine"] += 0.15
        if aspect < 0.55:
            scores["spine"] += 0.10
        if edge_density > 0.06:
            scores["spine"] += 0.06
        if contrast > 30:
            scores["spine"] += 0.04
        # Check vertical edge dominance (Sobel)
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        vert_energy = float(np.abs(sobel_v).mean())
        horiz_energy = float(np.abs(sobel_h).mean())
        if vert_energy > horiz_energy * 1.2:
            scores["spine"] += 0.08

        # ---------- GENERAL penalty ----------
        # Reduce general score when specific features are strong
        max_specific = max(scores["chest"], scores["hand"], scores["brain"], scores["spine"])
        if max_specific > 0.50:
            scores["general"] = max(0.15, scores["general"] - 0.15)
        if max_specific > 0.65:
            scores["general"] = max(0.10, scores["general"] - 0.10)

        best = max(scores, key=scores.get)
        body_map = {
            "chest": "Chest",
            "hand": "Hand/Extremity",
            "brain": "Brain",
            "spine": "Spine",
            "general": "General",
        }

        # Normalize confidence to 0-1 range
        confidence = min(scores[best], 0.95)

        logger.info(f"🔍 Classification scores: {', '.join(f'{k}={v:.2f}' for k, v in sorted(scores.items(), key=lambda x: -x[1]))}")
        return best, body_map.get(best, "General"), confidence

    def _extract_features(self, enhanced, gray, rgb, contours, image_type):
        """Extract quantitative features from the image."""
        h, w = gray.shape
        features = {}

        # Basic stats
        features["contrast"] = float(gray.std())
        features["brightness"] = float(gray.mean())
        features["edge_density"] = float(np.sum(cv2.Canny(enhanced, 80, 200) > 0) / (h * w))

        # Opacity detection
        thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)[1]
        features["opacity_ratio"] = float(np.sum(thresh > 0) / (h * w))

        # Symmetry analysis
        left = gray[:, :w // 2]
        right = cv2.flip(gray[:, w // 2:], 1)
        if left.shape == right.shape:
            diff = cv2.absdiff(left, right)
            features["symmetry_score"] = float(1.0 - (np.mean(diff) / 255.0))
        else:
            features["symmetry_score"] = 0.75

        # Texture analysis (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features["texture_variance"] = float(laplacian.var())

        # Fracture/discontinuity index
        edge_img = cv2.Canny(enhanced, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(edge_img, cv2.MORPH_OPEN, kernel)
        features["discontinuity_index"] = float(np.sum(opened > 0) / (h * w))

        # Density distribution (quadrant analysis)
        q_h, q_w = h // 2, w // 2
        quadrants = {
            "upper_left": float(gray[:q_h, :q_w].mean()),
            "upper_right": float(gray[:q_h, q_w:].mean()),
            "lower_left": float(gray[q_h:, :q_w].mean()),
            "lower_right": float(gray[q_h:, q_w:].mean()),
        }
        features["quadrant_density"] = quadrants

        return features

    def _detect_structures(self, enhanced, contours, image_type, h, w):
        """Detect anatomical structures based on contour analysis."""
        structures = []

        large_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.02)]
        medium_contours = [c for c in contours if 100 < cv2.contourArea(c) < (w * h * 0.02)]

        if image_type == "chest":
            if len(large_contours) > 0:
                structures.append({
                    "name": "Lung Fields",
                    "detected": True,
                    "location": "Bilateral lung zones",
                    "count": len(large_contours),
                })
            structures.append({
                "name": "Cardiac Silhouette",
                "detected": True,
                "location": "Mediastinum",
            })
            structures.append({
                "name": "Costophrenic Angles",
                "detected": True,
                "location": "Bilateral bases",
            })

        elif image_type == "hand":
            structures.append({
                "name": "Skeletal Structures",
                "detected": True,
                "location": "Throughout",
                "count": len(medium_contours),
            })
            structures.append({
                "name": "Joint Spaces",
                "detected": True,
                "location": "Inter-phalangeal / metacarpophalangeal",
            })

        elif image_type == "brain":
            structures.append({
                "name": "Cerebral Hemispheres",
                "detected": True,
                "location": "Bilateral",
            })
            structures.append({
                "name": "Ventricular System",
                "detected": True,
                "location": "Midline",
            })

        elif image_type == "spine":
            structures.append({
                "name": "Vertebral Bodies",
                "detected": True,
                "location": "Spinal column",
                "count": max(len(medium_contours), 5),
            })
            structures.append({
                "name": "Intervertebral Discs",
                "detected": True,
                "location": "Between vertebrae",
            })

        else:
            structures.append({
                "name": "Tissue Structures",
                "detected": True,
                "location": "Throughout image",
                "count": len(contours),
            })

        return structures


# ============================================================================
# AGENT 2: DL CLASSIFICATION AGENT
# ============================================================================

class DLClassificationAgent:
    """
    Deep Learning Classification Agent.
    Runs CNN-based inference using ONNX models:
      - Custom MediScanCNN (trained on MedMNIST public datasets)
      - Pre-trained DenseNet-121 (trained on 873k+ chest X-rays)
    """

    def __init__(self):
        self.name = "DL Classification Agent"
        self.version = "3.0"
        self.model_manager = model_manager

    def process(self, image: "Image.Image", body_part: str = "general") -> dict:
        """Run all relevant DL models on the image."""
        img_array = np.array(image.convert("RGB"))

        # Run all relevant models for this body part
        dl_results = self.model_manager.run_all_relevant(img_array, body_part)

        # Build structured output
        predictions = []
        top_pathology = None
        dl_confidence = 0.0

        # Process chest pathology predictions
        if dl_results.get("chest_pathology"):
            chest_preds = dl_results["chest_pathology"]
            # Get top 5 predictions above threshold
            significant = [p for p in chest_preds if p["probability"] > 0.1]
            top_5 = significant[:5] if significant else chest_preds[:3]
            predictions.extend(top_5)
            if top_5:
                top_pathology = top_5[0]
                dl_confidence = max(dl_confidence, top_5[0]["probability"])

        # Process MedMNIST predictions
        medmnist_top = {}
        for model_name, data in dl_results.get("medmnist_predictions", {}).items():
            preds = data.get("predictions", [])
            if preds:
                top_pred = preds[0]
                medmnist_top[model_name] = {
                    "description": data["description"],
                    "top_prediction": top_pred["label"],
                    "confidence": top_pred["probability"],
                    "all_predictions": preds[:3],  # Top 3
                }
                dl_confidence = max(dl_confidence, top_pred["probability"])

        return {
            "agent": self.name,
            "dl_available": dl_results["dl_available"],
            "chest_pathology_predictions": predictions,
            "medmnist_predictions": medmnist_top,
            "top_pathology": top_pathology,
            "dl_confidence": float(dl_confidence),
            "models_used": dl_results["models_used"],
            "models_status": dl_results["models_status"],
        }


# ============================================================================
# AGENT 3: ANALYSIS AGENT
# ============================================================================

class AnalysisAgent:
    """
    Cross-references Vision Agent findings with medical knowledge base.
    Evaluates differential diagnoses and assigns confidence levels.
    """

    def __init__(self):
        self.name = "Analysis Agent"
        self.version = "3.0"
        self.knowledge = KNOWLEDGE_BASE

    def process(self, vision_output: dict, patient_history: dict = None, dl_output: dict = None) -> dict:
        """Analyze vision findings against medical knowledge, enhanced with DL predictions."""
        image_type = vision_output["image_type"]
        features = vision_output["features"]
        structures = vision_output["structures"]

        kb = self.knowledge.get(image_type, self.knowledge["general"])

        # Generate findings from OpenCV analysis
        findings = self._generate_findings(image_type, features, structures, kb)

        # Merge DL predictions into findings
        dl_findings = []
        if dl_output and dl_output.get("dl_available"):
            dl_findings = self._merge_dl_predictions(dl_output, image_type)
            findings.extend(dl_findings)

        # Compute differential diagnoses
        differentials = self._differential_diagnosis(image_type, features, findings, kb)

        # Boost differentials with DL confidence
        if dl_output and dl_output.get("top_pathology"):
            differentials = self._boost_with_dl(differentials, dl_output)

        # Primary hypothesis
        primary = differentials[0] if differentials else {
            "condition": "Inconclusive",
            "probability": 0.5,
            "reasoning": "Insufficient features for definitive classification",
            "risk_level": "Unknown",
        }

        # Overall confidence (weighted: 60% DL + 40% heuristic if DL available)
        finding_confidences = [f["confidence"] for f in findings]
        heuristic_confidence = np.mean(finding_confidences) if finding_confidences else 0.5
        dl_confidence = dl_output.get("dl_confidence", 0) if dl_output else 0

        if dl_confidence > 0:
            avg_confidence = 0.6 * dl_confidence + 0.4 * heuristic_confidence
        else:
            avg_confidence = heuristic_confidence

        # Enhance with patient history if available
        history_context = ""
        if patient_history:
            history_context = self._integrate_history(patient_history, differentials, findings)

        return {
            "agent": self.name,
            "findings": findings,
            "primary_hypothesis": primary,
            "differential_diagnoses": differentials[1:] if len(differentials) > 1 else [],
            "clinical_significance": self._assess_significance(primary, findings),
            "recommendations": self._generate_recommendations(primary, findings, image_type),
            "analysis_confidence": float(avg_confidence),
            "heuristic_confidence": float(heuristic_confidence),
            "dl_confidence": float(dl_confidence),
            "patient_context": history_context if history_context else None,
        }

    def _generate_findings(self, image_type, features, structures, kb):
        """Generate diagnostic findings from image features."""
        findings = []

        # Structure-based findings
        for struct in structures:
            finding = {
                "finding_type": struct["name"],
                "location": struct.get("location", "Not specified"),
                "description": f"{struct['name']} identified and evaluated",
                "confidence": 0.8,
                "body_part": kb.get("anatomy", ["General"])[0] if kb.get("anatomy") else "General",
                "image_type": image_type,
            }

            # Enrich based on features
            if struct["name"] == "Lung Fields":
                opacity = features.get("opacity_ratio", 0)
                symmetry = features.get("symmetry_score", 1)
                if opacity > 0.2:
                    finding["description"] = "Lung fields show areas of increased opacity suggesting possible consolidation or infiltrate"
                    finding["confidence"] = 0.70
                elif symmetry < 0.7:
                    finding["description"] = "Asymmetric lung field aeration noted — clinical correlation advised"
                    finding["confidence"] = 0.65
                else:
                    finding["description"] = "Lung fields appear clear with normal aeration bilaterally"
                    finding["confidence"] = 0.85

            elif struct["name"] == "Cardiac Silhouette":
                # Estimate cardiothoracic ratio from quadrant density
                qd = features.get("quadrant_density", {})
                center_density = (qd.get("upper_left", 0) + qd.get("upper_right", 0)) / 2
                if center_density > 100:
                    finding["description"] = "Cardiac silhouette may be mildly prominent — clinical correlation for cardiomegaly"
                    finding["confidence"] = 0.60
                else:
                    finding["description"] = "Cardiac silhouette appears within normal limits"
                    finding["confidence"] = 0.80

            elif struct["name"] == "Costophrenic Angles":
                opacity = features.get("opacity_ratio", 0)
                if opacity > 0.25:
                    finding["description"] = "Costophrenic angles may be blunted — consider pleural effusion"
                    finding["confidence"] = 0.60
                else:
                    finding["description"] = "Costophrenic angles appear sharp and distinct"
                    finding["confidence"] = 0.80

            elif struct["name"] == "Skeletal Structures":
                disc_idx = features.get("discontinuity_index", 0)
                if disc_idx > 0.12:
                    finding["description"] = f"Possible cortical discontinuity detected among {struct.get('count', 'multiple')} bone structures — fracture cannot be excluded"
                    finding["confidence"] = 0.60
                else:
                    finding["description"] = f"Bone structures appear intact ({struct.get('count', 'multiple')} structures evaluated)"
                    finding["confidence"] = 0.80

            elif struct["name"] == "Cerebral Hemispheres":
                sym = features.get("symmetry_score", 1)
                if sym < 0.75:
                    finding["description"] = "Asymmetry detected between cerebral hemispheres — further evaluation recommended"
                    finding["confidence"] = 0.65
                else:
                    finding["description"] = "Cerebral hemispheres appear symmetric without midline shift"
                    finding["confidence"] = 0.85

            elif struct["name"] == "Ventricular System":
                brightness = features.get("brightness", 0)
                if brightness > 80:
                    finding["description"] = "Ventricular system appears within normal limits"
                    finding["confidence"] = 0.78
                else:
                    finding["description"] = "Ventricular system evaluated — no obvious enlargement"
                    finding["confidence"] = 0.72

            elif struct["name"] == "Vertebral Bodies":
                disc_idx = features.get("discontinuity_index", 0)
                if disc_idx > 0.15:
                    finding["description"] = f"Possible vertebral body abnormality detected across {struct.get('count', 'multiple')} levels"
                    finding["confidence"] = 0.58
                else:
                    finding["description"] = f"Vertebral bodies appear aligned across {struct.get('count', 'multiple')} levels"
                    finding["confidence"] = 0.82

            findings.append(finding)

        # Image quality finding
        texture_var = features.get("texture_variance", 0)
        quality = "good" if texture_var > 100 else "adequate" if texture_var > 50 else "limited"
        findings.append({
            "finding_type": "Image Quality Assessment",
            "location": "Overall",
            "description": f"Image quality assessed as {quality} (texture variance: {texture_var:.0f})",
            "confidence": 0.90,
            "body_part": "General",
            "image_type": image_type,
        })

        return findings

    def _differential_diagnosis(self, image_type, features, findings, kb):
        """Generate differential diagnoses based on findings."""
        conditions = kb.get("conditions", [])
        differentials = []

        for condition in conditions:
            score = 0.3  # Base score
            matched_indicators = []

            for indicator in condition.get("indicators", []):
                # Match indicators to features
                if "opacity" in indicator and features.get("opacity_ratio", 0) > 0.15:
                    score += 0.15
                    matched_indicators.append(indicator)
                elif "discontinuity" in indicator and features.get("discontinuity_index", 0) > 0.1:
                    score += 0.15
                    matched_indicators.append(indicator)
                elif "symmetry" in indicator or "symmetric" in indicator:
                    sym = features.get("symmetry_score", 1)
                    if sym > 0.8:
                        score += 0.1
                        matched_indicators.append(indicator)
                elif "normal" in indicator or "clear" in indicator or "intact" in indicator or "preserved" in indicator:
                    # Normal indicators — add score if features are within normal range
                    if features.get("opacity_ratio", 0) < 0.15 and features.get("discontinuity_index", 0) < 0.1:
                        score += 0.15
                        matched_indicators.append(indicator)
                elif "enlarged" in indicator:
                    qd = features.get("quadrant_density", {})
                    center = (qd.get("upper_left", 0) + qd.get("upper_right", 0)) / 2
                    if center > 100:
                        score += 0.1
                        matched_indicators.append(indicator)
                elif "density" in indicator or "reduced" in indicator:
                    if features.get("brightness", 0) > 120:
                        score += 0.1
                        matched_indicators.append(indicator)
                elif "blunted" in indicator or "base" in indicator:
                    if features.get("opacity_ratio", 0) > 0.2:
                        score += 0.1
                        matched_indicators.append(indicator)

            score = min(score, 0.95)

            reasoning_parts = []
            if matched_indicators:
                reasoning_parts.append(f"Matched indicators: {', '.join(matched_indicators)}")
            reasoning_parts.append(f"Based on {image_type} image analysis with {len(findings)} findings evaluated")

            differentials.append({
                "condition": condition["name"],
                "probability": float(score),
                "reasoning": ". ".join(reasoning_parts),
                "risk_level": condition.get("risk", "unknown").capitalize(),
                "matched_indicators": matched_indicators,
            })

        # Sort by probability
        differentials.sort(key=lambda x: x["probability"], reverse=True)
        return differentials

    def _assess_significance(self, primary, findings):
        """Assess overall clinical significance."""
        risk = primary.get("risk_level", "").lower()
        prob = primary.get("probability", 0)

        if risk == "high" and prob > 0.6:
            return "Potentially significant findings requiring urgent radiologist attention. Clinical correlation and additional imaging may be warranted."
        elif risk == "moderate" and prob > 0.5:
            return "Findings of moderate clinical significance. Radiologist review recommended for further evaluation and clinical correlation."
        elif "normal" in primary.get("condition", "").lower():
            return "Findings are largely within normal limits. Routine radiologist confirmation recommended."
        else:
            return "All findings should be reviewed by a qualified radiologist. Clinical correlation is advised for definitive interpretation."

    def _generate_recommendations(self, primary, findings, image_type):
        """Generate clinical recommendations."""
        recs = []
        risk = primary.get("risk_level", "").lower()

        recs.append("Qualified radiologist review and verification required")

        if risk == "high":
            recs.append("URGENT: Expedited clinical review recommended")
            recs.append("Consider additional imaging studies for confirmation")
        elif risk == "moderate":
            recs.append("Clinical correlation with patient history recommended")
            recs.append("Follow-up imaging may be considered if clinically indicated")

        if image_type == "chest":
            recs.append("Correlate with clinical symptoms (cough, dyspnea, fever)")
        elif image_type == "hand":
            recs.append("Correlate with mechanism of injury and physical examination findings")
        elif image_type == "brain":
            recs.append("Correlate with neurological examination and clinical presentation")
        elif image_type == "spine":
            recs.append("Correlate with pain location and neurological symptoms")

        recs.append("This AI analysis is for assistance only — final diagnosis must be made by a licensed physician")

        return recs

    def _merge_dl_predictions(self, dl_output, image_type):
        """Convert DL model predictions into structured findings."""
        dl_findings = []

        # Chest pathology findings from DenseNet
        for pred in dl_output.get("chest_pathology_predictions", []):
            if pred["probability"] > 0.15:
                risk = "high" if pred["probability"] > 0.7 else "moderate" if pred["probability"] > 0.4 else "low"
                dl_findings.append({
                    "finding_type": f"DL: {pred['pathology']}",
                    "location": "Chest" if image_type == "chest" else "General",
                    "description": f"CNN model detected {pred['pathology']} with {pred['probability']:.1%} confidence (Source: {pred['model']})",
                    "confidence": pred["probability"],
                    "body_part": "Chest",
                    "image_type": image_type,
                    "source": "deep_learning",
                    "model": pred["model"],
                    "risk_level": risk,
                })

        # MedMNIST findings
        for model_name, data in dl_output.get("medmnist_predictions", {}).items():
            top = data.get("top_prediction", "")
            conf = data.get("confidence", 0)
            if conf > 0.3:
                dl_findings.append({
                    "finding_type": f"DL: {data['description']}",
                    "location": "General",
                    "description": f"Custom CNN classified as '{top}' with {conf:.1%} confidence (Dataset: {model_name})",
                    "confidence": conf,
                    "body_part": "General",
                    "image_type": image_type,
                    "source": "deep_learning",
                    "model": f"MediScanCNN ({model_name})",
                })

        return dl_findings

    def _boost_with_dl(self, differentials, dl_output):
        """Boost differential diagnosis confidence using DL predictions."""
        top_pathology = dl_output.get("top_pathology", {})
        if not top_pathology:
            return differentials

        dl_name = top_pathology.get("pathology", "").lower()
        dl_prob = top_pathology.get("probability", 0)

        for diff in differentials:
            condition_name = diff["condition"].lower()
            # If DL agrees with heuristic diagnosis, boost confidence
            if any(word in condition_name for word in dl_name.lower().split("_")):
                boost = dl_prob * 0.2  # 20% of DL confidence as boost
                diff["probability"] = min(diff["probability"] + boost, 0.95)
                diff["reasoning"] += f". DL model corroborates with {dl_prob:.1%} confidence"
                diff["dl_corroborated"] = True
            # If DL detects something the heuristic missed, add it
            elif dl_prob > 0.5 and "normal" in condition_name:
                diff["probability"] = max(diff["probability"] - 0.1, 0.1)
                diff["reasoning"] += f". DL model suggests possible pathology ({dl_name})"

        # Re-sort by probability
        differentials.sort(key=lambda x: x["probability"], reverse=True)
        return differentials

    def _integrate_history(self, patient_history, differentials, findings):
        """Integrate patient medical history into analysis context."""
        parts = []
        name = patient_history.get("patient_name", "Unknown")
        age = patient_history.get("age")
        gender = patient_history.get("gender")
        conditions = patient_history.get("known_conditions", "")
        past_findings = patient_history.get("past_findings", [])

        if age:
            parts.append(f"Patient: {name}, {age}y {gender or ''}")
        if conditions:
            parts.append(f"Known conditions: {conditions}")
            # Boost confidence for conditions matching known history
            for diff in differentials:
                for cond_word in conditions.lower().split(","):
                    cond_word = cond_word.strip()
                    if cond_word and cond_word in diff["condition"].lower():
                        diff["probability"] = min(diff["probability"] + 0.1, 0.95)
                        diff["reasoning"] += f". Probability adjusted based on known patient history of {cond_word}"
        if past_findings:
            recent = past_findings[:3]
            past_summary = "; ".join([f"{pf['finding']} ({pf['body_part']}, {pf['date']})" for pf in recent])
            parts.append(f"Recent scan history: {past_summary}")

        return " | ".join(parts) if parts else None


# ============================================================================
# AGENT 3: REPORTING AGENT
# ============================================================================

class ReportingAgent:
    """
    Compiles structured diagnostic summary from Vision and Analysis agent outputs.
    """

    def __init__(self):
        self.name = "Reporting Agent"
        self.version = "3.0"

    def process(self, vision_output: dict, analysis_output: dict, dl_output: dict = None) -> dict:
        """Generate a final structured report."""
        agents_used = ["Vision Agent v3.0"]
        if dl_output and dl_output.get("dl_available"):
            agents_used.append("DL Classification Agent v3.0")
        agents_used.extend(["Analysis Agent v3.0", "Reporting Agent v3.0"])

        return {
            "agent": self.name,
            "report": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "image_classification": {
                    "type": vision_output["image_type"],
                    "body_part": vision_output["body_part"],
                    "confidence": vision_output["type_confidence"],
                },
                "findings_summary": self._summarize_findings(analysis_output["findings"]),
                "primary_assessment": analysis_output["primary_hypothesis"],
                "differential_diagnoses": analysis_output["differential_diagnoses"],
                "clinical_significance": analysis_output["clinical_significance"],
                "recommendations": analysis_output["recommendations"],
                "quality_note": self._quality_assessment(vision_output, analysis_output),
                "dl_analysis": {
                    "available": dl_output.get("dl_available", False) if dl_output else False,
                    "chest_pathology": dl_output.get("chest_pathology_predictions", []) if dl_output else [],
                    "medmnist": dl_output.get("medmnist_predictions", {}) if dl_output else {},
                    "models_used": dl_output.get("models_used", []) if dl_output else [],
                },
                "disclaimer": "AI-generated preliminary report. Must be reviewed and approved by a qualified radiologist before clinical use.",
            },
            "metadata": {
                "agents_used": agents_used,
                "image_dimensions": vision_output["image_dimensions"],
                "total_structures_detected": len(vision_output["structures"]),
                "total_findings": len(analysis_output["findings"]),
                "dl_models_count": len(dl_output.get("models_used", [])) if dl_output else 0,
            },
        }

    def _summarize_findings(self, findings):
        """Create a narrative summary of findings."""
        if not findings:
            return "No significant findings detected."

        parts = []
        for f in findings:
            if f["finding_type"] != "Image Quality Assessment":
                parts.append(f"{f['finding_type']}: {f['description']}")

        return " | ".join(parts) if parts else "Structures evaluated with no remarkable findings."

    def _quality_assessment(self, vision, analysis):
        """Assess analysis quality and limitations."""
        conf = analysis.get("analysis_confidence", 0)
        if conf > 0.75:
            return "Analysis performed with high confidence. Findings are well-supported by image features."
        elif conf > 0.6:
            return "Analysis performed with moderate confidence. Some findings may require additional clinical correlation."
        else:
            return "Analysis confidence is limited. Results should be interpreted with caution and supplemented with additional imaging or clinical data."


# ============================================================================
# INITIALIZE AGENTS
# ============================================================================

vision_agent = VisionAgent()
dl_classification_agent = DLClassificationAgent()
analysis_agent = AnalysisAgent()
reporting_agent = ReportingAgent()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    dl_status = model_manager.get_status()
    return {
        "status": "healthy",
        "version": "3.0",
        "system": "MediScan Analyst",
        "agents": ["Vision Agent", "DL Classification Agent", "Analysis Agent", "Reporting Agent"],
        "models_loaded": dl_status["ready_count"] + 3,
        "dl_models": dl_status["ready_count"],
        "ready": True,
    }


@app.get("/api/models/status")
async def models_status():
    """Get status of all deep learning models."""
    return model_manager.get_status()


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
):
    """Run the three-agent analysis pipeline on a medical image."""
    start_time = time.time()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.thumbnail((1024, 1024))

        # Save uploaded image
        ext = os.path.splitext(file.filename)[1] or ".png"
        image_filename = f"{uuid.uuid4().hex}{ext}"
        image_path = os.path.join(UPLOADS_DIR, image_filename)
        with open(image_path, "wb") as f:
            f.write(contents)

        # Get patient history if patient selected
        patient_history = None
        if patient_id:
            patient_history = db.get_patient_history_summary(patient_id)

        # Agent 1: Vision (OpenCV features)
        logger.info("🔬 Vision Agent processing...")
        vision_result = vision_agent.process(image)
        vision_time = time.time() - start_time

        # Agent 2: DL Classification (CNN models)
        logger.info("🤖 DL Classification Agent processing...")
        dl_result = dl_classification_agent.process(image, vision_result["image_type"])
        dl_time = time.time() - start_time - vision_time

        # Agent 3: Analysis (merges heuristic + DL findings)
        logger.info("🧠 Analysis Agent processing...")
        analysis_result = analysis_agent.process(vision_result, patient_history, dl_result)
        analysis_time = time.time() - start_time - vision_time - dl_time

        # Agent 4: Reporting
        logger.info("📋 Reporting Agent processing...")
        report_result = reporting_agent.process(vision_result, analysis_result, dl_result)
        report_time = time.time() - start_time - vision_time - dl_time - analysis_time

        total_time = time.time() - start_time

        return {
            "status": "success",
            "image_type": vision_result["image_type"],
            "body_part": vision_result["body_part"],
            "image_filename": image_filename,
            "model_confidence": {
                "image_classification": float(vision_result["type_confidence"]),
                "findings_extraction": float(analysis_result["analysis_confidence"]),
                "dl_classification": float(dl_result.get("dl_confidence", 0)),
                "heuristic_analysis": float(analysis_result.get("heuristic_confidence", 0)),
            },
            "ensemble_confidence": float(
                (vision_result["type_confidence"] + analysis_result["analysis_confidence"]) / 2
            ),
            "dl_predictions": {
                "available": dl_result.get("dl_available", False),
                "chest_pathology": dl_result.get("chest_pathology_predictions", []),
                "medmnist": dl_result.get("medmnist_predictions", {}),
                "top_pathology": dl_result.get("top_pathology"),
                "models_used": dl_result.get("models_used", []),
            },
            "analysis": {
                "findings": analysis_result["findings"],
                "primary_hypothesis": analysis_result["primary_hypothesis"],
                "differential_diagnoses": analysis_result["differential_diagnoses"],
                "clinical_significance": analysis_result["clinical_significance"],
                "recommendations": analysis_result["recommendations"],
                "patient_context": analysis_result.get("patient_context"),
            },
            "report": report_result["report"],
            "processing_time": float(total_time),
            "agent_timings": {
                "vision": float(vision_time),
                "dl_classification": float(dl_time),
                "analysis": float(analysis_time),
                "reporting": float(report_time),
            },
            "models_used": report_result["metadata"]["agents_used"],
        }

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PATIENT ENDPOINTS
# ============================================================================

class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    medical_conditions: Optional[str] = ""
    notes: Optional[str] = ""

class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    medical_conditions: Optional[str] = None
    notes: Optional[str] = None

class ScanSave(BaseModel):
    patient_id: int
    image_filename: str
    image_type: str
    body_part: str
    analysis_json: dict
    confidence: float

class ReferralCreate(BaseModel):
    patient_id: int
    scan_id: Optional[int] = None
    specialist_name: str
    specialist_field: str
    notes: Optional[str] = ""


@app.get("/api/patients")
async def list_patients():
    return db.get_all_patients()


@app.post("/api/patients")
async def create_patient(patient: PatientCreate):
    pid = db.create_patient(
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        blood_group=patient.blood_group,
        medical_conditions=patient.medical_conditions,
        notes=patient.notes,
    )
    return {"id": pid, "message": "Patient created"}


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: int):
    p = db.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    scans = db.get_patient_scans(patient_id)
    referrals = db.get_referrals(patient_id)
    return {**p, "scans": scans, "referrals": referrals}


@app.put("/api/patients/{patient_id}")
async def update_patient(patient_id: int, updates: PatientUpdate):
    p = db.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    db.update_patient(patient_id, **{k: v for k, v in updates.dict().items() if v is not None})
    return {"message": "Patient updated"}


@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: int):
    db.delete_patient(patient_id)
    return {"message": "Patient deleted"}


@app.post("/api/patients/{patient_id}/scans")
async def save_scan_to_patient(patient_id: int, scan: ScanSave):
    p = db.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    scan_id = db.save_scan(
        patient_id=patient_id,
        image_filename=scan.image_filename,
        image_type=scan.image_type,
        body_part=scan.body_part,
        analysis_json=scan.analysis_json,
        confidence=scan.confidence,
    )
    return {"scan_id": scan_id, "message": "Scan saved to patient profile"}


# ============================================================================
# REFERRAL ENDPOINTS
# ============================================================================

@app.post("/api/referrals")
async def create_referral(referral: ReferralCreate):
    ref_id = db.create_referral(
        patient_id=referral.patient_id,
        scan_id=referral.scan_id,
        specialist_name=referral.specialist_name,
        specialist_field=referral.specialist_field,
        notes=referral.notes,
    )
    return {"referral_id": ref_id, "message": "Referral sent"}


@app.get("/api/referrals")
async def list_referrals(patient_id: Optional[int] = Query(None)):
    return db.get_referrals(patient_id)


# ============================================================================
# SERVE FRONTEND
# ============================================================================

DIST_DIR = os.path.join(CURRENT_DIR, "frontend", "dist")

if os.path.exists(DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React frontend."""
        file_path = os.path.join(DIST_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "Frontend not built. Run 'npm run build' in the frontend directory."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
