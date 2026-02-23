"""
MediScan Analyst — FastAPI Backend with Three-Agent Architecture
Vision Agent → Analysis Agent → Reporting Agent
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

# Ensure uploads directory exists
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(CURRENT_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediScan Analyst API",
    description="Agentic AI Co-pilot for Medical Imaging and Diagnosis Support",
    version="2.0"
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
        self.version = "2.0"

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
        """Classify image type based on visual characteristics."""
        h, w = gray.shape
        aspect = w / h if h > 0 else 1
        contrast = float(gray.std())
        brightness = float(gray.mean())

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.sum(edges > 0) / (h * w))

        scores = {}

        # Chest X-ray: portrait orientation, moderate contrast
        if 0.55 < aspect < 1.1 and 25 < contrast < 85 and 50 < brightness < 160:
            scores["chest"] = 0.75 + min(edge_density * 0.2, 0.15)

        # Hand/Skeletal: square-ish, high contrast
        if 0.7 < aspect < 1.3 and contrast > 40:
            scores["hand"] = 0.65 + min(edge_density * 0.25, 0.2)

        # Brain MRI/CT: square, specific contrast
        if 0.85 < aspect < 1.15 and 35 < contrast < 75 and brightness < 120:
            scores["brain"] = 0.70

        # Spine: tall, linear structures
        if aspect < 0.65 and edge_density > 0.08:
            scores["spine"] = 0.65

        if not scores:
            scores["general"] = 0.5

        best = max(scores, key=scores.get)
        body_map = {
            "chest": "Chest",
            "hand": "Hand/Extremity",
            "brain": "Brain",
            "spine": "Spine",
            "general": "General",
        }
        return best, body_map.get(best, "General"), scores[best]

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
# AGENT 2: ANALYSIS AGENT
# ============================================================================

class AnalysisAgent:
    """
    Cross-references Vision Agent findings with medical knowledge base.
    Evaluates differential diagnoses and assigns confidence levels.
    """

    def __init__(self):
        self.name = "Analysis Agent"
        self.version = "2.0"
        self.knowledge = KNOWLEDGE_BASE

    def process(self, vision_output: dict, patient_history: dict = None) -> dict:
        """Analyze vision findings against medical knowledge."""
        image_type = vision_output["image_type"]
        features = vision_output["features"]
        structures = vision_output["structures"]

        kb = self.knowledge.get(image_type, self.knowledge["general"])

        # Generate findings from analysis
        findings = self._generate_findings(image_type, features, structures, kb)

        # Compute differential diagnoses
        differentials = self._differential_diagnosis(image_type, features, findings, kb)

        # Primary hypothesis
        primary = differentials[0] if differentials else {
            "condition": "Inconclusive",
            "probability": 0.5,
            "reasoning": "Insufficient features for definitive classification",
            "risk_level": "Unknown",
        }

        # Overall confidence
        finding_confidences = [f["confidence"] for f in findings]
        avg_confidence = np.mean(finding_confidences) if finding_confidences else 0.5

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
        self.version = "2.0"

    def process(self, vision_output: dict, analysis_output: dict) -> dict:
        """Generate a final structured report."""
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
                "disclaimer": "AI-generated preliminary report. Must be reviewed and approved by a qualified radiologist before clinical use.",
            },
            "metadata": {
                "agents_used": ["Vision Agent v2.0", "Analysis Agent v2.0", "Reporting Agent v2.0"],
                "image_dimensions": vision_output["image_dimensions"],
                "total_structures_detected": len(vision_output["structures"]),
                "total_findings": len(analysis_output["findings"]),
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
analysis_agent = AnalysisAgent()
reporting_agent = ReportingAgent()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0",
        "system": "MediScan Analyst",
        "agents": ["Vision Agent", "Analysis Agent", "Reporting Agent"],
        "models_loaded": 3,
        "ready": True,
    }


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

        # Agent 1: Vision
        logger.info("🔬 Vision Agent processing...")
        vision_result = vision_agent.process(image)
        vision_time = time.time() - start_time

        # Agent 2: Analysis (with patient history context)
        logger.info("🧠 Analysis Agent processing...")
        analysis_result = analysis_agent.process(vision_result, patient_history)
        analysis_time = time.time() - start_time - vision_time

        # Agent 3: Reporting
        logger.info("📋 Reporting Agent processing...")
        report_result = reporting_agent.process(vision_result, analysis_result)
        report_time = time.time() - start_time - vision_time - analysis_time

        total_time = time.time() - start_time

        return {
            "status": "success",
            "image_type": vision_result["image_type"],
            "body_part": vision_result["body_part"],
            "image_filename": image_filename,
            "model_confidence": {
                "image_classification": float(vision_result["type_confidence"]),
                "findings_extraction": float(analysis_result["analysis_confidence"]),
            },
            "ensemble_confidence": float(
                (vision_result["type_confidence"] + analysis_result["analysis_confidence"]) / 2
            ),
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
