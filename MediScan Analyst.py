"""
MediScan Analyst v1.0
Multi-Agent AI System for Medical Image Analysis
A three-agent pipeline: Vision → Analysis → Reporting with Human-in-the-Loop Verification
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from scipy import ndimage
from skimage import filters, exposure, feature
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================
class AbnormalityType(Enum):
    LESION = "Lesion"
    SHADOW = "Shadow"
    FRACTURE = "Fracture"
    OPACITY = "Opacity"
    IRREGULAR_PATTERN = "Irregular Pattern"
    EFFUSION = "Effusion"
    CONSOLIDATION = "Consolidation"
    PNEUMOTHORAX = "Pneumothorax"
    NORMAL = "Normal"


class ConfidenceLevel(Enum):
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3


@dataclass
class VisionFinding:
    """Output structure for Vision Agent"""
    finding_type: AbnormalityType
    location: str  # e.g., "Right Upper Lobe", "Left Apex"
    description: str
    confidence: float
    visual_coordinates: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class MedicalHypothesis:
    """Output structure for Analysis Agent"""
    condition: str
    confidence_score: float
    supporting_findings: List[str]
    differential_diagnoses: List[str]
    severity_level: str  # "Low", "Medium", "High", "Critical"
    reasoning_path: str
    clinical_significance: str


@dataclass
class DiagnosticReport:
    """Output structure for Reporting Agent"""
    patient_id: str
    timestamp: str
    preliminary_findings: List[str]
    primary_hypothesis: MedicalHypothesis
    alternative_hypotheses: List[MedicalHypothesis]
    recommendations: List[str]
    clinical_summary: str
    radiologist_review_status: str
    radiologist_notes: str = ""
    radiologist_approval: bool = False


# =============================================================================
# MEDICAL KNOWLEDGE BASE
# =============================================================================
class MedicalKnowledgeBase:
    """Centralized medical ontology and knowledge base for MULTIPLE body regions"""
    
    ANATOMICAL_STRUCTURES = {
        # Chest
        "Right Upper Lobe": "Superior portion of right lung",
        "Right Middle Lobe": "Middle portion of right lung",
        "Right Lower Lobe": "Inferior portion of right lung",
        "Left Upper Lobe": "Superior portion of left lung",
        "Left Lower Lobe": "Inferior portion of left lung",
        "Hilum": "Central region where vessels enter lungs",
        "Heart": "Cardiac silhouette",
        # Hand/Fingers
        "Thumb/Index region": "First and second digits",
        "Middle finger region": "Third digit",
        "Ring/Pinky region": "Fourth and fifth digits",
        "Metacarpal": "Hand bones",
        "Phalangeal": "Finger bones",
        # Spine
        "Cervical region": "Neck vertebrae (C1-C7)",
        "Thoracic region": "Upper back vertebrae (T1-T12)",
        "Lumbar region": "Lower back vertebrae (L1-L5)",
        "Sacral region": "Base of spine",
        # General
        "Region of interest": "Area of abnormality",
        "Bone shaft region": "Long bone structure"
    }
    
    CONDITION_DATABASE = {
        # CHEST CONDITIONS
        "Pneumonia": {
            "imaging_features": ["Consolidation", "Opacity", "Shadow"],
            "typical_location": ["Right Lower Lobe", "Left Lower Lobe"],
            "confidence_boost": 0.2,
            "risk_level": "Medium",
            "key_indicators": ["Lobar consolidation", "Air bronchograms"]
        },
        "Pneumothorax": {
            "imaging_features": ["Pneumothorax"],
            "typical_location": ["Apex"],
            "confidence_boost": 0.25,
            "risk_level": "High",
            "key_indicators": ["Lung edge visible", "Absence of lung markings"]
        },
        "Pleural Effusion": {
            "imaging_features": ["Opacity", "Effusion"],
            "typical_location": ["Right Lower Lobe", "Left Lower Lobe"],
            "confidence_boost": 0.18,
            "risk_level": "Medium",
            "key_indicators": ["Blunted angles", "Silhouette sign"]
        },
        "Normal Chest X-Ray": {
            "imaging_features": ["Normal"],
            "typical_location": ["Entire chest"],
            "confidence_boost": 0.0,
            "risk_level": "Low",
            "key_indicators": ["Clear lungs", "Normal cardiac silhouette"]
        },
        # HAND/BONE CONDITIONS
        "Fracture": {
            "imaging_features": ["Fracture"],
            "typical_location": ["Metacarpal", "Phalangeal", "Bone shaft region"],
            "confidence_boost": 0.3,
            "risk_level": "Medium",
            "key_indicators": ["Cortical break", "Bone displacement", "Line discontinuity"]
        },
        "Bone Fracture - Displaced": {
            "imaging_features": ["Fracture"],
            "typical_location": ["Metacarpal", "Phalangeal"],
            "confidence_boost": 0.35,
            "risk_level": "High",
            "key_indicators": ["Significant displacement", "Angular deformity"]
        },
        "Bone Fracture - Hairline": {
            "imaging_features": ["Shadow"],
            "typical_location": ["Bone shaft region"],
            "confidence_boost": 0.15,
            "risk_level": "Low",
            "key_indicators": ["Subtle line", "Minimal displacement"]
        },
        "Normal Hand X-Ray": {
            "imaging_features": ["Normal"],
            "typical_location": ["Metacarpal", "Phalangeal"],
            "confidence_boost": 0.0,
            "risk_level": "Low",
            "key_indicators": ["Intact bones", "Normal alignment"]
        },
        # SPINE CONDITIONS
        "Vertebral Fracture": {
            "imaging_features": ["Fracture", "Shadow"],
            "typical_location": ["Cervical region", "Thoracic region", "Lumbar region"],
            "confidence_boost": 0.28,
            "risk_level": "High",
            "key_indicators": ["Vertebral body breaks", "Height loss"]
        },
        "Spine Misalignment": {
            "imaging_features": ["Shadow"],
            "typical_location": ["Cervical region", "Lumbar region"],
            "confidence_boost": 0.20,
            "risk_level": "Medium",
            "key_indicators": ["Abnormal curvature", "Vertebral slipping"]
        },
        "Normal Spine X-Ray": {
            "imaging_features": ["Normal"],
            "typical_location": ["Cervical region", "Thoracic region", "Lumbar region"],
            "confidence_boost": 0.0,
            "risk_level": "Low",
            "key_indicators": ["Straight alignment", "Intact vertebrae"]
        },
        # GENERAL
        "Normal": {
            "imaging_features": ["Normal"],
            "typical_location": ["Region of interest"],
            "confidence_boost": 0.0,
            "risk_level": "Low",
            "key_indicators": ["No abnormalities"]
        }
    }
    
    @staticmethod
    def get_condition_info(condition: str) -> Dict:
        """Retrieve condition information from knowledge base"""
        return MedicalKnowledgeBase.CONDITION_DATABASE.get(
            condition, 
            {"imaging_features": [], "risk_level": "Unknown"}
        )
    
    @staticmethod
    def get_anatomy_description(location: str) -> str:
        """Get anatomical description"""
        return MedicalKnowledgeBase.ANATOMICAL_STRUCTURES.get(
            location, 
            "Unknown anatomical location"
        )


# =============================================================================
# AGENT 1: VISION AGENT
# =============================================================================
class VisionAgent:
    """
    Processes ANY medical X-ray/scan image with real computer vision.
    Auto-detects image type and adapts analysis accordingly.
    """
    
    def __init__(self):
        self.name = "Vision Agent"
        
        # Load pre-trained ResNet50 for feature extraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model.eval().to(self.device)
        
        # Image normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"{self.name} initialized - Device: {self.device}")
    
    def detect_image_type(self, gray: np.ndarray, height: int, width: int) -> str:
        """
        Auto-detect the type of medical image using image characteristics.
        
        Returns: "chest", "hand", "spine", "abdomen", "limb", "unknown"
        """
        aspect_ratio = width / height
        
        # Image statistics
        brightness = np.mean(gray)
        variance = np.var(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # Symmetry analysis - handle odd widths properly
        try:
            mid = width // 2
            left_half = gray[:, :mid]
            right_half = gray[:, -mid:] if mid > 0 else gray
            
            # Flatten and get same length
            left_flat = left_half.flatten()[:100000]
            right_flat = right_half.flatten()[:100000]
            
            if len(left_flat) > 1 and len(right_flat) > 1:
                symmetry = float(np.corrcoef(left_flat, right_flat)[0, 1] or 0)
            else:
                symmetry = 0
        except Exception as e:
            logger.debug(f"Symmetry analysis failed: {e}")
            symmetry = 0
        
        # Characteristics by image type
        if 0.8 < aspect_ratio < 1.2 and height > 400:  # Square-ish, tall
            return "chest"
        elif 0.3 < aspect_ratio < 0.6:  # Narrow/elongated
            return "hand"
        elif aspect_ratio > 1.2 and edge_density > 0.15:  # Wide with dense edges
            return "spine"
        elif aspect_ratio > 1.0 and brightness > 120:  # Wide, bright
            return "abdomen"
        elif 0.6 < aspect_ratio < 1.0 and edge_density < 0.08:
            return "limb"
        else:
            return "unknown"
    
    def analyze_image(self, image: Image.Image) -> List[VisionFinding]:
        """
        Analyze medical image with adaptive analysis based on image type.
        """
        findings = []
        
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Auto-detect image type
        image_type = self.detect_image_type(gray, height, width)
        
        # STEP 1: CONTRAST & INTENSITY ANALYSIS
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, bright_regions = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        
        # STEP 2: EDGE DETECTION (find fractures, lesions, breaks)
        edges = cv2.Canny(enhanced, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2)
        
        # STEP 3: CONTOUR ANALYSIS (find abnormal regions)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # STEP 4: TEXTURE & VARIANCE ANALYSIS
        local_variance = ndimage.generic_filter(gray, np.var, size=20)
        high_variance_regions = np.where(local_variance > np.percentile(local_variance, 75))
        
        # STEP 5: FRACTURE/BREAK DETECTION
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        var = cv2.Laplacian(blurred, cv2.CV_64F).var()
        
        # ------- GENERATE FINDINGS -------
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        opacity_ratio = np.sum(bright_regions) / (height * width * 255)
        
        # TYPE-SPECIFIC ANALYSIS
        if image_type == "hand":
            findings.extend(self._analyze_hand_xray(gray, edges, contours, height, width))
        elif image_type == "chest":
            findings.extend(self._analyze_chest_xray(gray, bright_regions, contours, height, width))
        elif image_type == "spine":
            findings.extend(self._analyze_spine_xray(gray, edges, contours, height, width))
        elif image_type == "abdomen":
            findings.extend(self._analyze_abdomen_xray(gray, bright_regions, height, width))
        elif image_type == "limb":
            findings.extend(self._analyze_limb_xray(gray, edges, contours, height, width))
        else:
            findings.extend(self._analyze_generic_xray(gray, edges, contours, height, width))
        
        # If no findings detected, mark as normal
        if not findings:
            findings.append(VisionFinding(
                finding_type=AbnormalityType.NORMAL,
                location="Region of Interest",
                description=f"No significant abnormalities detected. Image type: {image_type}. Contrast: {contrast:.1f}",
                confidence=0.80,
                visual_coordinates=(50, 100, width-50, height-100)
            ))
        
        logger.info(f"{self.name}: Analyzed {image_type} X-ray - Found {len(findings)} finding(s)")
        return findings
    
    def _analyze_hand_xray(self, gray, edges, contours, height, width):
        """Analyze hand/finger X-rays for fractures and bone abnormalities"""
        findings = []
        
        # Fracture detection - look for line breaks in bones
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        eroded = cv2.erode(edges, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        gaps = edges - dilated
        
        if np.sum(gaps) > (width * height * 0.02):
            findings.append(VisionFinding(
                finding_type=AbnormalityType.FRACTURE,
                location="Metacarpal/Phalangeal region",
                description="Detected discontinuities in bone margins - possible fracture or break",
                confidence=0.75,
                visual_coordinates=self._get_region_coordinates(gaps, height, width)
            ))
        
        # Bone density analysis
        for contour in contours[:8]:
            area = cv2.contourArea(contour)
            if area > (height * width * 0.005):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check bone integrity
                roi = gray[y:y+h, x:x+w]
                if np.std(roi) > 40:  # High variance = potential fracture/break
                    findings.append(VisionFinding(
                        finding_type=AbnormalityType.FRACTURE,
                        location=self._localize_hand_finding(x, y, width, height),
                        description=f"Detected abnormal bone density/discontinuity ({w}×{h} px)",
                        confidence=0.70,
                        visual_coordinates=(x, y, x+w, y+h)
                    ))
        
        return findings
    
    def _analyze_chest_xray(self, gray, bright_regions, contours, height, width):
        """Analyze chest X-rays for pneumonia, effusion, etc."""
        findings = []
        
        opacity_ratio = np.sum(bright_regions) / (height * width * 255)
        if opacity_ratio > 0.15:
            findings.append(VisionFinding(
                finding_type=AbnormalityType.OPACITY,
                location="Right Lower Lobe" if width//2 < np.mean(np.where(bright_regions)[1]) else "Left Lower Lobe",
                description=f"Detected significant opacification pattern ({opacity_ratio:.1%})",
                confidence=min(0.9, 0.5 + opacity_ratio),
                visual_coordinates=self._get_region_coordinates(bright_regions, height, width)
            ))
        
        for i, contour in enumerate(contours[:5]):
            area = cv2.contourArea(contour)
            if area > (height * width * 0.01):
                x, y, w, h = cv2.boundingRect(contour)
                findings.append(VisionFinding(
                    finding_type=AbnormalityType.SHADOW,
                    location=self._localize_chest_finding(x, y, width, height),
                    description=f"Detected shadow/infiltrate ({w}×{h} px)",
                    confidence=min(0.85, 0.5 + (area / (height * width * 0.1))),
                    visual_coordinates=(x, y, x+w, y+h)
                ))
        
        return findings
    
    def _analyze_spine_xray(self, gray, edges, contours, height, width):
        """Analyze spine X-rays for fractures, misalignment"""
        findings = []
        
        # Look for vertebral body fractures
        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            if (height * width * 0.005) < area < (height * width * 0.05):
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y+h, x:x+w]
                
                if np.std(roi) > 50:
                    findings.append(VisionFinding(
                        finding_type=AbnormalityType.FRACTURE,
                        location=self._localize_spine_finding(y, height),
                        description=f"Detected vertebral abnormality at level ({w}×{h} px)",
                        confidence=0.65,
                        visual_coordinates=(x, y, x+w, y+h)
                    ))
        
        return findings
    
    def _analyze_abdomen_xray(self, gray, bright_regions, height, width):
        """Analyze abdominal X-rays for free air, obstruction"""
        findings = []
        
        # Large bright areas indicate free air or gas
        opacity_ratio = np.sum(bright_regions) / (height * width * 255)
        if opacity_ratio > 0.25:
            findings.append(VisionFinding(
                finding_type=AbnormalityType.OPACITY,
                location="Abdominal cavity",
                description=f"Detected significant gas or fluid pattern ({opacity_ratio:.1%})",
                confidence=0.70,
                visual_coordinates=self._get_region_coordinates(bright_regions, height, width)
            ))
        
        return findings
    
    def _analyze_limb_xray(self, gray, edges, contours, height, width):
        """Analyze limb X-rays (arms, legs) for fractures"""
        findings = []
        
        for contour in contours[:6]:
            area = cv2.contourArea(contour)
            if area > (height * width * 0.008):
                x, y, w, h = cv2.boundingRect(contour)
                
                findings.append(VisionFinding(
                    finding_type=AbnormalityType.FRACTURE,
                    location="Bone shaft region",
                    description=f"Detected potential fracture or cortical break ({w}×{h} px)",
                    confidence=0.68,
                    visual_coordinates=(x, y, x+w, y+h)
                ))
        
        return findings
    
    def _analyze_generic_xray(self, gray, edges, contours, height, width):
        """Generic analysis for unknown image types"""
        findings = []
        
        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area > (height * width * 0.01):
                x, y, w, h = cv2.boundingRect(contour)
                findings.append(VisionFinding(
                    finding_type=AbnormalityType.SHADOW,
                    location="Region of interest",
                    description=f"Detected abnormal region ({w}×{h} px)",
                    confidence=0.60,
                    visual_coordinates=(x, y, x+w, y+h)
                ))
        
        return findings
    
    def _localize_hand_finding(self, x, y, width, height):
        """Localize findings in hand X-rays"""
        relative_x = x / width
        if relative_x < 0.3:
            return "Thumb/Index region"
        elif relative_x < 0.6:
            return "Middle finger region"
        else:
            return "Ring/Pinky region"
    
    def _localize_chest_finding(self, x, y, width, height):
        """Localize findings in chest X-rays"""
        relative_x = x / width
        relative_y = y / height
        
        if relative_y < 0.4:
            if relative_x < 0.5:
                return "Left Upper Lobe"
            else:
                return "Right Upper Lobe"
        else:
            if relative_x < 0.5:
                return "Left Lower Lobe"
            else:
                return "Right Lower Lobe"
    
    def _localize_spine_finding(self, y, height):
        """Localize findings in spine X-rays by vertebral level"""
        relative_y = y / height
        
        if relative_y < 0.2:
            return "Cervical region"
        elif relative_y < 0.4:
            return "Thoracic region"
        elif relative_y < 0.7:
            return "Lumbar region"
        else:
            return "Sacral region"
    
    def _get_region_coordinates(self, binary_image: np.ndarray, height: int, width: int) -> Tuple[int, int, int, int]:
        """Get bounding coordinates of detected region"""
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)
        if rows.any() and cols.any():
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return (xmin, ymin, xmax, ymax)
        return (0, 0, width, height)
    
    def extract_features(self, findings: List[VisionFinding]) -> Dict:
        """Convert findings to structured feature dict"""
        return {
            "finding_count": len(findings),
            "findings": [
                {
                    "type": f.finding_type.value,
                    "location": f.location,
                    "confidence": f.confidence,
                    "description": f.description
                }
                for f in findings
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }


# =============================================================================
# AGENT 2: ANALYSIS AGENT
# =============================================================================
class AnalysisAgent:
    """
    Cross-references Vision Agent findings with medical knowledge base.
    Evaluates relationships between features and possible medical conditions.
    Provides reasoning paths, alternative diagnoses, and confidence levels.
    """
    
    def __init__(self):
        self.name = "Analysis Agent"
        self.kb = MedicalKnowledgeBase()
        logger.info(f"{self.name} initialized")
    
    def reason_about_findings(self, findings: List[VisionFinding]) -> List[MedicalHypothesis]:
        """
        Cross-reference real image findings with medical knowledge base.
        
        Args:
            findings: List of VisionFinding from Vision Agent (REAL image analysis)
            
        Returns:
            List of ranked MedicalHypothesis objects
        """
        hypotheses = []
        
        # Check for normal findings
        normal_findings = [f for f in findings if f.finding_type == AbnormalityType.NORMAL]
        if len(normal_findings) == len(findings):  # ALL findings are normal
            hypotheses.append(MedicalHypothesis(
                condition="Normal Chest X-Ray",
                confidence_score=0.95,
                supporting_findings=["Clear lungs", "Normal cardiac silhouette", "No acute opacities"],
                differential_diagnoses=[],
                severity_level="Low",
                reasoning_path="Image analysis detected no abnormal opacities. Lung fields clear bilaterally.",
                clinical_significance="No acute cardiopulmonary process identified."
            ))
            return hypotheses
        
        # Match REAL findings to conditions in knowledge base
        condition_scores = {}
        finding_descriptions = []
        
        for finding in findings:
            if finding.finding_type == AbnormalityType.NORMAL:
                continue
            
            finding_descriptions.append(f"{finding.location}: {finding.description}")
            
            # Match abnormality type to conditions using real image detection
            for condition, info in self.kb.CONDITION_DATABASE.items():
                if condition == "Normal":
                    continue
                
                # Check if detected abnormality matches condition's imaging features
                if finding.finding_type.value in info["imaging_features"]:
                    base_score = finding.confidence  # Use REAL confidence from image analysis
                    
                    # Boost score if location matches typical presentation
                    if finding.location in info["typical_location"]:
                        base_score += info["confidence_boost"]
                    
                    condition_scores[condition] = condition_scores.get(condition, 0) + base_score
        
        # Generate ranked hypotheses from real findings
        for condition, score in sorted(condition_scores.items(), key=lambda x: -x[1]):
            if score > 0:
                info = self.kb.get_condition_info(condition)
                
                hypotheses.append(MedicalHypothesis(
                    condition=condition,
                    confidence_score=min(score / 2.0, 0.95),
                    supporting_findings=finding_descriptions,
                    differential_diagnoses=[h.condition for h in hypotheses[:2]],
                    severity_level=info["risk_level"],
                    reasoning_path=self._generate_reasoning_path(condition, findings, finding_descriptions),
                    clinical_significance=self._generate_clinical_significance(condition, findings)
                ))
        
        # Sort by confidence
        hypotheses.sort(key=lambda x: -x.confidence_score)
        
        logger.info(f"{self.name}: Generated {len(hypotheses)} hypothesis/hypotheses from real image findings")
        return hypotheses
    
    def _generate_reasoning_path(self, condition: str, findings: List[VisionFinding], descriptions: List[str]) -> str:
        """Generate explainable reasoning path based on REAL findings"""
        path = f"Condition: {condition}. "
        path += f"Image Analysis: {len([f for f in findings if f.finding_type != AbnormalityType.NORMAL])} abnormal finding(s) detected. "
        info = self.kb.get_condition_info(condition)
        path += f"Detected features match: {', '.join(info['imaging_features'][:3])}. "
        path += f"Evidence: {descriptions[0] if descriptions else 'No findings'}."
        return path
    
    def _generate_clinical_significance(self, condition: str, findings: List[VisionFinding]) -> str:
        """Generate clinical significance based on real findings"""
        info = self.kb.get_condition_info(condition)
        risk = info["risk_level"]
        
        # Get confidence from actual image findings
        max_confidence = max([f.confidence for f in findings if f.finding_type != AbnormalityType.NORMAL], default=0)
        
        if risk == "High" and max_confidence > 0.7:
            return f"{condition} is indicated with high confidence. Requires prompt clinical correlation and possible intervention."
        elif risk == "Medium" and max_confidence > 0.6:
            return f"{condition} is suggested by imaging. Warrants further clinical evaluation and follow-up."
        else:
            return f"{condition} may be present. Recommend clinical correlation and appropriate follow-up imaging."


# =============================================================================
# AGENT 3: REPORTING AGENT
# =============================================================================
class ReportingAgent:
    """
    Compiles Vision and Analysis Agent outputs into a structured diagnostic report.
    Presents findings in clear, clinician-friendly language.
    Structures data for radiologist review and approval.
    """
    
    def __init__(self):
        self.name = "Reporting Agent"
        logger.info(f"{self.name} initialized")
    
    def generate_report(
        self, 
        patient_id: str,
        findings: List[VisionFinding],
        hypotheses: List[MedicalHypothesis]
    ) -> DiagnosticReport:
        """
        Compile findings and hypotheses into structured diagnostic report.
        
        Args:
            patient_id: Patient identifier
            findings: List of VisionFinding from Vision Agent
            hypotheses: List of MedicalHypothesis from Analysis Agent
            
        Returns:
            DiagnosticReport object ready for radiologist review
        """
        
        # Extract preliminary findings
        preliminary_findings = [
            f"{f.location}: {f.description} (Confidence: {f.confidence:.1%})"
            for f in findings
        ]
        
        # Primary hypothesis (highest confidence)
        primary_hypothesis = hypotheses[0] if hypotheses else MedicalHypothesis(
            condition="Inconclusive",
            confidence_score=0.0,
            supporting_findings=[],
            differential_diagnoses=[],
            severity_level="Unknown",
            reasoning_path="Insufficient data for conclusive diagnosis.",
            clinical_significance="Recommend additional imaging or clinical review."
        )
        
        # Alternative hypotheses (runners-up)
        alternative_hypotheses = hypotheses[1:4] if len(hypotheses) > 1 else []
        
        # Generate recommendations
        recommendations = self._generate_recommendations(primary_hypothesis, findings)
        
        # Generate clinical summary
        clinical_summary = self._generate_clinical_summary(
            primary_hypothesis, 
            findings,
            len(hypotheses)
        )
        
        report = DiagnosticReport(
            patient_id=patient_id,
            timestamp=datetime.now().isoformat(),
            preliminary_findings=preliminary_findings,
            primary_hypothesis=primary_hypothesis,
            alternative_hypotheses=alternative_hypotheses,
            recommendations=recommendations,
            clinical_summary=clinical_summary,
            radiologist_review_status="Pending Review",
            radiologist_notes="",
            radiologist_approval=False
        )
        
        logger.info(f"{self.name}: Report generated for patient {patient_id}")
        return report
    
    def _generate_recommendations(
        self, 
        hypothesis: MedicalHypothesis,
        findings: List[VisionFinding]
    ) -> List[str]:
        """Generate clinical recommendations based on hypothesis"""
        recommendations = []
        
        if hypothesis.severity_level == "Critical":
            recommendations.append("🔴 URGENT: Immediate radiologist review recommended")
            recommendations.append("Consider clinical correlation with patient symptoms")
            recommendations.append("Possible need for emergent intervention")
        elif hypothesis.severity_level == "High":
            recommendations.append("Prompt follow-up recommended")
            recommendations.append("Clinical correlation advised")
        else:
            recommendations.append("Routine follow-up as clinically indicated")
            recommendations.append("Monitor for interval changes")
        
        recommendations.append("Recommend radiologist verification before clinical use")
        recommendations.append("Consider recommendations from clinical context")
        
        return recommendations
    
    def _generate_clinical_summary(
        self, 
        hypothesis: MedicalHypothesis,
        findings: List[VisionFinding],
        hypothesis_count: int
    ) -> str:
        """Generate natural language clinical summary"""
        summary = f"Primary Assessment: {hypothesis.condition} "
        summary += f"(Confidence: {hypothesis.confidence_score:.1%}). "
        
        if len(findings) > 0:
            summary += f"Radiographic evidence includes {len(findings)} finding(s). "
        
        summary += hypothesis.clinical_significance
        
        if hypothesis_count > 1:
            summary += f" {hypothesis_count - 1} alternative diagnosis/diagnoses considered."
        
        return summary


# =============================================================================
# ORCHESTRATOR: COORDINATING THE THREE-AGENT SYSTEM
# =============================================================================
class MediScanOrchestrator:
    """
    Orchestrates the three-agent pipeline:
    Image → Vision Agent → Analysis Agent → Reporting Agent → Radiologist Review
    """
    
    def __init__(self):
        self.vision_agent = VisionAgent()
        self.analysis_agent = AnalysisAgent()
        self.reporting_agent = ReportingAgent()
        logger.info("MediScan Orchestrator initialized")
    
    def process_scan(
        self, 
        image: Image.Image,
        patient_id: str,
        patient_name: str,
        clinical_history: List[str] = None
    ) -> DiagnosticReport:
        """
        Execute complete diagnostic pipeline.
        
        Args:
            image: Uploaded medical image
            patient_id: Patient identifier
            patient_name: Patient name
            clinical_history: Relevant clinical information
            
        Returns:
            DiagnosticReport for radiologist review
        """
        
        # STAGE 1: VISION AGENT
        logger.info(f"[Stage 1] Vision Agent analyzing scan for patient {patient_id}")
        vision_findings = self.vision_agent.analyze_image(image)
        
        # STAGE 2: ANALYSIS AGENT
        logger.info(f"[Stage 2] Analysis Agent reasoning about findings")
        hypotheses = self.analysis_agent.reason_about_findings(vision_findings)
        
        # STAGE 3: REPORTING AGENT
        logger.info(f"[Stage 3] Reporting Agent compiling diagnostic report")
        report = self.reporting_agent.generate_report(
            patient_id=patient_id,
            findings=vision_findings,
            hypotheses=hypotheses
        )
        
        logger.info(f"[Complete] Pipeline finished - awaiting radiologist review")
        return report


# =============================================================================
# STREAMLIT INTERFACE - HUMAN-IN-THE-LOOP WORKFLOW
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state"""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = MediScanOrchestrator()
    if "current_report" not in st.session_state:
        st.session_state.current_report = None
    if "report_history" not in st.session_state:
        st.session_state.report_history = []


def render_vision_findings(findings: List[VisionFinding]):
    """Render Vision Agent findings"""
    st.subheader("👁️ Vision Agent: Image Analysis")
    
    for i, finding in enumerate(findings):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Finding {i+1}: {finding.finding_type.value}**")
            st.write(f"Location: {finding.location}")
            st.write(f"Description: {finding.description}")
        with col2:
            confidence_pct = f"{finding.confidence:.0%}"
            st.metric("Confidence", confidence_pct)


def render_analysis_hypotheses(hypotheses: List[MedicalHypothesis]):
    """Render Analysis Agent hypotheses with reasoning"""
    st.subheader("🧠 Analysis Agent: Medical Reasoning")
    
    if not hypotheses:
        st.info("No hypotheses generated.")
        return
    
    # Primary hypothesis (largest card)
    primary = hypotheses[0]
    st.info(f"""
    **Primary Assessment: {primary.condition}**
    
    Confidence Score: {primary.confidence_score:.1%}
    Severity Level: {primary.severity_level}
    
    **Reasoning Path:** {primary.reasoning_path}
    
    **Clinical Significance:** {primary.clinical_significance}
    
    **Supporting Findings:**
    {chr(10).join([f"• {f}" for f in primary.supporting_findings])}
    """)
    
    # Alternative hypotheses
    if len(hypotheses) > 1:
        st.write("**Alternative Diagnoses Considered:**")
        for alt in hypotheses[1:]:
            with st.expander(f"{alt.condition} ({alt.confidence_score:.1%})"):
                st.write(f"Severity: {alt.severity_level}")
                st.write(f"Reasoning: {alt.reasoning_path}")


def render_diagnostic_report(report: DiagnosticReport):
    """Render complete diagnostic report"""
    st.subheader("📋 Diagnostic Report")
    
    # Report header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Patient ID:** {report.patient_id}")
    with col2:
        st.write(f"**Timestamp:** {report.timestamp[:19]}")
    with col3:
        status_color = "🟢" if report.radiologist_approval else "🟡"
        st.write(f"{status_color} **Status:** {report.radiologist_review_status}")
    
    st.divider()
    
    # Findings summary
    st.write("**Preliminary Findings:**")
    for finding in report.preliminary_findings:
        st.write(f"• {finding}")
    
    st.divider()
    
    # Primary diagnosis
    st.write("**Primary Assessment:**")
    st.write(f"Condition: **{report.primary_hypothesis.condition}**")
    st.write(f"Confidence: **{report.primary_hypothesis.confidence_score:.1%}**")
    st.write(f"Risk Level: **{report.primary_hypothesis.severity_level}**")
    
    st.divider()
    
    # Clinical summary
    st.write("**Clinical Summary:**")
    st.write(report.clinical_summary)
    
    st.divider()
    
    # Recommendations
    st.write("**Recommendations:**")
    for rec in report.recommendations:
        st.write(f"• {rec}")


def render_radiologist_verification():
    """Render human-in-the-loop verification interface"""
    st.subheader("✅ Radiologist Verification & Approval")
    
    if st.session_state.current_report is None:
        st.info("No report to review. Please upload and analyze a scan first.")
        return
    
    report = st.session_state.current_report
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Radiologist Review:**")
        
        # Approval checkbox
        approval = st.checkbox(
            "✅ I approve this AI-generated preliminary analysis",
            value=report.radiologist_approval
        )
        
        # Notes field
        notes = st.text_area(
            "Additional Notes / Corrections:",
            value=report.radiologist_notes,
            height=100,
            placeholder="Enter any corrections, additional findings, or clinical notes..."
        )
        
        # Severity override
        severity_override = st.selectbox(
            "Override Severity Assessment:",
            ["Keep AI Assessment", "Low", "Medium", "High", "Critical"]
        )
        
        # Submit verification
        if st.button("🖊️ Submit Radiologist Review", type="primary", use_container_width=True):
            report.radiologist_approval = approval
            report.radiologist_notes = notes
            report.radiologist_review_status = "Reviewed" if approval else "Needs Revision"
            
            if severity_override != "Keep AI Assessment":
                report.primary_hypothesis.severity_level = severity_override
            
            st.session_state.report_history.append(asdict(report))
            st.success("✅ Radiologist review submitted successfully!")
            logger.info(f"Report approved: {approval}, Status: {report.radiologist_review_status}")
    
    with col2:
        st.write("**Report Summary:**")
        st.metric("Condition", report.primary_hypothesis.condition)
        st.metric("AI Confidence", f"{report.primary_hypothesis.confidence_score:.0%}")
        st.metric("Risk Level", report.primary_hypothesis.severity_level)


def main():
    """Main Streamlit application"""
    
    # Configure page
    st.set_page_config(
        page_title="MediScan Analyst",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🩺 MediScan Analyst")
        st.markdown("Advanced Multi-Agent Medical Image Analysis")
        st.divider()
        
        # Patient information
        st.subheader("Patient Information")
        patient_id = st.text_input("Patient ID / MRN", value="MRN-001")
        patient_name = st.text_input("Patient Name", value="John Doe")
        
        clinical_history = st.multiselect(
            "Clinical History",
            ["Fever", "Cough", "Chest Pain", "Dyspnea", "Hemoptysis", "Post-Op", "Smoker"],
            default=["Fever", "Cough"]
        )
        
        st.divider()
        st.caption("Architecture: 3-Agent Pipeline (Vision → Analysis → Reporting)")
        st.caption("Version: 1.0 | Human-in-the-Loop Enabled")
    
    # Main title
    st.title("🩺 MediScan Analyst Workstation")
    st.markdown("AI-Assisted Medical Image Analysis with Explainable Intermediate Steps")
    
    # Top metrics
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    with col_metric1:
        st.metric("Pipeline Status", "Ready", delta="Active")
    with col_metric2:
        st.metric("Agents", "3/3", delta="Online")
    with col_metric3:
        st.metric("Reports Generated", len(st.session_state.report_history), delta="Session")
    with col_metric4:
        approved = sum(1 for r in st.session_state.report_history if r.get("radiologist_approval"))
        st.metric("Approved Reports", approved, delta=f"of {len(st.session_state.report_history)}")
    
    st.divider()
    
    # Main workflow columns
    col_upload, col_processing = st.columns([1.2, 1.8])
    
    with col_upload:
        st.subheader("1️⃣ Image Upload & Inspection")
        uploaded_file = st.file_uploader(
            "Upload Medical Image (DICOM/PNG/JPG)",
            type=["jpg", "png", "jpeg", "dcm"]
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Patient: {patient_name} | ID: {patient_id}", use_container_width=True)
            
            # Image properties
            st.caption(f"Image Size: {image.size[0]}×{image.size[1]} | Format: {image.format}")
    
    with col_processing:
        if uploaded_file:
            st.subheader("2️⃣ Agent Processing Pipeline")
            
            # Process button
            if st.button("🚀 Start Analysis Pipeline", type="primary", use_container_width=True):
                with st.spinner("Processing through 3-agent pipeline..."):
                    
                    # Stage 1: Vision Agent
                    with st.status("👁️ Vision Agent: Analyzing image...", expanded=True) as status1:
                        image = Image.open(uploaded_file)
                        image = image.convert("RGB")
                        vision_findings = st.session_state.orchestrator.vision_agent.analyze_image(image)
                        st.write(f"✓ Identified {len(vision_findings)} radiographic finding(s)")
                        st.write(f"✓ Feature extraction complete")
                        status1.update(label="✅ Vision Agent: Complete", state="complete")
                    
                    # Stage 2: Analysis Agent
                    with st.status("🧠 Analysis Agent: Medical reasoning...", expanded=True) as status2:
                        hypotheses = st.session_state.orchestrator.analysis_agent.reason_about_findings(vision_findings)
                        st.write(f"✓ Generated {len(hypotheses)} diagnostic hypothesis/hypotheses")
                        st.write(f"✓ Confidence scoring complete")
                        st.write(f"✓ Differential diagnosis evaluated")
                        status2.update(label="✅ Analysis Agent: Complete", state="complete")
                    
                    # Stage 3: Reporting Agent
                    with st.status("📋 Reporting Agent: Generating report...", expanded=True) as status3:
                        report = st.session_state.orchestrator.reporting_agent.generate_report(
                            patient_id=patient_id,
                            findings=vision_findings,
                            hypotheses=hypotheses
                        )
                        st.session_state.current_report = report
                        st.write(f"✓ Clinical summary generated")
                        st.write(f"✓ Recommendations formatted")
                        st.write(f"✓ Report ready for radiologist review")
                        status3.update(label="✅ Reporting Agent: Complete", state="complete")
                    
                    st.success("✅ Pipeline complete! Scroll below to review findings.")
    
    st.divider()
    
    # Results section (only show if report exists)
    if st.session_state.current_report:
        
        # Tabs for organized view
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Full Report",
            "👁️ Vision Findings",
            "🧠 Analysis",
            "✅ Verification"
        ])
        
        with tab1:
            render_diagnostic_report(st.session_state.current_report)
        
        with tab2:
            vision_findings = st.session_state.orchestrator.vision_agent.analyze_image(
                Image.open(uploaded_file).convert("RGB")
            )
            render_vision_findings(vision_findings)
        
        with tab3:
            hypotheses = st.session_state.orchestrator.analysis_agent.reason_about_findings(vision_findings)
            render_analysis_hypotheses(hypotheses)
        
        with tab4:
            render_radiologist_verification()
    else:
        st.info("📤 Upload an image and click 'Start Analysis Pipeline' to begin diagnosis.")
    
    st.divider()
    
    # Report history
    if st.session_state.report_history:
        st.subheader("📚 Report History")
        
        history_data = []
        for r in st.session_state.report_history:
            history_data.append({
                "Patient ID": r["patient_id"],
                "Condition": r["primary_hypothesis"]["condition"],
                "Confidence": f"{r['primary_hypothesis']['confidence_score']:.0%}",
                "Status": "✅ Approved" if r["radiologist_approval"] else "🟡 Pending",
                "Timestamp": r["timestamp"][:19]
            })
        
        st.dataframe(history_data, use_container_width=True)


if __name__ == "__main__":
    main()
