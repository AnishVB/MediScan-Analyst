# MediScan Analyst Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser / Dashboard                       │
│  (Vanilla JavaScript, localStorage, Responsive UI)          │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         MedicalImageAnalyzer                          │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │ Image Type Detection                             │ │  │
│  │  │ - Chest X-ray Detection                          │ │  │
│  │  │ - Hand/Skeletal Detection                        │ │  │
│  │  │ - Brain MRI/CT Detection                         │ │  │
│  │  │ - Spine Radiograph Detection                     │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │ Finding Extraction                               │ │  │
│  │  │ - CLAHE Image Enhancement                        │ │  │
│  │  │ - Edge Detection (Canny)                         │ │  │
│  │  │ - Contour Analysis                               │ │  │
│  │  │ - Structure Segmentation                         │ │  │
│  │  │ - Abnormality Detection                          │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │ Medical Knowledge Integration                    │ │  │
│  │  │ - Anatomical Structure Mapping                   │ │  │
│  │  │ - Finding Classification                         │ │  │
│  │  │ - Confidence Scoring                             │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  API Endpoints:                                              │
│  - POST /api/analyze (image processing)                      │
│  - GET /api/health (system status)                           │
│  - GET / (dashboard delivery)                                │
└──────────────────────────────────────────────────────────────┘
                     │ HTML + JS
┌────────────────────▼────────────────────────────────────────┐
│              Client-Side Storage (localStorage)              │
│  - Patients Array                                            │
│  - Reports Array                                             │
│  - Analysis History                                          │
└──────────────────────────────────────────────────────────────┘
```

## Component Details

### Backend: MedicalImageAnalyzer

**Image Type Detection**

- Analyzes height/width aspect ratio
- Measures contrast and brightness levels
- Computes edge density
- Detects histogram patterns
- Returns confidence scores for each image type

**Finding Extraction Process**

1. Convert image to grayscale
2. Apply CLAHE contrast enhancement
3. Perform edge detection (Canny algorithm)
4. Extract contours
5. Route to image-type-specific analyzer
6. Generate findings with locations and descriptions

**Image-Type Specific Analysis**

_Chest X-ray_:

- Detects lung field boundaries
- Analyzes cardiac silhouette
- Identifies opacity patterns
- Checks costophrenic angles
- Confidence per finding: 0.65-0.85

_Hand/Skeletal_:

- Counts bone structures
- Detects fracture lines
- Assesses bone density
- Confidence per finding: 0.6-0.8

_Brain MRI/CT_:

- Analyzes bilateral symmetry
- Checks ventricular system
- Assesses gray/white matter
- Confidence per finding: 0.75-0.8

_Spine Radiograph_:

- Counts vertebral bodies
- Examines disc spaces
- Analyzes spinal alignment
- Confidence per finding: 0.75-0.85

### Frontend: Dashboard Application

**Pages**

- Dashboard: System overview, statistics
- Patients: Patient management, search
- Patient Profile: Individual patient records, history
- Analysis: Medical image upload and processing
- Reports: Historical reports and findings
- Settings: System configuration

**Data Flow**

1. User uploads image
2. Frontend sends FormData to `/api/analyze`
3. Backend analyzes image, returns findings
4. Results displayed in Analysis page
5. User can save analysis to patient record
6. Data persisted to localStorage
7. Reports accessible from Reports page

## Data Structures

### Patient Object

```javascript
{
  name: string,
  mrn: string,
  age: number,
  dob: string,
  notes: string,
  scans: [ScanObject]
}
```

### Scan Object

```javascript
{
  timestamp: ISO string,
  image_type: string,
  body_part: string,
  analysis: AnalysisObject,
  ensemble_confidence: float,
  notes: string
}
```

### Analysis Object

```javascript
{
  findings: [FindingObject],
  primary_hypothesis: HypothesisObject,
  differential_diagnoses: [HypothesisObject],
  clinical_significance: string,
  recommendations: [string]
}
```

## Processing Pipeline

```
Image Upload
    ↓
File Validation
    ↓
Image Loading & Normalization
    ↓
Type Detection
    ↓
Image Enhancement (CLAHE)
    ↓
Edge Detection (Canny)
    ↓
Contour Extraction
    ↓
Type-Specific Analysis
    ↓
Finding Generation
    ↓
Confidence Scoring
    ↓
Report Generation
    ↓
Display Results
    ↓
User Save Decision
    ↓
Persist to localStorage
```

## Performance Considerations

- **Processing Time**: ~0.7s per image
- **Image Size Limit**: 50MB (resized to 1024x1024 max)
- **Memory**: Minimal (no GPU required)
- **Scalability**: Can handle multiple concurrent analyses

## Error Handling

- Invalid image format rejection
- Graceful fallback for edge cases
- User-friendly error messages
- Automatic retry mechanisms

## Security

- Client-side data storage only
- No external API calls
- CORS configured for development
- Input validation on all endpoints
