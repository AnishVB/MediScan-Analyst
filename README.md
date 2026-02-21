# MediScan Analyst v1.0

Professional medical image analysis system with real-time findings extraction and clinical decision support.

## Features

- **Medical Image Analysis**: Detects chest X-rays, hand/skeletal imaging, brain MRI/CT, and spine radiographs
- **Real-time Findings Extraction**: Identifies anatomical structures, abnormalities, and potential pathologies
- **Patient Management**: Complete patient record tracking with scan history
- **Clinical Reports**: Generates detailed analysis reports with recommendations
- **Professional UI**: Clean, intuitive interface designed for medical professionals

## Quick Start

1. **Start Backend**:

   ```bash
   python backend.py
   ```

2. **Access Dashboard**:
   - Open browser to `http://localhost:8000/`

3. **Add Patient**:
   - Navigate to Patients page
   - Click "Add Patient" button
   - Enter patient information

4. **Analyze Image**:
   - Select patient
   - Click "Analyze" button
   - Upload medical image
   - System extracts findings and generates report

## Architecture

### Backend (FastAPI)

- **MedicalImageAnalyzer**: Computer vision-based image analysis
- **Image Type Detection**: Classifies imaging modality
- **Finding Extraction**: Identifies anatomical structures and abnormalities
- **REST API**: `/api/analyze` for image processing

### Frontend (Vanilla JavaScript)

- **Dashboard**: System overview and statistics
- **Patients**: Patient management and records
- **Analysis**: Medical image upload and analysis
- **Reports**: Historical analysis and findings
- **Settings**: System configuration

## Supported Imaging Types

- **Chest X-ray**: Lung fields, cardiac silhouette, costophrenic angles
- **Hand/Skeletal**: Bone structures, fracture detection
- **Brain MRI/CT**: Symmetry analysis, ventricular system
- **Spine Radiograph**: Vertebrae, disc spaces, alignment

## API Endpoints

- `GET /api/health` - System health check
- `POST /api/analyze` - Analyze medical image
- `GET /` - Serve dashboard

## Data Storage

- Patient records stored in browser localStorage
- Scan history maintained per patient
- Clinical notes automatically saved

## Requirements

- Python 3.8+
- FastAPI
- Pillow
- OpenCV
- NumPy
