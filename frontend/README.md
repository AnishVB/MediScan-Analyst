# MediScan Analyst

**Agentic AI Co-pilot for Medical Imaging & Diagnosis Support**

MediScan Analyst is a full-stack web application that analyses medical images (X-rays, CT/MRI scans) using a **three-agent AI pipeline** and presents structured diagnostic reports. It also includes a patient management system with scan history tracking and specialist referrals.

> [!IMPORTANT]
> This tool is for **educational / assistive purposes only**. All findings must be reviewed and confirmed by a qualified radiologist or physician before any clinical use.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Three-Agent Pipeline](#three-agent-pipeline)
3. [Medical Knowledge Base](#medical-knowledge-base)
4. [Patient Management](#patient-management)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [API Endpoints](#api-endpoints)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                   │
│  Landing Page → Dashboard → Image Upload → Diagnostic Report│
│              Patient List / Profile / Referrals             │
└───────────────────────────┬─────────────────────────────────┘
                            │  REST API (fetch)
┌───────────────────────────▼─────────────────────────────────┐
│                  FastAPI Backend (Python)                    │
│                                                             │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────────┐   │
│   │  Vision   │──▶│  Analysis    │──▶│  Reporting Agent  │   │
│   │  Agent    │   │  Agent       │   │                  │   │
│   └──────────┘   └──────┬───────┘   └──────────────────┘   │
│                         │                                   │
│               Knowledge Base (dict)                         │
│               SQLite  (mediscan.db)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Three-Agent Pipeline

When a medical image is uploaded, it flows through three sequential agents:

### 1. Vision Agent (`VisionAgent`)

Responsible for **low-level image processing**. It runs entirely on OpenCV + NumPy — no external ML model is required.

| Step                         | What it does                                                                                                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pre-processing**           | Converts to grayscale, applies CLAHE (Contrast-Limited Adaptive Histogram Equalisation) for enhancement                                                                       |
| **Body-part classification** | Heuristic scoring based on aspect ratio, contrast, brightness, and edge density to classify the image as _Chest / Hand / Brain / Spine / General_                             |
| **Feature extraction**       | Computes quantitative metrics: contrast, brightness, edge density, opacity ratio, symmetry score, texture variance, discontinuity index, and quadrant density distribution    |
| **Structure detection**      | Uses Canny edge detection + contour analysis to identify anatomical structures (e.g. Lung Fields, Cardiac Silhouette, Skeletal Structures) specific to the detected body part |

**Output →** A dictionary containing `image_type`, `body_part`, extracted `features`, and detected `structures`.

---

### 2. Analysis Agent (`AnalysisAgent`)

Takes the Vision Agent's output and cross-references it against the **Medical Knowledge Base**.

| Step                            | What it does                                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Finding generation**          | Maps detected structures to clinical findings using feature thresholds (e.g. opacity > 0.2 suggests consolidation in Lung Fields)                             |
| **Differential diagnosis**      | Scores each known condition by matching its indicators against extracted features; produces a ranked list of possible diagnoses with confidence probabilities |
| **Clinical significance**       | Assesses urgency based on the primary diagnosis risk level and probability                                                                                    |
| **Recommendations**             | Generates contextual clinical recommendations (e.g. correlate with symptoms, consider follow-up imaging)                                                      |
| **Patient history integration** | If a patient is linked, it pulls their medical history and past scan findings to adjust diagnostic probabilities                                              |

**Output →** Findings, primary hypothesis, differential diagnoses, clinical significance, and recommendations.

---

### 3. Reporting Agent (`ReportingAgent`)

Compiles outputs from the Vision and Analysis agents into a **structured diagnostic report**.

| Step                   | What it does                                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Report assembly**    | Merges image classification, findings summary, primary assessment, differentials, recommendations, and quality notes into one JSON report |
| **Narrative summary**  | Converts individual findings into a readable narrative string                                                                             |
| **Quality assessment** | Rates overall analysis confidence (high / moderate / limited)                                                                             |
| **Disclaimer**         | Attaches a mandatory AI-disclaimer to every report                                                                                        |

**Output →** The final JSON payload sent back to the frontend for display.

---

## Medical Knowledge Base

The backend embeds a structured knowledge base (`KNOWLEDGE_BASE` dict) covering five body regions:

| Body Region | Conditions Detected                                                      | Anatomical Structures                                                                              |
| ----------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **Chest**   | Pneumonia, Pleural Effusion, Cardiomegaly, Pulmonary Edema, Normal Study | Lung fields, Cardiac silhouette, Mediastinum, Costophrenic angles, Diaphragm, Trachea, Aortic arch |
| **Hand**    | Fracture, Osteoporosis, Arthritis, Normal Study                          | Phalanges, Metacarpals, Carpals, Radius, Ulna, Joint spaces                                        |
| **Brain**   | Mass Effect, Hemorrhage, Hydrocephalus, Normal Study                     | Cerebral hemispheres, Ventricles, Midline structures, Gray/White matter, Cerebellum                |
| **Spine**   | Disc Herniation, Compression Fracture, Scoliosis, Normal Study           | Vertebral bodies, Intervertebral discs, Pedicles, Spinous processes, Spinal canal                  |
| **General** | General Assessment                                                       | Soft tissue, Bony structures, Air spaces                                                           |

Each condition includes diagnostic **indicators**, a description, and a **risk level** (low / moderate / high).

---

## Patient Management

A full patient database is backed by **SQLite** (`mediscan.db`), managed through `database.py`.

### Database Tables

| Table       | Purpose                                                                |
| ----------- | ---------------------------------------------------------------------- |
| `patients`  | Stores name, age, gender, blood group, medical conditions, notes       |
| `scans`     | Links uploaded images to patients, stores analysis JSON and confidence |
| `referrals` | Tracks specialist referral records per patient/scan                    |

### Key Features

- **CRUD operations** for patients, scans, and referrals
- **History-aware analysis** — the Analysis Agent receives a patient's past scan summaries and known conditions, boosting diagnostic accuracy
- **Referral management** — create referrals to specialists directly from scan results

---

## Tech Stack

| Layer                | Technology                      |
| -------------------- | ------------------------------- |
| **Frontend**         | React 19 · Vite 7 · Vanilla CSS |
| **Backend**          | Python · FastAPI · Uvicorn      |
| **Image Processing** | OpenCV · NumPy · Pillow         |
| **Database**         | SQLite 3 (via Python `sqlite3`) |
| **Data Validation**  | Pydantic                        |

---

## Project Structure

```
MediScan Analyst/
├── backend.py            # FastAPI server + three-agent pipeline
├── database.py           # SQLite schema & CRUD helpers
├── mediscan.db           # SQLite database file
├── uploads/              # Stored uploaded scan images
├── test scans/           # Sample images for testing
└── frontend/             # React + Vite application
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx                  # React entry point
        ├── App.jsx                   # Root component & routing
        ├── index.css                 # Global styles
        └── components/
            ├── LandingPage.jsx       # Hero / welcome screen
            ├── Header.jsx            # Navigation bar
            ├── Dashboard.jsx         # Main workspace
            ├── ImageUploader.jsx     # Drag-and-drop upload
            ├── AgentPipeline.jsx     # Live agent progress display
            ├── DiagnosticReport.jsx  # Full analysis report view
            ├── FindingsPanel.jsx     # Individual findings cards
            ├── ConfidenceOverview.jsx# Confidence meter
            ├── PatientList.jsx       # Patient directory
            ├── PatientProfile.jsx    # Patient detail + scan history
            ├── AddPatientModal.jsx   # New patient form
            ├── ReferralModal.jsx     # Create specialist referral
            └── ToastContainer.jsx    # Notification toasts
```

---

## Getting Started

### Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`

### 1. Install Backend Dependencies

```bash
pip install fastapi uvicorn python-multipart pillow opencv-python numpy pydantic
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Run in Development Mode

Start the backend API server:

```bash
# From the project root
uvicorn backend:app --reload --port 8000
```

Start the frontend dev server (in a second terminal):

```bash
cd frontend
npm run dev
```

The frontend dev server (Vite) proxies API calls to `http://localhost:8000`.

### 4. Production Build (optional)

```bash
cd frontend
npm run build
```

The built files are placed in `frontend/dist/`. The FastAPI server automatically serves these static files in production, so you only need to run the backend:

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### Analysis

| Method | Path           | Description                                                                              |
| ------ | -------------- | ---------------------------------------------------------------------------------------- |
| `GET`  | `/api/health`  | Health check — returns agent status                                                      |
| `POST` | `/api/analyze` | Upload an image for three-agent analysis (multipart form: `file`, optional `patient_id`) |

### Patients

| Method   | Path                 | Description          |
| -------- | -------------------- | -------------------- |
| `GET`    | `/api/patients`      | List all patients    |
| `POST`   | `/api/patients`      | Create a new patient |
| `GET`    | `/api/patients/{id}` | Get patient details  |
| `PUT`    | `/api/patients/{id}` | Update patient info  |
| `DELETE` | `/api/patients/{id}` | Delete a patient     |

### Scans & Referrals

| Method | Path                       | Description                                     |
| ------ | -------------------------- | ----------------------------------------------- |
| `POST` | `/api/patients/{id}/scans` | Save a scan to a patient                        |
| `POST` | `/api/referrals`           | Create a specialist referral                    |
| `GET`  | `/api/referrals`           | List referrals (optional `?patient_id=` filter) |
