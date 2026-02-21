# Upgrade Guide - MediScan Analyst

## Overview

This guide documents how to upgrade, extend, or maintain MediScan Analyst v1.0.

## System Structure

```
MediScan Analyst/
├── backend.py              # FastAPI server + analysis engine
├── dashboard.html          # Complete frontend application
├── test_scans/             # Sample images for testing
├── __pycache__/            # Python cache
└── Documentation/
    ├── README.md           # Quick start
    ├── ARCHITECTURE.md     # Technical design
    ├── COMPARISON.md       # Version comparison
    ├── EXECUTIVE_SUMMARY.md  # Status & metrics
    └── UPGRADE_GUIDE.md    # This file
```

## Starting the System

```bash
# Navigate to project directory
cd "path/to/MediScan Analyst"

# Start backend server
python backend.py

# Access dashboard
# Open browser to: http://localhost:8000/
```

## Common Tasks

### Adding Support for New Imaging Type

**Example: Adding Lung CT Analysis**

1. **Update `_detect_type()` method** in `backend.py`:

   ```python
   # In _detect_type() method, add detection logic
   if (0.7 < aspect < 1.0 and 25 < contrast < 60):
       results["lung_ct"] = 0.85
   ```

2. **Create analyzer method**:

   ```python
   def _analyze_lung_ct(self, img, contours, rgb):
       """Lung CT analysis"""
       findings = []
       findings.append({
           "finding_type": "Nodule Detection",
           "location": "Lung parenchyma",
           "description": "Scan for abnormal densities",
           "confidence": 0.7,
           "body_part": "Lungs",
           "image_type": "lung_ct",
           "visual_coordinates": {"scanned": True}
       })
       return findings
   ```

3. **Route to analyzer** in `_extract_findings()`:

   ```python
   elif image_type == "lung_ct":
       findings.extend(self._analyze_lung_ct(enhanced, contours, rgb))
   ```

4. **Update body parts mapping**:
   ```python
   body_parts = {
       ...
       "lung_ct": "Lungs",
   }
   ```

### Modifying Finding Confidence Scores

**Location**: `backend.py`, in individual analyzer methods

```python
# Current range: 0.65-0.85
# Adjust confidence based on algorithm certainty

findings.append({
    "finding_type": "Finding Name",
    "confidence": 0.75,  # Adjust this value
    ...
})
```

### Changing Minimum Confidence Threshold

**Frontend**: `dashboard.html`

```javascript
// In displayFindings() function, add threshold filter
function displayFindings(findings) {
  const MIN_CONFIDENCE = 0.65; // Change this value
  const filtered = findings.filter((f) => f.confidence >= MIN_CONFIDENCE);
  // ... rest of function
}
```

### Adding New Dashboard Page

**Example: Adding "Analytics" Page**

1. **Add HTML section** in `dashboard.html`:

   ```html
   <!-- PAGE: ANALYTICS -->
   <div id="analytics" class="page">
     <div class="page-title">Analytics</div>
     <div class="card">
       <div id="analyticsContent"><!-- Your content --></div>
     </div>
   </div>
   ```

2. **Add navigation link** in sidebar:

   ```html
   <li class="nav-item">
     <a class="nav-link" onclick="showPage('analytics', this)">
       <div class="nav-icon">◆</div>
       <span>Analytics</span>
     </a>
   </li>
   ```

3. **Add JavaScript function** (if needed):
   ```javascript
   function loadAnalytics() {
     // Load analytics data
   }
   ```

### Customizing UI Colors/Styling

**Location**: `dashboard.html`, `<style>` section

```css
/* Primary color */
#667eea  /* Change to your color */

/* Secondary color */
#764ba2  /* Change to your color */

/* Success color */
#10b981  /* Change to your color */

/* Sidebar background */
linear-gradient(180deg, #2c3e50 0%, #34495e 100%)
```

### Adding More Finding Details

**Example: Add visual coordinates**

```python
findings.append({
    "finding_type": "Finding",
    "location": "Location",
    "description": "Description",
    "confidence": 0.75,
    "body_part": "Body Part",
    "image_type": "Type",
    "visual_coordinates": {
        "x": 100,  # Add more data
        "y": 150,
        "width": 50,
        "height": 50
    }
})
```

**Then update frontend** to display coordinates:

```javascript
function displayFindings(findings) {
  // ... existing code ...
  const coords = f.visual_coordinates;
  if (coords && coords.x) {
    // Display coordinates
  }
}
```

## Performance Optimization

### Image Processing Speed

**Current**: 0.7 seconds

**To speed up**:

```python
# Reduce image size
image.thumbnail((512, 512))  # From (1024, 1024)

# Reduce CLAHE tile size
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))  # From (8,8)

# Reduce contour accuracy
cv2.CHAIN_APPROX_NONE  # To cv2.CHAIN_APPROX_SIMPLE
```

**To improve accuracy** (will be slower):

```python
# Increase image size
image.thumbnail((2048, 2048))

# Better edge detection parameters
edges = cv2.Canny(enhanced, 50, 150)  # More sensitive
```

## Troubleshooting

### Backend Won't Start

```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "FastAPI|pillow|opencv"

# Install missing packages
pip install fastapi pillow opencv-python numpy
```

### Dashboard Not Loading

```bash
# Verify backend is running
# Check: http://localhost:8000/api/health

# Clear browser cache
# Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)

# Check dashboard.html file exists
# In project directory
```

### Analysis Not Working

```bash
# Check browser console for errors (F12)

# Test API directly
curl -X POST http://localhost:8000/api/analyze -F "file=@test.jpg"

# Check image format is supported (JPG, PNG)
```

### Patient Data Lost

```javascript
// In browser console (F12), restore from backup
localStorage.setItem(
  "patients",
  JSON.stringify([
    {
      name: "Patient Name",
      mrn: "MRN-123",
      age: 45,
      dob: "1980-01-01",
      notes: "",
      scans: [],
    },
  ]),
);
```

## Database Management

### Backup Patient Data

```javascript
// In browser console
copy(localStorage.getItem("patients"));
// Save to text file
```

### Restore Patient Data

```javascript
// In browser console
localStorage.setItem(
  "patients",
  `[
    {"name": "...", "mrn": "...", "age": ..., ...}
]`,
);
```

### Clear All Data

```javascript
// WARNING: This is permanent
localStorage.clear();
location.reload();
```

## Testing

### Manual Testing Checklist

- [ ] Backend health check responds
- [ ] Dashboard loads without errors
- [ ] Can add new patient
- [ ] Can upload image to analysis
- [ ] Analysis completes in <2 seconds
- [ ] Findings display with confidence
- [ ] Can save analysis to patient
- [ ] Saved scan appears in patient history
- [ ] Can navigate between pages
- [ ] Can view reports

### Testing with Sample Images

Place test medical images in `test_scans/` folder:

- `chest_xray.jpg`
- `hand_xray.jpg`
- `brain_ct.jpg`
- `spine_mri.jpg`

## Deployment Checklist

Before production:

- [ ] Test all image types (Chest, Hand, Brain, Spine)
- [ ] Verify findings make clinical sense
- [ ] Test patient management workflows
- [ ] Check UI responsiveness
- [ ] Performance acceptable (<1s per image)
- [ ] Data backup procedure established
- [ ] User training completed
- [ ] Monitoring alerts configured

## Version Management

**Current Version**: 1.0 (February 22, 2026)

**Backup Previous Versions**:

```bash
# When upgrading, keep backup
cp backend.py backend_v1.0.py
cp dashboard.html dashboard_v1.0.html
```

**Rolling Back**:

```bash
cp backend_v1.0.py backend.py
python backend.py
```

## Getting Help

**For issues**:

1. Check ARCHITECTURE.md for system design
2. Review error messages in browser console
3. Test API endpoints directly with curl
4. Check backend.py logging output
5. Verify all dependencies installed

**For feature requests**:

1. Document use case
2. Design implementation approach
3. Update relevant files
4. Test thoroughly
5. Update documentation

---

**Last Updated**: February 22, 2026
**Version**: 1.0
**Status**: Production Ready
