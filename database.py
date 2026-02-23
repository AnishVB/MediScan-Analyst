"""
MediScan Analyst — Database Module
SQLite database for patients, scans, and referrals
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediscan.db")


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            blood_group TEXT,
            medical_conditions TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            image_filename TEXT,
            image_type TEXT,
            body_part TEXT,
            analysis_json TEXT,
            confidence REAL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS referrals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            patient_id INTEGER NOT NULL,
            specialist_name TEXT NOT NULL,
            specialist_field TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE SET NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()


# ============================================================================
# PATIENT CRUD
# ============================================================================

def create_patient(name, age=None, gender=None, blood_group=None, medical_conditions="", notes=""):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO patients (name, age, gender, blood_group, medical_conditions, notes) VALUES (?, ?, ?, ?, ?, ?)",
        (name, age, gender, blood_group, medical_conditions, notes)
    )
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id


def get_all_patients():
    conn = get_db()
    rows = conn.execute("""
        SELECT p.*, 
            (SELECT COUNT(*) FROM scans WHERE patient_id = p.id) as scan_count,
            (SELECT MAX(created_at) FROM scans WHERE patient_id = p.id) as last_scan
        FROM patients p ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(patient_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_patient(patient_id, **kwargs):
    conn = get_db()
    fields = []
    values = []
    for k, v in kwargs.items():
        if k in ("name", "age", "gender", "blood_group", "medical_conditions", "notes"):
            fields.append(f"{k} = ?")
            values.append(v)
    if fields:
        values.append(patient_id)
        conn.execute(f"UPDATE patients SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
    conn.close()


def delete_patient(patient_id):
    conn = get_db()
    conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()


# ============================================================================
# SCAN CRUD
# ============================================================================

def save_scan(patient_id, image_filename, image_type, body_part, analysis_json, confidence):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scans (patient_id, image_filename, image_type, body_part, analysis_json, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (patient_id, image_filename, image_type, body_part, json.dumps(analysis_json), confidence)
    )
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id


def get_patient_scans(patient_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM scans WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,)
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["analysis_json"] = json.loads(d["analysis_json"]) if d["analysis_json"] else None
        except:
            pass
        results.append(d)
    return results


def get_scan(scan_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    conn.close()
    if row:
        d = dict(row)
        try:
            d["analysis_json"] = json.loads(d["analysis_json"]) if d["analysis_json"] else None
        except:
            pass
        return d
    return None


# ============================================================================
# REFERRAL CRUD
# ============================================================================

def create_referral(patient_id, scan_id, specialist_name, specialist_field, notes=""):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO referrals (patient_id, scan_id, specialist_name, specialist_field, notes) VALUES (?, ?, ?, ?, ?)",
        (patient_id, scan_id, specialist_name, specialist_field, notes)
    )
    conn.commit()
    ref_id = cursor.lastrowid
    conn.close()
    return ref_id


def get_referrals(patient_id=None):
    conn = get_db()
    if patient_id:
        rows = conn.execute("""
            SELECT r.*, p.name as patient_name 
            FROM referrals r 
            JOIN patients p ON r.patient_id = p.id 
            WHERE r.patient_id = ? 
            ORDER BY r.created_at DESC
        """, (patient_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT r.*, p.name as patient_name 
            FROM referrals r 
            JOIN patients p ON r.patient_id = p.id 
            ORDER BY r.created_at DESC
        """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient_history_summary(patient_id):
    """Get a summary of patient's medical history for AI context."""
    patient = get_patient(patient_id)
    if not patient:
        return None

    scans = get_patient_scans(patient_id)
    
    history = {
        "patient_name": patient["name"],
        "age": patient["age"],
        "gender": patient["gender"],
        "blood_group": patient["blood_group"],
        "known_conditions": patient["medical_conditions"],
        "total_scans": len(scans),
        "past_findings": [],
    }

    for scan in scans[:10]:  # Last 10 scans
        analysis = scan.get("analysis_json")
        if analysis and isinstance(analysis, dict):
            findings = analysis.get("analysis", {}).get("findings", [])
            for f in findings:
                if f.get("finding_type") != "Image Quality Assessment":
                    history["past_findings"].append({
                        "date": scan["created_at"],
                        "body_part": scan["body_part"],
                        "finding": f.get("finding_type"),
                        "description": f.get("description"),
                        "confidence": f.get("confidence"),
                    })

    return history


# Initialize on import
init_db()
