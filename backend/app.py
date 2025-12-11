# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sqlite3
import os
import joblib
import json
import logging
import re
from cryptography.fernet import Fernet
import sqlite3 as _sqlite
from sqlite3 import IntegrityError, OperationalError

# ---------- CONFIG ----------
DATA_PATH = r"C:\Users\akash\OneDrive\Desktop\farmers_dataset.csv"
FALLBACK_DATA_PATH = r"/mnt/data/b799548b-bdbf-40ab-bbff-2f6c7a23c1f0.csv"
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "subsidy_model_v1.pkl")
DB_PATH = os.path.join(BASE_DIR, "applications.db")
FERNET_KEY_PATH = os.path.join(BASE_DIR, "fernet.key")
LOG_PATH = os.path.join(BASE_DIR, "backend.log")

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)

# ---------- Flask ----------
app = Flask(__name__)
# allow all origins for dev; tighten in production
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------- Aadhaar normalization helper ----------
def normalize_aadhaar_value(v):
    s = str(v).strip()
    if not s:
        return ""
    s = s.replace(",", "").replace(" ", "")
    try:
        if "e" in s.lower() or "." in s:
            n = int(float(s))
            s = str(n)
    except Exception:
        pass
    s = re.sub(r"\D", "", s)
    if not s:
        return ""
    return s.zfill(12)[-12:]


# ---------- Load dataset ----------
if os.path.exists(DATA_PATH):
    dataset_path = DATA_PATH
elif os.path.exists(FALLBACK_DATA_PATH):
    dataset_path = FALLBACK_DATA_PATH
else:
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH} or {FALLBACK_DATA_PATH}")

farmers_df = pd.read_csv(dataset_path, dtype=str)
if "acres" in farmers_df.columns:
    farmers_df["acres"] = pd.to_numeric(farmers_df["acres"], errors="coerce").fillna(0)
else:
    farmers_df["acres"] = 0

AADHAAR_COL = None
for col in farmers_df.columns:
    cl = col.lower().strip()
    if "aadhaar" in cl or "aadhar" in cl:
        AADHAAR_COL = col
        break
if AADHAAR_COL is None:
    raise KeyError(f"No Aadhaar-like column found in dataset. Columns: {list(farmers_df.columns)}")

# normalize dataset aadhaar
farmers_df[AADHAAR_COL] = farmers_df[AADHAAR_COL].apply(normalize_aadhaar_value)

logging.info(f"Loaded dataset from: {dataset_path}  rows={len(farmers_df)}")
logging.info(f"Using Aadhaar column: {AADHAAR_COL}")
logging.info("Sample normalized Aadhaar values:")
logging.info(farmers_df[AADHAAR_COL].head().to_string(index=False))

# ---------- Load model ----------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"Loaded model: {MODEL_PATH}")
    except Exception as e:
        logging.exception("Failed to load model, continuing without ML: %s", e)
        model = None
else:
    logging.warning("Model not found at %s — predictions will be None", MODEL_PATH)

# ---------- Encryption (Fernet) ----------
def get_fernet():
    if os.path.exists(FERNET_KEY_PATH):
        key = open(FERNET_KEY_PATH, "rb").read()
    else:
        key = Fernet.generate_key()
        open(FERNET_KEY_PATH, "wb").write(key)
    return Fernet(key)

fernet = get_fernet()

# ---------- Utilities ----------
AADHAAR_RE = re.compile(r"^\d{12}$")
def is_valid_aadhaar(a): return bool(AADHAAR_RE.match(a or ""))
def mask_aadhaar(a):
    if not a or len(a) < 4: return a
    return "****" + a[-4:]

def get_db_connection():
    """
    Return a new sqlite3 connection configured with a timeout and WAL mode.
    Use check_same_thread=False to reduce 'database is locked' in dev (safe for simple apps).
    """
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    # Speed/locking improvements
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn

def ensure_db_schema():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            aadhaar TEXT,
            name TEXT,
            acres REAL,
            location TEXT,
            soil_type TEXT,
            irrigation_type TEXT,
            income REAL,
            crop TEXT,
            phone TEXT,
            bank_account TEXT,
            status TEXT DEFAULT 'Submitted',
            predicted_amount REAL,
            eligible_schemes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER,
            old_status TEXT,
            new_status TEXT,
            changed_by TEXT,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # unique index ensures one Aadhaar -> one application
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_app_unique_aadhaar ON applications(aadhaar);")
    conn.commit()
    conn.close()

ensure_db_schema()

# ---------- Multi-subsidy rules ----------
def evaluate_subsidies(acres, income, crop, soil_type, irrigation_type, location=None):
    eligible = []
    if income is not None and income < 100000:
        eligible.append({"scheme_name":"PM-Kisan","amount":6000,"reason":"Income < 1,00,000"})
    if isinstance(crop,str) and crop.lower() in ["rice","wheat","maize","cotton"]:
        proxy_value = max(0, acres * 50000)
        eligible.append({"scheme_name":"PMFBY Crop Insurance","amount":round(0.02*proxy_value,2),"reason":f"Crop eligible for PMFBY: {crop}"})
    if isinstance(soil_type,str) and soil_type.lower() in ["sandy","clay","loamy"]:
        eligible.append({"scheme_name":"Soil Health Card Scheme","amount":3000,"reason":f"Soil type registered: {soil_type}"})
    if isinstance(irrigation_type,str) and "rain" in irrigation_type.lower():
        eligible.append({"scheme_name":"Irrigation Support Subsidy","amount":15000,"reason":"Rainfed irrigation"})
    if isinstance(crop,str) and crop.lower() in ["rice","wheat","maize"]:
        eligible.append({"scheme_name":"Fertilizer Support Subsidy","amount":2000,"reason":"Crop eligible for fertilizer subsidy"})
    if acres is not None and acres > 5:
        eligible.append({"scheme_name":"Machinery Purchase Subsidy","amount":25000,"reason":"Farmer has large landholding (acres > 5)"})
    if isinstance(soil_type,str) and soil_type.lower() in ["loamy","clay"]:
        eligible.append({"scheme_name":"Organic Farming Support","amount":12000,"reason":"Soil suitable for organic cultivation"})
    if isinstance(location,str) and location.upper() == "UP":
        eligible.append({"scheme_name":"State Special Support (UP)","amount":5000,"reason":"State-level top-up for UP"})
    if len(eligible)==0:
        return [{"scheme_name":"Not Eligible","amount":0,"reason":"Does not meet any subsidy criteria"}]
    return eligible

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return "<p>Smart Subsidy Backend running.</p>"

@app.route("/lookup", methods=["POST"])
def lookup():
    data = request.get_json(force=True, silent=True) or {}
    aadhaar_raw = data.get("aadhaar", "")
    aadhaar = normalize_aadhaar_value(aadhaar_raw)
    if not aadhaar:
        return jsonify({"found": False, "message": "Aadhaar missing"}), 400
    if not is_valid_aadhaar(aadhaar):
        return jsonify({"found": False, "message": "Aadhaar invalid format"}), 400
    row = farmers_df[farmers_df[AADHAAR_COL] == aadhaar]
    if row.empty:
        return jsonify({"found": False, "message": "Aadhaar not found"}), 404
    rec = row.iloc[0].to_dict()
    rec["aadhaar_masked"] = mask_aadhaar(aadhaar)
    return jsonify({"found": True, "record": rec, "details": rec}), 200

@app.route("/submit", methods=["POST"])
def submit():
    payload = request.get_json(force=True, silent=True) or {}
    aadhaar_raw = payload.get("aadhaar", "")
    aadhaar = normalize_aadhaar_value(aadhaar_raw)
    if not aadhaar or not is_valid_aadhaar(aadhaar):
        return jsonify({"error": "aadhaar missing or invalid"}), 400

    # verify farmer exists
    row = farmers_df[farmers_df[AADHAAR_COL] == aadhaar]
    if row.empty:
        return jsonify({"error": "Aadhaar not found"}), 400

    # prepare data
    r = row.iloc[0].to_dict()
    name = r.get("name", "")
    try:
        acres = float(r.get("acres", 0) or 0)
    except Exception:
        acres = 0.0
    location = r.get("location", "")
    soil_type = r.get("soil_type", "")
    irrigation_type = r.get("irrigation_type", "")

    try:
        income = float(payload.get("income", 0) or 0)
    except Exception:
        income = 0.0
    if income <= 0:
        return jsonify({"error":"Income must be a positive number"}), 400
    crop = payload.get("crop", "") or ""
    phone = payload.get("phone", "") or ""
    bank_account = payload.get("bank_account", "") or ""

    # ML prediction
    predicted_amount = None
    if model is not None:
        try:
            X_predict = pd.DataFrame([{
                "acres": acres,
                "soil_type": soil_type,
                "irrigation_type": irrigation_type,
                "location": location,
                "crop": crop,
                "income": income
            }])
            logging.info("Predict input: %s", X_predict.to_dict(orient='records'))
            predicted_amount = float(model.predict(X_predict)[0])
        except Exception as e:
            logging.exception("Model prediction error: %s", e)
            predicted_amount = None

    eligible_schemes = evaluate_subsidies(acres, income, crop, soil_type, irrigation_type, location)

    # encrypt bank
    enc_bank = None
    if bank_account:
        try:
            enc_bank = fernet.encrypt(bank_account.encode()).decode()
        except Exception:
            enc_bank = None

    # Save to DB with safe handling
    conn = None
    try:
        conn = get_db_connection()
        c = conn.cursor()
        # check existing application for aadhaar (defensive)
        c.execute("SELECT id, status, created_at FROM applications WHERE aadhaar = ?", (aadhaar,))
        existing = c.fetchone()
        if existing:
            return (
                jsonify({
                    "error": "Application already exists for this Aadhaar",
                    "application_id": existing[0],
                    "status": existing[1],
                    "created_at": existing[2]
                }),
                400,
            )
        # attempt insert; catch IntegrityError if race happened
        try:
            c.execute("""
                INSERT INTO applications
                (aadhaar, name, acres, location, soil_type, irrigation_type,
                 income, crop, phone, bank_account, predicted_amount, eligible_schemes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                aadhaar, name, acres, location, soil_type, irrigation_type,
                income, crop, phone, enc_bank, predicted_amount, json.dumps(eligible_schemes)
            ))
            app_id = c.lastrowid
            conn.commit()
        except IntegrityError as ie:
            logging.warning("Insert IntegrityError (likely duplicate): %s", ie)
            # fetch existing id and return friendly message
            c.execute("SELECT id, status, created_at FROM applications WHERE aadhaar = ?", (aadhaar,))
            ex = c.fetchone()
            if ex:
                return (
                    jsonify({
                        "error": "Application already exists for this Aadhaar",
                        "application_id": ex[0],
                        "status": ex[1],
                        "created_at": ex[2]
                    }),
                    400,
                )
            else:
                # re-raise if unexpected
                raise

    except OperationalError as oe:
        logging.exception("Database operational error (locked?): %s", oe)
        return jsonify({"error": "Temporary database error, please try again"}), 500
    finally:
        if conn:
            conn.close()

    logging.info("Saved application %s for aadhaar %s (masked: %s)", app_id, mask_aadhaar(aadhaar), mask_aadhaar(aadhaar))
    return jsonify({"application_id": app_id, "predicted_amount": predicted_amount, "eligible_schemes": eligible_schemes}), 201

@app.route("/status/<int:app_id>", methods=["GET"])
def status(app_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, aadhaar, name, status, predicted_amount, eligible_schemes, created_at FROM applications WHERE id=?", (app_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error":"Not found"}), 404
    keys = ["id","aadhaar","name","status","predicted_amount","eligible_schemes","created_at"]
    result = dict(zip(keys, row))
    result["aadhaar"] = mask_aadhaar(result.get("aadhaar"))
    try:
        result["eligible_schemes"] = json.loads(result.get("eligible_schemes") or "[]")
    except Exception:
        result["eligible_schemes"] = []
    c.execute("SELECT old_status, new_status, changed_by, changed_at FROM status_history WHERE application_id=? ORDER BY changed_at", (app_id,))
    history_rows = c.fetchall()
    conn.close()
    history = []
    for h in history_rows:
        history.append({"from": h[0], "to": h[1], "by": h[2], "at": h[3]})
    result["status_history"] = history
    return jsonify(result), 200

@app.route("/admin/update", methods=["POST"])
def admin_update():
    data = request.get_json(force=True, silent=True) or {}
    app_id = data.get("application_id")
    new_status = data.get("status")
    changed_by = data.get("changed_by", "admin")
    if not app_id or not new_status:
        return jsonify({"error":"application_id and status required"}), 400
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT status FROM applications WHERE id=?", (app_id,))
    old = c.fetchone()
    old_status = old[0] if old else None
    c.execute("UPDATE applications SET status=? WHERE id=?", (new_status, app_id))
    c.execute("INSERT INTO status_history (application_id, old_status, new_status, changed_by) VALUES (?, ?, ?, ?)", (app_id, old_status, new_status, changed_by))
    conn.commit()
    conn.close()
    logging.info("Admin update: app %s status %s -> %s by %s", app_id, old_status, new_status, changed_by)
    return jsonify({"ok": True}), 200

@app.route("/admin/applications", methods=["GET"])
def admin_applications():
    status_filter = request.args.get("status")
    conn = get_db_connection()
    c = conn.cursor()
    if status_filter:
        c.execute("SELECT id, name, status, created_at FROM applications WHERE status=? ORDER BY created_at DESC", (status_filter,))
    else:
        c.execute("SELECT id, name, status, created_at FROM applications ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    apps = [{"id": r[0], "name": r[1], "status": r[2], "created_at": r[3]} for r in rows]
    return jsonify({"applications": apps}), 200

@app.route("/admin/export", methods=["GET"])
def admin_export():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM applications", conn)
    conn.close()
    csv_content = df.to_csv(index=False)
    return (csv_content, 200, {"Content-Type":"text/csv","Content-Disposition":"attachment;filename=subsidy_applications_export.csv"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"running","model_loaded": model is not None,"dataset_records": len(farmers_df),"db_exists": os.path.exists(DB_PATH),"aadhaar_column": AADHAAR_COL}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
