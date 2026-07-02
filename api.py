"""
api.py
======
Unified Flask API for EMS Dashboard.
Features: Supabase DB, Geo-Tracking, Automated Emails, Admin Audit Trail.
"""
import io
import json
import threading
import time
import hashlib
import random
import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask.logging import default_handler
from supabase import create_client, Client
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from solver_core import run_static_suite, run_uc_suite
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.logger.removeHandler(default_handler)

# ── SUPABASE CONFIGURATION ──────────────────────────────────────────────────
SUPABASE_URL = "https://xaihmzhpvtftwmgmbbgz.supabase.co"   # <-- your project URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhhaWhtemhwdnRmdHdtZ21iYmd6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODI0NTk5NDEsImV4cCI6MjA5ODAzNTk0MX0.BmA7mb0WRoGb9Jc-YKlbF9qrpLNqZUzEg6ZFDmXzPbg"  # <-- your anon key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── EMAIL CONFIGURATION ─────────────────────────────────────────────────────
SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 465
SENDER_EMAIL   = "gridsuite.test@gmail.com"   # <-- your Gmail address
SENDER_PASSWORD = "gzxbacjgflnlvsei"          # <-- your 16-char Gmail App Password

# ── ADMIN SECRET ────────────────────────────────────────────────────────────
# Change this to any secret string. Use it as ?key=<ADMIN_SECRET> in the URL.
ADMIN_SECRET = "gridoptima_admin_2025"

# ── MISC GLOBALS ────────────────────────────────────────────────────────────
pending_otps = {}
_job  = {"status": "idle", "logs": [], "result": None, "error": None, "started": None, "elapsed": None}
_lock = threading.Lock()


# ── CORS ─────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def hash_password(password: str) -> str:
    return hashlib.sha256((password + "NIT_HAMIRPUR_SALT").encode("utf-8")).hexdigest()


def send_automated_email(to_email: str, subject: str, html_content: str):
    """Send an HTML email via Gmail SMTP-SSL. Silently ignores failures."""
    if not to_email:
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"GridOptima EMS <{SENDER_EMAIL}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(html_content, "html"))
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
    except Exception as e:
        print(f"[EMAIL] Failed to send to {to_email}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH  &  GEO-TRACKING  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/auth/request-otp", methods=["POST"])
def request_otp():
    """
    Step 1 of registration.
    Checks that the email is not already registered, generates a 6-digit OTP,
    stores it in memory, and e-mails it to the user.
    """
    email = request.json.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Email is required."}), 400

    # Reject if account already exists
    existing = supabase.table("users").select("id").eq("email", email).execute()
    if existing.data:
        return jsonify({"error": "Account already exists. Please login."}), 400

    otp = str(random.randint(100000, 999999))
    pending_otps[email] = otp
    print(f"\n[SECURITY] OTP for {email}: {otp}\n")   # visible in server terminal

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:28px;
                border:1px solid #e5e7eb;border-radius:10px;background:#f9fafb;">
        <h2 style="color:#1d4ed8;margin-bottom:6px;">GridOptima Workspace</h2>
        <p style="color:#374151;font-size:14px;">
            Someone (hopefully you!) requested access to the GridOptima EMS platform.
            Use the code below to complete your registration.
        </p>
        <div style="text-align:center;margin:28px 0;">
            <span style="font-size:38px;font-weight:700;letter-spacing:8px;color:#10b981;
                         font-family:'Courier New',monospace;">{otp}</span>
        </div>
        <p style="color:#6b7280;font-size:12px;">
            This code is valid for the current server session.<br>
            If you did not request this, please ignore this email.
        </p>
    </div>
    """
    send_automated_email(email, "GridOptima — System Registration OTP", html_body)
    return jsonify({"message": "OTP sent. Check your inbox (and spam folder)."})


@app.route("/auth/verify", methods=["POST"])
def verify_otp():
    """
    Step 2 of registration.
    Validates the OTP, hashes the password, and creates the user record in Supabase.
    """
    data     = request.json or {}
    email    = data.get("email",    "").strip().lower()
    otp      = data.get("otp",      "").strip()
    password = data.get("password", "")

    if not email or not otp or not password:
        return jsonify({"error": "Email, OTP, and password are all required."}), 400

    if pending_otps.get(email) != otp:
        return jsonify({"error": "Invalid or expired OTP."}), 400

    supabase.table("users").insert({
        "email":    email,
        "password": hash_password(password),
    }).execute()
    del pending_otps[email]

    # Welcome email
    send_automated_email(
        email,
        "Welcome to GridOptima EMS",
        f"""
        <div style="font-family:Arial,sans-serif;padding:24px;border-left:4px solid #10b981;
                    background:#f0fdf4;border-radius:8px;">
            <h2 style="color:#065f46;">Account Created Successfully</h2>
            <p>Your GridOptima EMS account is active.<br>
               You can now log in with your registered email and password.</p>
        </div>
        """,
    )
    return jsonify({"message": "Account created successfully!"})


@app.route("/auth/login", methods=["POST"])
def login():
    """
    Authenticates a user, records IP address + geolocation to Supabase login_logs,
    and returns the resolved location to the frontend.
    """
    data     = request.json or {}
    email    = data.get("email",    "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    res = supabase.table("users").select("*").eq("email", email).execute()
    if not res.data or res.data[0]["password"] != hash_password(password):
        return jsonify({"error": "Invalid email or password."}), 401

    # ── Geo-tracking ────────────────────────────────────────────────
    ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.headers.get("X-Real-IP", "")
        or request.remote_addr
        or "127.0.0.1"
    )
    location = "Local / Unknown"
    city     = ""
    country  = ""
    if ip not in ("127.0.0.1", "::1", ""):
        try:
            geo = requests.get(f"http://ip-api.com/json/{ip}?fields=status,city,regionName,country,countryCode,isp", timeout=3).json()
            if geo.get("status") == "success":
                city     = geo.get("city",       "")
                region   = geo.get("regionName", "")
                country  = geo.get("country",    "")
                isp      = geo.get("isp",        "")
                location = f"{city}, {region}, {country}"
        except Exception as geo_err:
            print(f"[GEO] Lookup failed for {ip}: {geo_err}")

    # Write audit row to Supabase
    try:
        supabase.table("login_logs").insert({
            "email":      email,
            "ip_address": ip,
            "city":       city,
            "country":    country,
            "location":   location,
        }).execute()
    except Exception as db_err:
        print(f"[DB] login_logs insert failed: {db_err}")

    return jsonify({
        "message":  "Login successful",
        "email":    email,
        "location": location,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN  AUDIT  TRAIL  ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/admin/logins", methods=["GET"])
def admin_logins():
    """
    Returns the last 200 login records (most recent first).
    Protected by a secret key: GET /admin/logins?key=<ADMIN_SECRET>

    Usage:
        http://localhost:5000/admin/logins?key=gridoptima_admin_2025

    Change ADMIN_SECRET at the top of this file before deploying!
    """
    if request.args.get("key", "") != ADMIN_SECRET:
        return jsonify({"error": "Forbidden — invalid admin key."}), 403

    try:
        rows = (
            supabase.table("login_logs")
            .select("*")
            .order("id", desc=True)
            .limit(200)
            .execute()
        )
        return jsonify({
            "count":  len(rows.data),
            "logins": rows.data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/users", methods=["GET"])
def admin_users():
    """
    Returns a list of all registered users (email + created_at only, no passwords).
    Protected by the same ADMIN_SECRET key.

    Usage:
        http://localhost:5000/admin/users?key=gridoptima_admin_2025
    """
    if request.args.get("key", "") != ADMIN_SECRET:
        return jsonify({"error": "Forbidden — invalid admin key."}), 403

    try:
        rows = (
            supabase.table("users")
            .select("id, email, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return jsonify({
            "count": len(rows.data),
            "users": rows.data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  SOLVER  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/status", methods=["GET"])
def status():
    with _lock:
        elapsed = (
            round(time.time() - _job["started"], 1)
            if _job["started"] and _job["status"] == "running"
            else _job["elapsed"]
        )
        return jsonify({
            "status":  _job["status"],
            "logs":    _job["logs"][-15:],
            "elapsed": elapsed,
            "error":   _job["error"],
        })


@app.route("/result", methods=["GET"])
def result():
    with _lock:
        return jsonify(_job["result"])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _run_job(gen_df, bus_df, line_df, suite_type, user_email):
    def log(msg):
        with _lock:
            _job["logs"].append(msg)
        print(msg)

    t0 = time.time()
    try:
        if suite_type == "static":
            res  = run_static_suite(gen_df, bus_df, line_df, log_cb=log)
            cost = (res.get("ed") or {}).get("total_cost") or (res.get("dcopf") or {}).get("total_cost")
        else:
            res  = run_uc_suite(gen_df, bus_df, line_df, log_cb=log)
            cost = (res.get("scuc") or {}).get("total_cost")

        elapsed = round(time.time() - t0, 2)

        with _lock:
            _job["status"]  = "done"
            _job["result"]  = json.loads(json.dumps(res, cls=NumpyEncoder))
            _job["elapsed"] = elapsed

        if user_email and cost is not None:
            report = f"""
            <div style="font-family:Arial,sans-serif;padding:24px;
                        border-left:4px solid #3b82f6;background:#f8fafc;border-radius:8px;">
                <h2 style="color:#1d4ed8;">Optimization Execution Complete</h2>
                <table style="border-collapse:collapse;font-size:14px;">
                    <tr><td style="padding:4px 12px 4px 0;color:#6b7280;">Mode</td>
                        <td style="font-weight:600;">{suite_type.upper()}</td></tr>
                    <tr><td style="padding:4px 12px 4px 0;color:#6b7280;">Total Cost</td>
                        <td style="font-weight:600;">${cost:,.2f}</td></tr>
                    <tr><td style="padding:4px 12px 4px 0;color:#6b7280;">Solve Time</td>
                        <td style="font-weight:600;">{elapsed}s</td></tr>
                </table>
            </div>
            """
            send_automated_email(user_email, f"GridOptima — {suite_type.upper()} Report", report)

    except Exception as exc:
        with _lock:
            _job["status"] = "error"
            _job["error"]  = str(exc)
        print(f"[SOLVER ERROR] {exc}")


@app.route("/solve", methods=["POST"])
def solve():
    with _lock:
        if _job["status"] == "running":
            return jsonify({"error": "Solver is busy with another job."}), 409

    try:
        suite_type = request.form.get("suite_type", "uc")
        user_email = request.form.get("user_email", "")

        gen_df = pd.read_csv(io.StringIO(request.files["gen_file"].read().decode("utf-8")))
        bus_df = pd.read_csv(io.StringIO(request.files["bus_file"].read().decode("utf-8")))
        line_df = (
            pd.read_csv(io.StringIO(request.files["line_file"].read().decode("utf-8")))
            if "line_file" in request.files and request.files["line_file"].filename != ""
            else None
        )

        with _lock:
            _job.update({
                "status":  "running",
                "logs":    ["Initializing Engine..."],
                "result":  None,
                "error":   None,
                "started": time.time(),
                "elapsed": None,
            })

        threading.Thread(
            target=_run_job,
            args=(gen_df, bus_df, line_df, suite_type, user_email),
            daemon=True,
        ).start()

        return jsonify({"status": "started"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n  GridOptima EMS Server Running on http://localhost:5000\n")
    print(f"  Admin audit trail → http://localhost:5000/admin/logins?key={ADMIN_SECRET}")
    print(f"  Registered users  → http://localhost:5000/admin/users?key={ADMIN_SECRET}\n")
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)