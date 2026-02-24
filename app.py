from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json
import os 
from datetime import datetime, timedelta
import statistics
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from itsdangerous import URLSafeTimedSerializer
DATABASE_URL = os.getenv("DATABASE_URL")
# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:PostgreSQL@localhost/women_support_app"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "super-secret-key"
app.config["SECRET_KEY"] = "email-secret-key"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"])
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://medical-risk-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================================================
# DATABASE MODELS
# =========================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user")
    is_verified = db.Column(db.Boolean, default=False)
    cyber_strikes = db.Column(db.Integer, default=0)
    is_blocked = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class CyberLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    text = db.Column(db.Text)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class LegalLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class HealthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.Text)
    prediction = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PeriodLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    last_period = db.Column(db.String(20))
    avg_cycle = db.Column(db.Integer)
    next_period = db.Column(db.String(20))
    ovulation_day = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# =========================================================
# LOAD ML MODELS
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def load_model(path):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

cyber_model = load_model("models/cyber_distilbert_model.pt")
legal_model = load_model("models/legal_distilbert_model.pt")

# =========================================================
# HEALTH ML MODEL
# =========================================================

SYMPTOM_LIST = [
    "irregular periods", "missed periods", "acne", "weight gain",
    "excess facial hair", "hair thinning", "dark neck patches",
    "ovarian cyst history", "fatigue", "cold sensitivity",
    "heat intolerance", "hair loss", "weight change",
    "dry skin", "constipation", "depression",
    "mood swings", "bloating", "sleep issues",
    "anxiety", "pelvic pain", "heavy bleeding", "low energy"
]

class HealthClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

health_model = HealthClassifier(len(SYMPTOM_LIST), 3)

if os.path.exists("models/health_model.pt"):
    health_model.load_state_dict(torch.load("models/health_model.pt", map_location=DEVICE))

health_model.to(DEVICE)
health_model.eval()

# =========================================================
# RULE-BASED SYMPTOM CHECKER
# =========================================================

CONDITION_RULES = {
    "PCOS": ["irregular periods", "acne", "weight gain", "excess facial hair"],
    "Thyroid Disorder": ["fatigue", "weight change", "hair loss", "cold sensitivity"],
    "Anemia": ["low energy", "fatigue", "heavy bleeding"],
    "Hormonal Imbalance": ["mood swings", "sleep issues", "bloating"],
}

# =========================================================
# AUTH ROUTES
# =========================================================

@app.route("/api/register", methods=["POST"])
def register():
    data = request.json

    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already exists"}), 400

    user = User(
        name=data["name"],
        email=data["email"],
        password_hash=bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    )

    db.session.add(user)
    db.session.commit()

    token = serializer.dumps(user.email)

    return jsonify({
        "message": "Registered successfully. Verify email.",
        "verification_token": token
    })

@app.route("/api/verify/<token>")
def verify_email(token):
    try:
        email = serializer.loads(token, max_age=3600)
        user = User.query.filter_by(email=email).first()
        user.is_verified = True
        db.session.commit()
        return jsonify({"message": "Email verified"})
    except:
        return jsonify({"error": "Invalid or expired token"}), 400

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    user = User.query.filter_by(email=data["email"]).first()

    if not user or not bcrypt.check_password_hash(user.password_hash, data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    if not user.is_verified:
        return jsonify({"error": "Email not verified"}), 403

    if user.is_blocked:
        return jsonify({"error": "Account blocked"}), 403

    token = create_access_token(identity={"id": user.id, "role": user.role})
    return jsonify({"access_token": token})

# =========================================================
# ADMIN PANEL
# =========================================================

@app.route("/api/admin/users")
@jwt_required()
def admin_users():
    identity = get_jwt_identity()
    if identity["role"] != "admin":
        return jsonify({"error": "Admin access required"}), 403

    users = User.query.all()
    return jsonify([{
        "id": u.id,
        "email": u.email,
        "blocked": u.is_blocked,
        "strikes": u.cyber_strikes
    } for u in users])

# =========================================================
# CYBER DETECTION
# =========================================================

@app.route("/api/cyber", methods=["POST"])
@jwt_required()
def detect_cyber():
    text = request.json.get("text", "")
    identity = get_jwt_identity()
    user = User.query.get(identity["id"])

    if user.is_blocked:
        return jsonify({"error": "Account blocked"}), 403

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = cyber_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs).item()

    label = "Cyberbullying" if prediction == 1 else "Safe"
    confidence = float(probs[0][prediction])

    if label == "Cyberbullying":
        user.cyber_strikes += 1
        if user.cyber_strikes >= 3:
            user.is_blocked = True

    db.session.add(CyberLog(
        user_id=user.id,
        text=text,
        prediction=label,
        confidence=confidence
    ))
    db.session.commit()

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "strikes": user.cyber_strikes,
        "blocked": user.is_blocked
    })

# =========================================================
# LEGAL DETECTION
# =========================================================

@app.route("/api/legal", methods=["POST"])
def detect_legal():
    text = request.json.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = legal_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs).item()

    label = "Legal Issue Detected" if prediction == 1 else "No Legal Issue"
    confidence = float(probs[0][prediction])

    db.session.add(LegalLog(text=text, prediction=label, confidence=confidence))
    db.session.commit()

    return jsonify({"prediction": label, "confidence": confidence})

# =========================================================
# HEALTH ML ANALYSIS
# =========================================================

@app.route("/api/health", methods=["POST"])
@jwt_required()
def analyze_health():
    symptoms = request.json.get("symptoms", [])
    vector = [1 if s in symptoms else 0 for s in SYMPTOM_LIST]
    input_tensor = torch.tensor([vector], dtype=torch.float32)

    with torch.no_grad():
        outputs = health_model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probs).item()

    labels = ["PCOS Risk", "Thyroid Risk", "Low Risk"]
    risk_score = float(probs[0][prediction])

    db.session.add(HealthLog(
        symptoms=", ".join(symptoms),
        prediction=labels[prediction],
        confidence=risk_score,
        risk_score=risk_score
    ))
    db.session.commit()

    return jsonify({
        "prediction": labels[prediction],
        "risk_score": risk_score
    })

# =========================================================
# SYMPTOM CHECKER
# =========================================================

@app.route("/api/symptom-checker", methods=["POST"])
@jwt_required()
def symptom_checker():
    symptoms = request.json.get("symptoms", [])
    matched = []

    for condition, rule_symptoms in CONDITION_RULES.items():
        match_count = len(set(symptoms) & set(rule_symptoms))
        if match_count >= 2:
            matched.append({"condition": condition, "score": match_count})

    if not matched:
        result = "No major condition detected"
        recommendation = "Monitor symptoms."
        risk_score = 0.2
    else:
        matched.sort(key=lambda x: x["score"], reverse=True)
        result = matched[0]["condition"]
        recommendation = "Consult doctor for confirmation."
        risk_score = 0.7

    db.session.add(HealthLog(
        symptoms=", ".join(symptoms),
        prediction=result,
        confidence=risk_score,
        risk_score=risk_score
    ))
    db.session.commit()

    return jsonify({
        "possible_condition": result,
        "recommendation": recommendation,
        "risk_score": risk_score,
        "matches": matched
    })

# =========================================================
# RISK TREND
# =========================================================

@app.route("/api/risk-trend")
@jwt_required()
def risk_trend():
    logs = HealthLog.query.order_by(HealthLog.created_at).all()
    scores = [log.risk_score for log in logs]

    if len(scores) < 3:
        moving_avg = scores
    else:
        moving_avg = [
            statistics.mean(scores[i-2:i+1])
            for i in range(2, len(scores))
        ]

    return jsonify({
        "risk_scores": scores,
        "moving_average": moving_avg
    })

# =========================================================
# PERIOD TRACKER
# =========================================================

@app.route("/api/period", methods=["POST"])
@jwt_required()
def calculate_period():
    data = request.json
    last_period = datetime.strptime(data["last_period"], "%Y-%m-%d")
    avg_cycle = int(data["avg_cycle"])

    next_period = last_period + timedelta(days=avg_cycle)
    ovulation_day = next_period - timedelta(days=14)

    db.session.add(PeriodLog(
        user_id=get_jwt_identity()["id"],
        last_period=data["last_period"],
        avg_cycle=avg_cycle,
        next_period=next_period.strftime("%Y-%m-%d"),
        ovulation_day=ovulation_day.strftime("%Y-%m-%d")
    ))
    db.session.commit()

    return jsonify({
        "next_period": next_period.strftime("%Y-%m-%d"),
        "ovulation_day": ovulation_day.strftime("%Y-%m-%d")
    })

# =========================================================
# INIT
# =========================================================

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)