"""
Radiology Clinical Decision Support API
- DenseNet-169 (ImageNet-pretrained) for multi-label chest X-ray classification (NIH ChestXray14)
- Grad-CAM overlays per predicted class
- GPT-5 report generation (clinical + patient-friendly)

Endpoints
---------
GET  /health                      -> Basic health check
POST /predict                     -> Upload X-ray -> JSON probs + saved Grad-CAM overlays
POST /report                      -> JSON (preds + metadata + overlay paths) -> GPT-5 narrative
POST /predict_and_report          -> Upload X-ray + metadata -> runs both steps
GET  /classes                     -> Returns class list

Notes
-----
* Set env var OPENAI_API_KEY for GPT-5 calls.
* This API is for research/education; NOT a medical device. Include disclaimers in outputs.
* Adjust IMG_MEAN/STD, transforms, thresholds as you validate.
* Place your dataset images in IMG_ROOT or adapt loader for your storage.
"""

import os
import io
import json
import uuid
import uvicorn
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import DenseNet
from torchvision import models, transforms

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Config (move this above app.mount)
STATIC_DIR = os.getenv("STATIC_DIR", "static")
os.makedirs(STATIC_DIR, exist_ok=True)



# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES: List[str] = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
NUM_CLASSES = len(CLASSES)

MODEL_WEIGHTS = os.getenv("DENSENET169_WEIGHTS", "densenet169_chestxray14.pth")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# -----------------------------
# Image transforms
# -----------------------------
INFER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# -----------------------------
# Model
# -----------------------------

def build_model() -> nn.Module:
    model = models.densenet169(weights='DEFAULT')
    in_feats = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_feats, NUM_CLASSES),
        nn.Sigmoid()
    )
    return model


def disable_inplace_relu(module):
    """Recursively disable inplace operations in ReLU layers"""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            child.inplace = False
        else:
            disable_inplace_relu(child)

def densenet_forward_inplace_false(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=False)  # Force inplace=False
    out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
    out = self.classifier(out)
    return out

model = build_model().to(DEVICE)
if os.path.exists(MODEL_WEIGHTS):
    state = torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

# Disable inplace operations to fix Grad-CAM compatibility
disable_inplace_relu(model)

if isinstance(model, DenseNet):
    model.forward = densenet_forward_inplace_false.__get__(model, DenseNet)

model.eval()


# -----------------------------
# Grad-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self._fwd)
        self.bh = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        # Clone to avoid view issues
        self.activations = o.detach().clone()

    def _bwd(self, m, gi, go):
        # Clone to avoid view issues
        self.gradients = go[0].detach().clone()

    @torch.no_grad()
    def _norm(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def generate(self, class_idx: int, logits: torch.Tensor, x: torch.Tensor) -> np.ndarray:
        # Clear any previous gradients
        self.model.zero_grad()

        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(logits)
        one_hot[:, class_idx] = 1.0

        # Compute gradients
        (logits * one_hot).sum().backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured. Check hook registration.")

        acts = self.activations  # [B, K, h, w]
        grads = self.gradients  # [B, K, h, w]

        # Compute weights and cam
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)  # [B, h, w]
        cam = torch.relu(cam)

        cams = []
        for b in range(cam.shape[0]):
            c = self._norm(cam[b])
            c = torch.nn.functional.interpolate(
                c.unsqueeze(0).unsqueeze(0),
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
            cams.append(c.cpu().numpy())
        return np.stack(cams, axis=0)

    def __del__(self):
        """Clean up hooks when object is deleted"""
        if hasattr(self, 'fh'):
            self.fh.remove()
        if hasattr(self, 'bh'):
            self.bh.remove()


cam = GradCAM(model, model.features)


# -----------------------------
# Pydantic Schemas
# -----------------------------
class PatientMeta(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None  # "Male" | "Female" | "Other" | None
    symptoms: Optional[List[str]] = None
    notes: Optional[str] = None


class PredictResponse(BaseModel):
    image_id: str
    probabilities: Dict[str, float]
    top_classes: List[str]
    overlays: Dict[str, str]  # class -> URL path


class ReportRequest(BaseModel):
    probabilities: Dict[str, float]
    patient: Optional[PatientMeta] = None
    overlays: Optional[Dict[str, str]] = None
    mode: str = "clinician"  # or "patient"


class ReportResponse(BaseModel):
    report: str
    disclaimer: str


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Radiology CDS API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (after STATIC_DIR is defined)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}


# Utility: save overlay
import cv2


def save_overlay(pil_img: Image.Image, heat: np.ndarray, out_path: str):
    # Resize original image to match heatmap shape
    orig = np.array(pil_img.resize((heat.shape[1], heat.shape[0])))
    heat_uint8 = (255 * heat).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.35 * heatmap + 0.65 * orig).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)

# -----------------------------
# /predict
# -----------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), topk: int = 3):
    # Load image
    topk = int(topk)
    content = await file.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")
    x = INFER_TF(pil).unsqueeze(0).to(DEVICE)

    # Forward pass for probabilities (without gradients first)
    with torch.no_grad():
        probs = model(x).cpu().numpy()[0]

    prob_dict = {c: float(p) for c, p in zip(CLASSES, probs)}
    idxs = np.argsort(-probs)[:topk]
    top_classes = [CLASSES[i] for i in idxs]

    # Grad-CAM for top-k classes (separate forward pass with gradients)
    overlays: Dict[str, str] = {}
    for i in idxs:
        try:
            # Fresh forward pass for each class with gradients enabled
            x_grad = x.clone().detach().requires_grad_(True)
            logits = model(x_grad)

            heat = cam.generate(i, logits, x_grad)[0]  # [H, W]

            img_id = f"{uuid.uuid4().hex}__{CLASSES[i]}__{file.filename}.png"
            out_path = os.path.join(STATIC_DIR, img_id)
            save_overlay(pil, heat, out_path)
            overlays[CLASSES[i]] = f"/static/{img_id}"

            # Clear gradients for next iteration
            del x_grad, logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM for {CLASSES[i]}: {e}")
            continue

    return PredictResponse(
        image_id=f"{uuid.uuid4().hex}",
        probabilities=prob_dict,
        top_classes=top_classes,
        overlays=overlays
    )


# -----------------------------
# /report (GPT-5)
# -----------------------------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = (
    "You are a clinical decision support assistant for chest radiography. "
    "Summarize CNN predictions into a concise, structured note. "
    "Include: Findings, Supporting evidence, Differential considerations, and Next steps. "
    "Cite probabilities with two decimals. Tailor tone to the audience. "
    "Do NOT provide definitive diagnoses. Use hedging language."
)

DISCLAIMER = (
    "This AI-generated summary is for research and educational purposes only and is not a medical diagnosis. "
    "Clinical decisions must be made by qualified healthcare professionals."
)

CLINICIAN_TEMPLATE = (
    "Patient: age={age}, sex={sex}, symptoms={symptoms}.\n"
    "Model probabilities: {probs}.\n"
    "If provided, heatmaps per class are available at: {overlays}.\n"
    "Write a brief radiology-style impression for clinicians (bullet format), include suggested follow-up per guidelines when appropriate."
)

PATIENT_TEMPLATE = (
    "Patient-friendly explanation requested.\n"
    "Age={age}, sex={sex}, symptoms={symptoms}.\n"
    "Model probabilities: {probs}.\n"
    "Explain in simple language what the findings could mean, and suggest reasonable next steps to discuss with their doctor."
)


@app.post("/report", response_model=ReportResponse)
async def report(req: ReportRequest):
    if client is None:
        # Offline fallback: template fill without calling GPT
        meta = req.patient or PatientMeta()
        tmpl = CLINICIAN_TEMPLATE if req.mode == "clinician" else PATIENT_TEMPLATE
        text = tmpl.format(
            age=meta.age, sex=meta.sex, symptoms=meta.symptoms,
            probs=json.dumps(req.probabilities, indent=2),
            overlays=json.dumps(req.overlays or {}, indent=2)
        )
        return ReportResponse(report=text, disclaimer=DISCLAIMER)

    meta = req.patient or PatientMeta()
    user_prompt = (CLINICIAN_TEMPLATE if req.mode == "clinician" else PATIENT_TEMPLATE).format(
        age=meta.age, sex=meta.sex, symptoms=meta.symptoms,
        probs=json.dumps(req.probabilities), overlays=json.dumps(req.overlays or {})
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5",  # Changed from "gpt-5" as it doesn't exist yet
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        # Fallback if OpenAI API fails
        print(f"OpenAI API error: {e}")
        meta = req.patient or PatientMeta()
        tmpl = CLINICIAN_TEMPLATE if req.mode == "clinician" else PATIENT_TEMPLATE
        text = f"[API Error - Fallback Response]\n\n{tmpl.format(age=meta.age, sex=meta.sex, symptoms=meta.symptoms, probs=json.dumps(req.probabilities, indent=2), overlays=json.dumps(req.overlays or {}, indent=2))}"

    return ReportResponse(report=text, disclaimer=DISCLAIMER)


# -----------------------------
# /predict_and_report
# -----------------------------
@app.post("/predict_and_report")
async def predict_and_report(
        file: UploadFile = File(...),
        age: Optional[int] = Form(None),
        sex: Optional[str] = Form(None),
        symptoms: Optional[str] = Form(None),  # comma-separated
        mode: str = Form("clinician"),
        topk: int = Form(3)
):
    # Step 1: predict
    pred_res: PredictResponse = await predict(file, topk=topk)

    # Step 2: report
    meta = PatientMeta(
        age=age,
        sex=sex,
        symptoms=[s.strip() for s in symptoms.split(',')] if symptoms else None
    )
    rep_req = ReportRequest(
        probabilities=pred_res.probabilities,
        patient=meta,
        overlays=pred_res.overlays,
        mode=mode
    )
    rep_res = await report(rep_req)

    return JSONResponse({
        "prediction": pred_res.model_dump(),
        "report": rep_res.model_dump()
    })


# -----------------------------
# Local dev entrypoint
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("radiology_cds_api:app", host="0.0.0.0", port=8000, reload=True)