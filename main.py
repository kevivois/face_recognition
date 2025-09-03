import os, io, base64
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from deepface import DeepFace

# --------- Config ---------
MODEL_NAME = os.getenv("MODEL_NAME", "Facenet512")  # alternatives: "Facenet", "ArcFace"
DETECTOR   = os.getenv("DETECTOR", "mediapipe")     # + stable sur Render que retinaface
ALIGN      = True
ENFORCE    = True  # lèvera ValueError s'il n'y a pas de visage

# --------- FastAPI ---------
app = FastAPI(title="Face Recognition API")

ALLOWED_ORIGINS = [ "http://localhost:37729"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,           # ex: ["http://localhost:59371","https://ton-domaine"]
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,                 # IMPORTANT: laisse False si tu utilises "*" ou regex
)

# --------- Schemas ---------
class FaceReq(BaseModel):
    image_base64: Optional[str] = Field(None, description="PNG/JPEG base64 sans header data:")
    img_path: Optional[str]     = Field(None, description="Chemin serveur (debug)")

class VerifyReq(BaseModel):
    # soit 2 images, soit 1 image + un template d'embedding
    probe_base64: Optional[str] = None
    ref_base64: Optional[str]   = None
    ref_embedding: Optional[List[float]] = None
    metric: str = Field("cosine", pattern="^(cosine|l2)$")
    threshold: Optional[float] = None  # si None, on met des valeurs par défaut

def _b64_to_bgr(image_base64: str) -> np.ndarray:
    # accepte b64 avec ou sans "data:image/jpeg;base64,"
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(img)[:, :, ::-1]  # RGB -> BGR pour OpenCV/DeepFace
    return arr

def _pick_largest(result: List[Dict[str, Any]]) -> Dict[str, Any]:
    def area(r):
        fa = r.get("facial_area", {})
        return int(fa.get("w", 0)) * int(fa.get("h", 0))
    return max(result, key=area)

def _represent_bgr(bgr: np.ndarray) -> Dict[str, Any]:
    reps = DeepFace.represent(
        img_path=bgr,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        align=ALIGN,
        enforce_detection=ENFORCE,
    )
    return _pick_largest(reps) if isinstance(reps, list) else reps

def _cosine(a: List[float], b: List[float]) -> float:
    import math
    s = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return s / max(na*nb, 1e-9)

def _l2(a: List[float], b: List[float]) -> float:
    import math
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "detector": DETECTOR}

@app.post("/face/represent")
def represent(req: FaceReq):
    try:
        if not req.image_base64 and not req.img_path:
            return {"ok": False, "error": "Provide image_base64 or img_path"}
        if req.image_base64:
            bgr = _b64_to_bgr(req.image_base64)
        else:  # chemin serveur (debug local)
            bgr = np.asarray(Image.open(req.img_path).convert("RGB"))[:, :, ::-1]
        r = _represent_bgr(bgr)
        return {"ok": True, "model": MODEL_NAME, "embedding": r["embedding"], "facial_area": r.get("facial_area")}
    except ValueError as ve:
        return {"ok": False, "error": f"{ve}"}
    except Exception as e:
        return {"ok": False, "error": f"Unhandled: {e}"}

@app.post("/face/verify")
def verify(req: VerifyReq):
    try:
        if not req.probe_base64:
            return {"ok": False, "error": "probe_base64 required"}

        # Embedding probe
        probe = _represent_bgr(_b64_to_bgr(req.probe_base64))["embedding"]

        # Embedding ref: soit image, soit vecteur fourni
        if req.ref_embedding is not None:
            ref = req.ref_embedding
        elif req.ref_base64:
            ref = _represent_bgr(_b64_to_bgr(req.ref_base64))["embedding"]
        else:
            return {"ok": False, "error": "Provide ref_base64 or ref_embedding"}

        metric = req.metric
        if metric == "cosine":
            score = _cosine(probe, ref)
            threshold = req.threshold if req.threshold is not None else 0.70
            match = score >= threshold
        else:
            score = _l2(probe, ref)
            threshold = req.threshold if req.threshold is not None else 1.20
            match = score <= threshold

        return {"ok": True, "match": match, "score": score, "metric": metric, "threshold": threshold}
    except ValueError as ve:
        return {"ok": False, "error": f"{ve}"}
    except Exception as e:
        return {"ok": False, "error": f"Unhandled: {e}"}
