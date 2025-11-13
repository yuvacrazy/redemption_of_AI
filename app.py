# app.py
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
import uvicorn

# ---------------------------
# CONFIG (override via env vars in production)
# ---------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "salary_model_pipeline.pkl")
TFIDF_PATH = os.getenv("TFIDF_PATH", "job_title_tfidf.joblib")
SVD_PATH = os.getenv("SVD_PATH", "job_title_svd.joblib")
META_PATH = os.getenv("META_PATH", "preprocessing_meta.joblib")
API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("smartpay-api")

# ---------------------------
# LOAD ARTIFACTS
# ---------------------------
try:
    pipeline = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    svd = joblib.load(SVD_PATH)
    meta = joblib.load(META_PATH)
    model_input_cols = meta["model_input_cols"]
    svd_cols = meta["svd_cols"]
    logger.info("Model, TF-IDF, SVD and metadata loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model/artifacts: %s", e)
    raise RuntimeError("Model or preprocessing artifacts failed to load. Check MODEL_PATH/TFIDF_PATH/SVD_PATH/META_PATH.")

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="SmartPay Salary Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request schema (reduced fields)
# ---------------------------
class PredictRequest(BaseModel):
    experience_level: str = Field(..., description="Experience level (e.g. ENTR/MID/SEN/SE)")
    employment_type: str = Field(..., description="Employment type (e.g. Full-time)")
    job_title: str = Field(..., description="Job title / role (free text)")
    employee_residence: str = Field(..., description="Country or region code")
    company_location: str = Field(..., description="Company country or location")
    remote_ratio: float = Field(0.0, ge=0.0, le=100.0)
    company_size: str = Field(..., description="Company size (Small/Medium/Large)")
    education: str = Field("unknown", description="Education level (optional)")

    @validator('*', pre=True)
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

class PredictResponse(BaseModel):
    predicted_salary_usd: float

# ---------------------------
# API key auth
# ---------------------------
def api_key_auth(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return True

# ---------------------------
# Helper: convert single payload -> pipeline input DataFrame
# ---------------------------
def build_input_df(payload: dict):
    """
    Build DataFrame with the same columns the pipeline was trained on.
    This uses tfidf + svd to convert job_title -> svd_cols, then composes all input columns.
    """
    # 1) base row from metadata input columns (fill with NaN)
    base = {c: np.nan for c in model_input_cols}

    # 2) fill categorical/numeric columns from payload where available
    for k, v in payload.items():
        key = k
        # dataset columns used snake_case; accept either snake or hyphen/space variants if necessary
        if key in base:
            base[key] = v

    # 3) compute job_title SVD columns
    job_title_text = payload.get("job_title", "")
    if job_title_text is None:
        job_title_text = ""
    # ensure str
    jt = str(job_title_text)
    try:
        tf = tfidf.transform([jt])
        sv = svd.transform(tf)
        # sv is 2D array (1, n_components)
        for i, col in enumerate(svd_cols):
            base[col] = float(sv[0, i])
    except Exception as e:
        # fallback: zeros for SVD components
        logger.warning("Job title vectorization failed: %s", e)
        for i, col in enumerate(svd_cols):
            base[col] = 0.0

    # 4) ensure numeric types for remote_ratio etc.
    if "remote_ratio" in base and base["remote_ratio"] is not None:
        try:
            base["remote_ratio"] = float(base["remote_ratio"])
        except Exception:
            base["remote_ratio"] = 0.0

    # 5) return single-row DataFrame
    return pd.DataFrame([base])

# ---------------------------
# Predict endpoint
# ---------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, auth: bool = Depends(api_key_auth)):
    payload = req.dict()
    try:
        X = build_input_df(payload)
        pred = pipeline.predict(X)[0]            # pipeline returns USD value (no log transform here)
        return PredictResponse(predicted_salary_usd=float(pred))
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

# ---------------------------
# Health & root
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.get("/")
def root():
    return {"service": "SmartPay Salary Prediction API", "status": "running"}

# ---------------------------
# Run (for local dev)
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
