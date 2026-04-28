# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import joblib
import numpy as np
from datetime import datetime

app = FastAPI(title="CKD Prediction Service")

# ---- Models for request/response ----
class Record(BaseModel):
    # 用 dict 以支持任意字段，不过也可定义具体 schema
    features: Dict[str, Any]

class PredictRequest(BaseModel):
    model: Optional[str] = "ensemble"
    task: Optional[str] = "ckd"
    records: List[Record]
    impute_strategy: Optional[str] = "auto"
    use_chain_predictors: Optional[bool] = True

# ---- load model registry (示意) ----
MODEL_REGISTRY = {
    "xgboost": joblib.load("models/xgb_ckd_v1.joblib"),
    "lgbm": joblib.load("models/lgbm_ckd_v1.joblib"),
    "logreg": joblib.load("models/logreg_ckd_v1.joblib"),
    # ...
}
IMPUTER_REGISTRY = {
    "median": joblib.load("imputers/median.joblib"),
    "iterative": joblib.load("imputers/iterative.joblib"),
    # ...
}

def impute_record(rec: dict, strategy="auto"):
    # 简化演示：若 strategy == "auto"则使用 iterative if available else median
    imputer = IMPUTER_REGISTRY.get("iterative", IMPUTER_REGISTRY["median"])
    # 这里需要把 dict 转成有固定顺序的向量，依赖 feature_list
    # 返回填充后的 record 和 metadata
    return rec, {"filled": {}}

def call_chain_predictors(rec):
    # 若缺失关键字段，调用其他病预测器（可以是本服务内部函数）
    # 例如预测 diabetes/hypertension/cvd，并返回概率
    return {"diabetes_prob": 0.2, "hypertension_prob": 0.7, "cvd_prob": 0.12}

@app.post("/predict")
def predict(req: PredictRequest):
    responses = []
    for i, r in enumerate(req.records):
        rec = r.features
        # 1) 检查缺失并impute
        filled_rec, impute_meta = impute_record(rec, req.impute_strategy)
        # 2) 若关键字段缺失并 use_chain_predictors
        chain_meta = {}
        if req.use_chain_predictors:
            chain_meta = call_chain_predictors(filled_rec)
            # 把 chain_meta 的概率并入特征向量
        # 3) 选择模型并预测
        model = MODEL_REGISTRY.get(req.model, MODEL_REGISTRY["xgboost"])
        # convert filled_rec -> vector X
        X = np.array([0.0])  # TODO
        proba = model.predict_proba(X.reshape(1, -1))[0,1]
        # optionally calibrated_proba, uncertainty via ensemble...
        responses.append({
            "id": i,
            "prediction": int(proba > 0.5),
            "probability": float(proba),
            "calibrated_probability": float(proba),
            "uncertainty_std": 0.03,
            "used_imputation": impute_meta,
            "used_chain_preds": chain_meta
        })
    return {"status":"ok", "data": responses, "meta": {"timestamp": datetime.utcnow().isoformat()}}
