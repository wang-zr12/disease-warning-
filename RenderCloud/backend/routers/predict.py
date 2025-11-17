import pandas as pd
from fastapi import APIRouter, Query
from typing import Optional
from pydantic import BaseModel
from core_services.predict_service import predict_disease, predict_all

router = APIRouter()


class PredictionRequest(BaseModel):
    input_data: dict


@router.post("/{disease}")
def predict(
        disease: str,
        req: PredictionRequest,
        model_version: Optional[str] = Query(None, description="模型版本，如 v1, v2")
):
    """
    接受JSON输入，返回预测值
    示例: POST /diabetes?model_version=v2
    """
    result = predict_disease(disease, req.input_data, model_version)
    return result


@router.post("/all")
def predict(req: PredictionRequest):
    result = predict_all(req.input_data)
    return result
