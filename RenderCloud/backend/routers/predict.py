from fastapi import APIRouter
from pydantic import BaseModel
from core_services.predict_service import predict_disease

router = APIRouter()


class PredictionRequest(BaseModel):
    disease: str
    input_data: dict


@router.post("/")
def predict(req: PredictionRequest):
    """
    接受JSON输入，返回预测值和置信度
    """
    result = predict_disease(req.disease, req.input_data)
    return result
