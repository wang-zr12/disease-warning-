from fastapi import APIRouter
from ..core_services.metrics_service import compute_metrics

router = APIRouter()

@router.get("/{disease}")
def get_metrics(disease: str):
    """
    返回指标板块，比如置信度、重要特征等
    """
    metrics = compute_metrics(disease)
    return metrics
