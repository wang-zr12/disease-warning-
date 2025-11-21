from fastapi import APIRouter, Query, HTTPException, Path
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from core_services.metrics_service import metrics_service

router = APIRouter()
class MetricRequest(BaseModel):
    """指标计算请求"""
    disease: str = Field(
        ...,
        description="疾病类型，可以是具体疾病名或'all'"
    )
    input_data: Dict[str, Any] = Field(..., description="输入特征数据")
    # 移除 model_version 字段，由后端自动读取


class BatchMetricRequest(BaseModel):
    """批量指标计算请求"""
    disease: str = Field(
        ...,
        description="疾病类型，可以是具体疾病名或'all'"
    )
    input_data: Dict[str, Any] = Field(..., description="输入特征数据")
    metrics: Optional[List[str]] = Field(
        None,
        description="要计算的指标列表",
        example=["shap", "confidence"]
    )


@router.post("/{metric_type}")
async def compute_single_metric(
        metric_type: str = Path(
            ...,
            description="指标类型",
            example="shap"
        ),
        request: MetricRequest = None
):
    """
    计算单个指标

    支持的指标类型:
    - **shap**: SHAP值分析
    - **feature_importance**: 特征重要性
    - **confidence**: 置信度分析
    - **performance**: 模型性能指标

    示例请求:
```json
    {
        "disease": "diabetes",
        "input_data": {"age": 45, "bmi": 28.5, "glucose": 120},
        "model_version": "v2"
    }
```
    """
    try:
        result = metrics_service.compute_metric(
            metric_type=metric_type,
            disease=request.disease,
            input_data=request.input_data,
            model_version=request.model_version
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"指标计算失败: {str(e)}")


@router.post("/batch")
async def compute_multiple_metrics(request: BatchMetricRequest):
    """
    批量计算多个指标

    可以一次性计算多个指标，提高效率

    示例请求:
```json
    {
        "disease": "diabetes",
        "input_data": {"age": 45, "bmi": 28.5},
        "model_version": "v2",
        "metrics": ["shap", "confidence", "feature_importance"]
    }
```
    """
    try:
        result = metrics_service.compute_all_metrics(
            disease=request.disease,
            input_data=request.input_data,
            model_version=request.model_version,
            metrics=request.metrics
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量指标计算失败: {str(e)}")


@router.get("/available")
async def get_available_metrics():
    """
    获取所有可用的指标类型
    """
    return {
        "available_metrics": list(metrics_service.calculators.keys()),
        "descriptions": {
            "shap": "SHAP值分析 - 解释模型预测",
            "feature_importance": "特征重要性 - 显示哪些特征对模型最重要",
            "confidence": "置信度分析 - 预测的可信程度",
            "performance": "模型性能 - 准确率、召回率等指标"
        }
    }


@router.post("/cache/clear")
async def clear_model_cache():
    """
    清除模型缓存

    当模型文件更新后，可以调用此接口清除缓存
    """
    try:
        metrics_service.clear_cache()
        return {"message": "模型缓存已清除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除缓存失败: {str(e)}")

