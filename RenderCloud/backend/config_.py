# config.py
from pathlib import Path
from typing import Dict, Optional
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 返回到 project_root/
MODEL_DIR = Path(f"{PROJECT_ROOT}/backend/models")
DATA_DIR = Path(f"{PROJECT_ROOT}/data")
raw_data_file = DATA_DIR / "nhanes_2021_2023_master.csv"

MODEL_REGISTRY = {
    "ckd_default": joblib.load(f"{MODEL_DIR}/ckd_lgbm.pkl"),
    "diabetes_full": joblib.load(f"{MODEL_DIR}/diabetes_full.joblib"),
    "diabetes_basic": joblib.load(f"{MODEL_DIR}/diabetes_basic.joblib"),
    "hypertension_full": joblib.load(f"{MODEL_DIR}/hypertension_full.joblib"),
    "hypertension_basic": joblib.load(f"{MODEL_DIR}/hypertension_basic.joblib"),
    "cvd_full": joblib.load(f"{MODEL_DIR}/cvd_full.joblib"),
    "cvd_basic": joblib.load(f"{MODEL_DIR}/cvd_basic.joblib"),
}

# 默认模型版本（当疾病不在配置中时使用）
DEFAULT_MODEL_VERSION: Optional[str] = None  # None表示使用基础版本

# 支持的疾病列表（可选，用于验证）
SUPPORTED_DISEASES = ["diabetes", "heart_disease", "ckd"]

# 是否启用模型缓存
ENABLE_MODEL_CACHE = True

# 日志配置
LOG_LEVEL = "INFO"


def get_model_version(disease: str) -> Optional[str]:
    """
    根据疾病名称获取对应的模型版本

    Args:
        disease: 疾病名称

    Returns:
        模型版本字符串或None
    """
    disease_model_map: Dict[str, Optional[str]] = {
        "diabetes": "full",
        "heart_disease": "full",
        "ckd": "default",
    }
    return disease_model_map.get(disease, DEFAULT_MODEL_VERSION)

def get_all_diseases() -> list:
    """
    获取所有支持的疾病名称列表

    Returns:
        疾病名称列表
    """
    return SUPPORTED_DISEASES
