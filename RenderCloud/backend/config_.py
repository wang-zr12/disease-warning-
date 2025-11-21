# config.py
from pathlib import Path
from typing import Dict, Optional
import joblib
import onnxruntime as ort
import os
import json
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 返回到 project_root/
MODEL_DIR = Path(f"{PROJECT_ROOT}/backend/models")
DATA_DIR = Path(f"{PROJECT_ROOT}/data")
raw_data_file = DATA_DIR / "nhanes_2021_2023_master.csv"
POPULATION_STATS_FILE = DATA_DIR / "population_stats.json"
train_data_df = pd.read_csv(raw_data_file)
# 加载人群统计数据（懒加载）
_population_stats_cache = None

def load_population_stats():
    """加载人群统计数据"""
    global _population_stats_cache
    if _population_stats_cache is None:
        if POPULATION_STATS_FILE.exists():
            try:
                with open(POPULATION_STATS_FILE, 'r') as f:
                    _population_stats_cache = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load population stats: {e}")
                _population_stats_cache = {}
        else:
            _population_stats_cache = {}
    return _population_stats_cache

def get_population_stats():
    """获取人群统计数据"""
    return load_population_stats()


def safe_load_model(model_path):
    """
    尝试加载 joblib/pkl；失败时自动加载 ONNX。
    """
    try:
        print(f"Loading joblib model: {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        print(f"[WARN] Joblib load failed for {model_path}: {e}")

        # 替换扩展名为 onnx
        onnx_path = os.path.splitext(model_path)[0] + ".onnx"
        if not os.path.exists(onnx_path):
            raise ValueError(f"Neither joblib model nor ONNX exists for: {model_path}")

        print(f"Falling back to ONNX: {onnx_path}")
        return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


MODEL_REGISTRY = {
    # ---- CKD 保持不变 ----
    "ckd_default": joblib.load(f"{MODEL_DIR}/ckd_lgbm_mask.pkl"),

    # ---- 其他模型支持 fallback ----
    "diabetes_full": safe_load_model(f"{MODEL_DIR}/diabetes_full.joblib"),
    "diabetes_basic": safe_load_model(f"{MODEL_DIR}/diabetes_basic.joblib"),

    "hypertension_full": safe_load_model(f"{MODEL_DIR}/hypertension_full.joblib"),
    "hypertension_basic": safe_load_model(f"{MODEL_DIR}/hypertension_basic.joblib"),

    "cvd_full": safe_load_model(f"{MODEL_DIR}/cvd_full.joblib"),
    "cvd_basic": safe_load_model(f"{MODEL_DIR}/cvd_basic.joblib"),
}

# 默认模型版本（当疾病不在配置中时使用）
DEFAULT_MODEL_VERSION: Optional[str] = None  # None表示使用基础版本

# 支持的疾病列表（可选，用于验证）
SUPPORTED_DISEASES = ["diabetes", "heart_disease", "ckd"]

# 是否启用模型缓存
ENABLE_MODEL_CACHE = True

# 日志配置
LOG_LEVEL = "INFO"

# Feature Definitions

DIABETES_FEATURES = [
    'RIDAGEYR',  # Age (年龄)
    'BMXWAIST',  # Waist circumference (cm) (腰围)
    'RIAGENDR',  # Gender (性别)
    'BMXBMI',  # Body Mass Index (体重指数)
    'LBXGH',  # HbA1c (Hemoglobin A1c) (糖化血红蛋白)
    'LBXGLU',  # Fasting Glucose (空腹血糖)
    'LBDLDLSI',  # LDL cholesterol (mmol/L) (低密度脂蛋白胆固醇)
    'LUXCAPM',  # FVC, mL (用力肺活量)
    'BPXOSY1',  # Systolic Blood Pressure (收缩压)
    'LBXTC',  # Total Cholesterol (总胆固醇)
    'LBDHDD',  # HDL Cholesterol (高密度脂蛋白胆固醇)
    'LBXSTR',  # Triglycerides (甘油三酯)
    'SMQ020',  # Ever Smoked 100+ Cigarettes (是否吸过100支以上香烟)
    'SMQ040',  # Current Smoking Status (当前吸烟状态)
    'ALQ121',  # Alcohol (饮酒情况)
    'PAD680'  # Physical activity (体育活动)
]

DIABETES_FEATURES_BASIC = [
    'RIDAGEYR',  # Age (年龄)
    'RIAGENDR',  # Gender (性别)
    'BMXWAIST',  # Waist circumference (cm) (腰围)
    'BMXBMI',  # BMI (体重指数)
    'SMQ020',  # Ever Smoked 100+ Cigarettes (是否吸过100支以上香烟)
    'SMQ040',  # Current Smoking Status (当前吸烟状态)
    'ALQ121',  # Alcohol (饮酒情况)
    'PAD680'  # Physical activity (体育活动)
]

HYPERTENSION_FEATURES = [
    'RIDAGEYR',  # Age (年龄)
    'BMXBMI',  # Body Mass Index (体重指数)
    'RIAGENDR',  # Gender (性别)
    'BMXWAIST',  # Waist circumference (cm) (腰围)
    'BPXOSY1',  # Systolic BP 1st reading (第一次收缩压读数)
    'BPXOSY2',  # Systolic BP 2nd reading (第二次收缩压读数)
    'BPXOSY3',  # Systolic BP 3rd reading (第三次收缩压读数)
    'LBDGLUSI',  # Fasting glucose (空腹血糖)
    'SMQ020',  # Ever Smoked 100+ Cigarettes (是否吸过100支以上香烟)
    'ALQ121',  # Alcohol Days/Year (每年饮酒天数)
    'MCQ500',  # Ever told you had any kind of cancer (是否被告知患有癌症)
    'MCQ160A',  # Ever told you had arthritis (是否被告知患有关节炎)
    'SMQ040',  # Current Smoking Status (当前吸烟状态)
    'PAD680'  # Physical activity (体育活动)
]

HYPERTENSION_FEATURES_BASIC = [
    'RIDAGEYR',  # Age (年龄)
    'RIAGENDR',  # Gender (性别)
    'BMXBMI',  # BMI (体重指数)
    'BMXWAIST',  # Waist circumference (cm) (腰围)
    'SMQ020',  # Ever Smoked 100+ Cigarettes (是否吸过100支以上香烟)
    'SMQ040',  # Current Smoking Status (当前吸烟状态)
    'ALQ121',  # Alcohol (饮酒情况)
    'PAD680',  # Physical activity (体育活动)
    'MCQ500',  # Ever told you had any kind of cancer (是否被告知患有癌症)
    'MCQ160A'  # Ever told you had arthritis (是否被告知患有关节炎)
]

CVD_FEATURES = [
    'RIDAGEYR',  # Age (年龄)
    'RIAGENDR',  # Gender (性别)
    'BMXBMI',  # BMI (体重指数)
    'MCQ160D',  # Ever told you had angina/angina pectoris (是否被告知患有心绞痛)
    'LBDLDL',  # LDL Cholesterol (低密度脂蛋白胆固醇)
    'MCQ160P',  # Ever told you had COPD, emphysema, or chronic bronchitis (是否被告知患有慢阻肺/肺气肿/慢性支气管炎)
    'OSQ230',  # Any metal objects inside body (体内是否有金属物体)
    'MCQ160A',  # Ever told you had arthritis (是否被告知患有关节炎)
    'LBXTC',  # Total Cholesterol (总胆固醇)
    'BPXOSY1',  # Systolic Blood Pressure (收缩压)
    'SMQ020',  # Ever Smoked 100+ Cigarettes (是否吸过100支以上香烟)
    'SMQ040',  # Current Smoking Status (当前吸烟状态)
    'ALQ121',  # Alcohol (饮酒情况)
    'PAD680'  # Physical activity (体育活动)
]

CVD_FEATURES_BASIC = [
    'RIDAGEYR',  # Age (年龄)
    'RIAGENDR',  # Gender (性别)
    'BMXBMI',  # BMI (体重指数)
    'SMQ020',  # Ever Smoked 100+ Cigarettes (是否吸过100支以上香烟)
    'SMQ040',  # Current Smoking Status (当前吸烟状态)
    'ALQ121',  # Alcohol (饮酒情况)
    'PAD680',  # Physical activity (体育活动)
    'MCQ160D',  # Ever told you had angina/angina pectoris (是否被告知患有心绞痛)
    'MCQ160P',  # Ever told you had COPD, emphysema, or chronic bronchitis (是否被告知患有慢阻肺/肺气肿/慢性支气管炎)
    'OSQ230',  # Any metal objects inside body (体内是否有金属物体)
    'MCQ160A',  # Ever told you had arthritis (是否被告知患有关节炎)
]

CKD_FEATURES = json.loads(Path(f"{MODEL_DIR}/ckd_features.json").read_text())

FEATURE_SETS = {
    'ckd_default': CKD_FEATURES,
    'diabetes_full': DIABETES_FEATURES,
    'diabetes_basic': DIABETES_FEATURES_BASIC,
    'hypertension_full': HYPERTENSION_FEATURES,
    'hypertension_basic': HYPERTENSION_FEATURES_BASIC,
    'cvd_full': CVD_FEATURES,
    'cvd_basic': CVD_FEATURES_BASIC,
}

# Feature Modifiability Mapping
# True = 可以通过自我努力改变 (modifiable)
# False = 不可改变 (non-modifiable, 如年龄、性别、病史等)
FEATURE_MODIFIABILITY = {
    # 可修改的特征 (生活方式、体检指标等)
    'BMXBMI': True,
    'BMXWAIST': True,
    'BPXOSY1': True, 'BPXOSY2': True, 'BPXOSY3': True, 'BPXOSY4': True,
    'BPXSY1': True, 'BPXSY2': True, 'BPXSY3': True, 'BPXSY4': True,
    'BPXDI1': True, 'BPXDI2': True, 'BPXDI3': True, 'BPXDI4': True,
    'LBXGH': True,
    'LBXGLU': True,
    'LBDGLUSI': True,
    'LBDLDLSI': True,
    'LBDLDL': True,
    'LBXTC': True,
    'LBDHDD': True,
    'LBXSTR': True,
    'LBXSUA': True,
    'LBXSATSI': True,
    'LUXCAPM': True,
    'SMQ020': True,
    'SMQ040': True,
    'ALQ121': True,
    'PAD680': True,
    'INDFMPIR': True,
    
    # 不可修改的特征 (人口统计学、病史等)
    'RIDAGEYR': False,
    'RIAGENDR': False,
    'RIDRETH3': False,
    'MCQ160A': False,
    'MCQ160P': False,
    'MCQ160D': False,
    'MCQ500': False,
    'OSQ230': False,
    'BPQ020': False,
    'DIQ010': False,
}

# Feature Recommendations Mapping
# Short recommendations for each feature (in English)
FEATURE_RECOMMENDATIONS = {
    # 可修改的特征建议
    'BMXBMI': "Maintain a healthy weight through balanced diet and regular exercise",
    'BMXWAIST': "Reduce waist circumference by losing weight and doing core exercises",
    'BPXOSY1': "Lower blood pressure through diet, exercise, and stress management",
    'BPXOSY2': "Lower blood pressure through diet, exercise, and stress management",
    'BPXOSY3': "Lower blood pressure through diet, exercise, and stress management",
    'BPXOSY4': "Lower blood pressure through diet, exercise, and stress management",
    'BPXSY1': "Lower blood pressure through diet, exercise, and stress management",
    'BPXSY2': "Lower blood pressure through diet, exercise, and stress management",
    'BPXSY3': "Lower blood pressure through diet, exercise, and stress management",
    'BPXSY4': "Lower blood pressure through diet, exercise, and stress management",
    'BPXDI1': "Lower blood pressure through diet, exercise, and stress management",
    'BPXDI2': "Lower blood pressure through diet, exercise, and stress management",
    'BPXDI3': "Lower blood pressure through diet, exercise, and stress management",
    'BPXDI4': "Lower blood pressure through diet, exercise, and stress management",
    'LBXGH': "Control blood sugar through diet, exercise, and medication if needed",
    'LBXGLU': "Manage fasting glucose with healthy eating and regular physical activity",
    'LBDGLUSI': "Manage fasting glucose with healthy eating and regular physical activity",
    'LBDLDLSI': "Lower LDL cholesterol with diet changes and regular exercise",
    'LBDLDL': "Lower LDL cholesterol with diet changes and regular exercise",
    'LBXTC': "Reduce total cholesterol through diet and lifestyle modifications",
    'LBDHDD': "Increase HDL cholesterol through regular exercise and healthy fats",
    'LBXSTR': "Lower triglycerides by reducing sugar intake and increasing exercise",
    'LBXSUA': "Reduce uric acid by limiting purine-rich foods and staying hydrated",
    'LBXSATSI': "Improve liver health through diet, exercise, and limiting alcohol",
    'LUXCAPM': "Improve lung capacity through regular aerobic exercise and breathing exercises",
    'SMQ020': "Quit smoking to reduce health risks",
    'SMQ040': "Quit smoking to reduce health risks",
    'ALQ121': "Reduce alcohol consumption or quit drinking for better health",
    'PAD680': "Increase physical activity levels for better cardiovascular health",
    'INDFMPIR': "Consider financial planning and career development opportunities",
    
    # 不可修改的特征建议（一般性建议）
    'RIDAGEYR': "Focus on age-appropriate health screenings and preventive care",
    'RIAGENDR': "Follow gender-specific health guidelines and regular checkups",
    'RIDRETH3': "Be aware of any ethnicity-specific health risk factors",
    'MCQ160A': "Manage arthritis with medication, exercise, and joint care",
    'MCQ160P': "Follow COPD management plan with medications and pulmonary rehabilitation",
    'MCQ160D': "Manage angina with medications, lifestyle changes, and regular monitoring",
    'MCQ500': "Follow cancer survivorship care plan and regular follow-ups",
    'OSQ230': "Inform healthcare providers about metal implants before medical procedures",
    'BPQ020': "Manage hypertension with medications and lifestyle modifications",
    'DIQ010': "Manage diabetes with medications, diet, exercise, and regular monitoring",
}

def is_feature_modifiable(feature_name: str) -> bool:
    """
    判断特征是否可以通过自我努力改变
    
    Args:
        feature_name: 特征名称
        
    Returns:
        True if modifiable, False if non-modifiable
        如果特征不在映射中，默认返回 True (假设是可修改的)
    """
    return FEATURE_MODIFIABILITY.get(feature_name, True)  # 默认返回 True

def get_feature_recommendation(feature_name: str) -> Optional[str]:
    """
    获取特征的健康建议
    
    Args:
        feature_name: 特征名称
        
    Returns:
        建议字符串，如果特征不在映射中，返回 None
    """
    return FEATURE_RECOMMENDATIONS.get(feature_name, None)

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
