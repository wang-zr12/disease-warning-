import pandas as pd
import pickle
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import os
from config_ import MODEL_DIR, MODEL_REGISTRY
from metrics_service import SHAPCalculator
from utils_ import is_onnx_model, ONNXModelWrapper

logger = logging.getLogger(__name__)
FEA_TOTAL_AMOUNT = 20  # 假设总特征数量为20，根据实际情况调整

from typing import Dict, List, Any

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

FEATURE_SETS = {
    'diabetes_full': DIABETES_FEATURES,
    'diabetes_basic': DIABETES_FEATURES_BASIC,
    'hypertension_full': HYPERTENSION_FEATURES,
    'hypertension_basic': HYPERTENSION_FEATURES_BASIC,
    'cvd_full': CVD_FEATURES,
    'cvd_basic': CVD_FEATURES_BASIC,
}

CKD_FEATURES = json.loads(Path(f"{MODEL_DIR}/ckd_features.json").read_text())



def preprocess_input_3(user_data: Dict[str, Any], required_features: List[str]) -> Dict[str, Any]:
    """
    从用户数据中提取模型所需的特征，如果缺失则报错

    Parameters:
    user_data : dict
        用户输入的完整数据字典
    required_features : list
        模型需要的特征列表（按顺序）
        例如: DIABETES_FEATURES_BASIC

    Returns:
    Dict[str, Any] : 包含所有必需特征的字典，键顺序与 ``required_features`` 一致

    Raises:
    ValueError : 当存在缺失或为None的特征时
    """
    missing_features: List[str] = []

    # 预先分配顺序字典，保持 required_features 的顺序
    result: Dict[str, Any] = {}

    for feature in required_features:
        if feature not in user_data:
            missing_features.append(feature)
            continue

        value = user_data[feature]
        if value is None:
            missing_features.append(feature)
            continue

        result[feature] = value

    if missing_features:
        raise ValueError(f"Feature Missing: {missing_features}")

    return result


def prepare_input_ckd(user_dict: Dict[str, Any]) -> (Dict[str, Any], Dict[str, Any]):
    """
    准备 CKD 模型输入，返回 **dict**（可直接喂给模型或前端）

    Returns
    -------
    result : tuple
        - features_dict : dict
            包含所有 CKD 特征的字典（含 missing 掩码）
        - imputation_meta : dict
    """
    # === 1. 加载 Imputer ===
    IMPUTER = None
    imputer_path = os.path.join(MODEL_DIR, "imputer_iterative.joblib")
    if os.path.exists(imputer_path):
        try:
            IMPUTER = joblib.load(imputer_path)
        except Exception as e:
            print(f"[Warning] Failed to load imputer: {e}")

    # === 2. 构建基础特征 ===
    base_cols = [f for f in CKD_FEATURES if not f.endswith("_missing")]
    raw = {col: user_dict.get(col, np.nan) for col in base_cols}
    X = pd.DataFrame([raw])

    # === 3. 强制 object → numeric ===
    object_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        print(f"[Info] Converting object columns to numeric: {object_cols}")
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # === 4. 添加 missing 掩码 ===
    for col in base_cols:
        missing_col = col + "_missing"
        X[missing_col] = X[col].isna().astype(int)

    # === 5. 填充缺失值（记录元信息）===
    imputation_meta = {
        "used_imputer": False,
        "imputed_fields": {}
    }

    if IMPUTER is not None and X.isna().any().any():
        # 对齐 imputer 训练时的列顺序
        if hasattr(IMPUTER, "feature_names_in_"):
            expected_cols = IMPUTER.feature_names_in_.tolist()
        else:
            expected_cols = CKD_FEATURES + ["proxy_diabetes_prob", "proxy_diabetes_missing"]

        cols_to_impute = [c for c in expected_cols if c in X.columns]
        X_sub = X[cols_to_impute]

        try:
            X_imp_array = IMPUTER.transform(X_sub)
            X_imp = pd.DataFrame(X_imp_array, columns=cols_to_impute, index=X.index)

            for col in cols_to_impute:
                if X[col].isna().any():
                    imputed_val = X_imp.loc[0, col]
                    X.loc[0, col] = imputed_val
                    imputation_meta["used_imputer"] = True
                    imputation_meta["imputed_fields"][col] = float(imputed_val)
        except Exception as e:
            print(f"[Error] Imputation failed: {e}")

    # === 6. 转为 dict（单行）===
    features_dict = X.iloc[0].to_dict()
    # 转为普通 Python 类型（避免 numpy 类型）
    features_dict = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else int(v) if isinstance(v, np.integer) else v
        for k, v in features_dict.items()
    }

    return features_dict, imputation_meta


def predict_all(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    对所有疾病进行预测，自动选择合适的模型版本

    Args:
        input_data: 输入特征字典

    Returns:
        results : dict
            包含每个疾病预测结果的字典
            例如:
            {
                "diabetes": {"prediction": 1, "confidence": 0.85, "risk": 85.0, "shap": {fea1: +5, fea2: -3, ...}},
                "hypertension": {"prediction": 0, "confidence": 0.90, "risk": 10.0, "shap": {...}},
                ...
            }
    """

    model_routing = choose_model_version(input_data)
    results = {}
    for disease, model_type in model_routing.items():
        # 获取所需特征列表
        key = f"{disease}_{model_type}"
        required_features = FEATURE_SETS[key]

        try:
            if disease == 'ckd':
                X, _ = prepare_input_ckd(input_data)
            else:
                X = preprocess_input_3(input_data, required_features)

            model = MODEL_REGISTRY[key]
            shap = SHAPCalculator().compute(model, X)

            # 只保留原本不为空的变量shap值
            valid_features = [k for k, v in X.items() if v is not None]
            shap['shap_values'] = {k: v for k, v in shap['shap_values'].items() if k in valid_features}

            X = np.array([list(X[feat] for feat in sorted(X.keys()))])  # 转Dict为2D数组

            # 根据模型类型执行预测
            if is_onnx_model(model):
                logger.info(f"使用ONNX模型预测 {disease}")
                prediction = int(ONNXModelWrapper.onnx_predict(model, X)[0])
                probabilities = ONNXModelWrapper.onnx_predict_proba(model, X)[0]
            else:
                logger.info(f"使用sklearn模型预测 {disease}")
                prediction = int(model.predict(X)[0])
                probabilities = model.predict_proba(X)[0]

            confidence = float(max(probabilities))
            risk = confidence * 100 if prediction == 1 else (1 - confidence) * 100

            results[disease] = {"prediction": prediction, "confidence": confidence, "risk": risk,
                                "shap": shap['shap_values']}
        except Exception as e:
            results[disease] = {"error": str(e)}

    return results


def choose_model_version(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据用户输入的特征列表，决定每个疾病使用哪个模型版本

    Parameters:
    input_features : Dict[str, Any]


    Returns:
    dict : 疾病名称到模型版本的映射
           例如: {'diabetes': 'full', 'hypertension': 'basic', 'cvd': 'full'}
           如果某个疾病无法预测（连basic版本都不满足），则不返回该疾病

    """
    input_features = [k for k, v in input_data.items() if v is not None]
    input_features_set = set(input_features)
    model_routing = {'ckd': 'default'}
    if 'DIQ010' not in input_features_set:
        if input_features_set.issuperset(DIABETES_FEATURES):
            model_routing['diabetes'] = 'full'
        elif input_features_set.issuperset(DIABETES_FEATURES_BASIC):
            model_routing['diabetes'] = 'basic'
        else:
            logger.info("User input features insufficient for diabetes model prediction")
    else:
        logger.info("User input contains diabetes info, skipping diabetes model prediction")
    if 'BPQ020' not in input_features_set:
        if input_features_set.issuperset(HYPERTENSION_FEATURES):
            model_routing['hypertension'] = 'full'
        elif input_features_set.issuperset(HYPERTENSION_FEATURES_BASIC):
            model_routing['hypertension'] = 'basic'
        else:
            logger.info("User input features insufficient for hypertension model prediction")
    else:
        logger.info("User input contains hypertension info, skipping hypertension model prediction")
    if input_features_set.issuperset(CVD_FEATURES):
        model_routing['cvd'] = 'full'
    elif input_features_set.issuperset(CVD_FEATURES_BASIC):
        model_routing['cvd'] = 'basic'

    return model_routing


def preprocess_input_all(input_data: Dict[str, Any], model) -> np.ndarray:
    """
    预处理输入数据，转换为模型可接受的格式

    Args:
        input_data: 输入特征字典
        model: 已加载的模型对象

    Returns:
        numpy数组
    """
    sorted_keys = sorted(input_data.keys())
    X = [input_data[key] for key in sorted_keys]

    if hasattr(model, 'feature_names_in_'):
        X = [input_data.get(feature, 0) for feature in model.feature_names_in_]

    # 转换为2D数组
    X = np.array([X])

    return X

def predict_single_disease(disease: str, input_data: Dict[str, Any], model_version: Optional[str] = None) -> Dict[str, Any]:
    """
    执行疾病预测

    Args:
        disease: 疾病类型
        input_data: 输入特征字典
        model_version: 模型版本（可选）

    Returns:
        包含预测结果的字典
    """
    # 构建模型路径
    if model_version:
        model_path = MODEL_DIR / f"{disease}_{model_version}.pkl"
    else:
        model_path = MODEL_DIR / f"{disease}.pkl"

    # 检查模型是否存在
    if not model_path.exists():
        available_models = list(MODEL_DIR.glob(f"{disease}*.pkl"))
        if available_models:
            available_versions = [
                f.stem.replace(f"{disease}_", "") if f.stem != disease else "default"
                for f in available_models
            ]
            error_msg = (
                f"模型 '{disease}' 版本 '{model_version or 'default'}' 不存在。"
                f"可用版本: {', '.join(available_versions)}"
            )
        else:
            error_msg = f"未找到疾病 '{disease}' 的任何模型"

        raise FileNotFoundError(error_msg)

    # 加载模型
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"成功加载模型: {model_path.name}")
    except Exception as e:
        raise ValueError(f"模型加载失败: {str(e)}")

    # 预处理输入数据
    try:
        X = preprocess_input_all(input_data, model)
    except Exception as e:
        raise ValueError(f"输入数据预处理失败: {str(e)}")

    # 执行预测
    try:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))

        # 如果是分类问题，获取所有类别的概率
        if hasattr(model, 'classes_'):
            class_probabilities = {
                str(cls): float(prob)
                for cls, prob in zip(model.classes_, probabilities)
            }
        else:
            class_probabilities = None

    except Exception as e:
        raise ValueError(f"预测执行失败: {str(e)}")

    # 构建返回结果
    result = {
        "disease": disease,
        "model_version": model_version or "default",
        "prediction": int(prediction) if isinstance(prediction, (np.integer, np.bool_)) else float(prediction),
        "confidence": confidence,
    }

    if class_probabilities:
        result["class_probabilities"] = class_probabilities

    return result


def get_model_info(disease: str, model_version: Optional[str] = None) -> Dict[str, Any]:
    """
    获取模型的元信息

    Args:
        disease: 疾病类型
        model_version: 模型版本

    Returns:
        模型信息字典
    """
    if model_version:
        model_path = MODEL_DIR / f"{disease}_{model_version}.pkl"
    else:
        model_path = MODEL_DIR / f"{disease}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    info = {
        "disease": disease,
        "version": model_version or "default",
        "model_type": type(model).__name__,
        "file_path": str(model_path),
        "file_size_mb": model_path.stat().st_size / (1024 * 1024)
    }

    # 添加特征信息
    if hasattr(model, 'feature_names_in_'):
        info["features"] = list(model.feature_names_in_)
        info["n_features"] = len(model.feature_names_in_)

    if hasattr(model, 'classes_'):
        info["classes"] = [str(c) for c in model.classes_]
        info["n_classes"] = len(model.classes_)

    return info
