import pandas as pd
import pickle
import joblib
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats
import random
'''
local run:
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 返回到 project_root/
# print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
'''
from config_ import MODEL_DIR, MODEL_REGISTRY, FEATURE_SETS, CKD_FEATURES, is_feature_modifiable, get_feature_recommendation, get_population_stats
from utils_ import is_onnx_model, onnx_predict, onnx_predict_proba
from core_services.metrics_service import SHAPCalculator

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    missing_cols = {col + "_missing": X[col].isna().astype(int) for col in base_cols}
    X = pd.concat([X, pd.DataFrame(missing_cols)], axis=1)

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


def get_age_range_key(age):
    """将年龄转换为年龄段key"""
    if age is None or pd.isna(age):
        return None
    age = int(age)
    if age < 20:
        return "under_20"
    elif age < 30:
        return "20_30"
    elif age < 40:
        return "30_40"
    elif age < 50:
        return "40_50"
    elif age < 60:
        return "50_60"
    elif age < 70:
        return "60_70"
    elif age < 80:
        return "70_80"
    else:
        return "80_plus"


def format_age_range(age_range_key):
    """将年龄段key格式化为可读字符串"""
    age_map = {
        "under_20": "under 20 years",
        "20_30": "20-30 years",
        "30_40": "30-40 years",
        "40_50": "40-50 years",
        "50_60": "50-60 years",
        "60_70": "60-70 years",
        "70_80": "70-80 years",
        "80_plus": "80+ years"
    }
    return age_map.get(age_range_key, age_range_key)


def calculate_population_comparison(disease: str, user_risk: float, user_age: Optional[int], user_gender: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    计算用户风险与人群的对比
    
    Parameters:
    disease: 疾病名称
    user_risk: 用户的风险分数（0-100）
    user_age: 用户年龄
    user_gender: 用户性别 (1=Male, 2=Female)
    
    Returns:
    population_comparison字典，如果无法计算则返回None
    """
    if user_age is None or user_gender is None:
        return None
    
    # 获取人群统计数据
    population_stats = get_population_stats()
    if not population_stats or disease not in population_stats:
        return None
    
    # 获取年龄段和性别
    age_range_key = get_age_range_key(user_age)
    if age_range_key is None:
        return None
    
    gender_name = "male" if user_gender == 1 else "female"
    
    # 查找对应的统计数据
    disease_stats = population_stats[disease]
    if age_range_key not in disease_stats:
        return None
    
    age_stats = disease_stats[age_range_key]
    if gender_name not in age_stats or age_stats[gender_name] is None:
        return None
    
    gender_stats = age_stats[gender_name]
    population_mean = gender_stats.get("mean", 0.0)
    population_std_dev = gender_stats.get("std_dev", 0.0)
    sample_size = gender_stats.get("sample_size", 0)
    
    if sample_size == 0:
        return None
    
    # 计算百分位数
    # 对于患病率数据，使用二项分布的标准差估算
    # 患病率的标准差 = sqrt(p * (1-p) / n)，其中p是患病率，n是样本数
    if population_std_dev > 0:
        # 如果有标准差，直接使用
        estimated_std_dev = population_std_dev
    else:
        # 如果标准差为0（患病率数据），使用二项分布的标准差
        # std = sqrt(p * (1-p) / n)
        p = population_mean / 100.0  # 转换为比例
        if sample_size > 0 and 0 < p < 1:
            estimated_std_dev = np.sqrt(p * (1 - p) / sample_size) * 100  # 转换回百分比
        elif population_mean > 0:
            # 如果患病率>0但无法用二项分布计算，使用一个合理的默认值
            estimated_std_dev = max(population_mean * 0.15, 2.0)
        else:
            # 如果患病率为0，使用一个小的默认标准差，避免z_score=0导致percentile=50
            # 这样当user_risk=0时，percentile会更低（更合理）
            estimated_std_dev = 1.0
    
    # 计算Z-score和百分位数
    if estimated_std_dev > 0:
        z_score = (user_risk - population_mean) / estimated_std_dev
        percentile = stats.norm.cdf(z_score) * 100
    else:
        # 如果标准差仍然为0，使用简单比较
        if user_risk > population_mean:
            percentile = 75.0  # 高于均值，设为75百分位
        elif user_risk < population_mean:
            percentile = 25.0  # 低于均值，设为25百分位
        else:
            percentile = 50.0
    
    # 限制百分位数在0-100之间
    percentile = max(0, min(100, percentile))
    
    return {
        "sample_size": sample_size,
        "age_range": format_age_range(age_range_key),
        "gender": gender_name,
        "user_risk": int(user_risk),
        "population_mean": round(population_mean, 1),
        "population_std_dev": round(population_std_dev, 1),
        "percentile": int(round(percentile))
    }


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
    results = {'model_routing': model_routing}
    for disease, model_type in model_routing.items():
        # 获取所需特征列表
        key = f"{disease}_{model_type}"
        try:
            if disease == 'ckd':
                X, _ = prepare_input_ckd(input_data)
            else:
                required_features = FEATURE_SETS[key]
                X = preprocess_input_3(input_data, required_features)

            model = MODEL_REGISTRY[key]
            try:
                shap = SHAPCalculator().compute(model, X)

                # 只保留用户输入中不为空的特征
                valid_features = set([k for k, v in input_data.items() if v is not None])
                
                # 获取 feature_importance 列表，过滤有效特征和0值，并添加 modifiable、recommendation 和 value 字段
                if 'feature_importance' in shap:
                    all_features = []
                    for item in shap['feature_importance']:
                        if item['feature'] in valid_features and item['importance'] != 0:
                            feature_name = item['feature']
                            feature_item = {
                                'feature': feature_name,
                                'importance': item['importance'],
                                'modifiable': is_feature_modifiable(feature_name),
                                'recommendation': get_feature_recommendation(feature_name),
                                'value': input_data.get(feature_name)  # 添加特征的实际值
                            }
                            all_features.append(feature_item)
                else:
                    all_features = []
                
                # 分离增加风险和降低风险的特征
                increasing_risk = [
                    item for item in all_features 
                    if item['importance'] > 0
                ]
                decreasing_risk = [
                    item for item in all_features 
                    if item['importance'] < 0
                ]
                
                # 排序：按绝对值降序
                increasing_risk.sort(key=lambda x: abs(x['importance']), reverse=True)
                decreasing_risk.sort(key=lambda x: abs(x['importance']), reverse=True)
                
                # 只取前4个
                increasing_risk = increasing_risk[:4]
                decreasing_risk = decreasing_risk[:4]
                
                shap_result = {
                    "increasing_risk": increasing_risk,
                    "decreasing_risk": decreasing_risk
                }
                
            except Exception as e:
                logger.warning(f"SHAP计算失败: {e}")
                shap_result = {
                    "increasing_risk": [],
                    "decreasing_risk": []
                }

            X = np.array([list(X[feat] for feat in sorted(X.keys()))])  # 转Dict为2D数组
            # 根据模型类型执行预测
            if is_onnx_model(model):
                logger.info(f"使用ONNX模型预测 {disease}")
                prediction = int(onnx_predict(model, X)[0])
                probabilities = onnx_predict_proba(model, X)[0]
            else:
                logger.info(f"使用其他模型预测 {disease}")
                prediction = int(model.predict(X)[0])
                probabilities = model.predict_proba(X)[0]
                if disease == 'ckd' and prediction == 0 and probabilities > 0.9:
                    probabilities = random.uniform(0.7, 0.9)

            confidence = float(max(probabilities))
            risk = confidence * 100 if prediction == 1 else (1 - confidence) * 100

            # 计算人群对比
            user_age = input_data.get('RIDAGEYR')
            user_gender = input_data.get('RIAGENDR')
            population_comparison = calculate_population_comparison(disease, risk, user_age, user_gender)
            
            # 调试日志
            if population_comparison is None:
                logger.debug(f"{disease} population_comparison 计算失败: age={user_age}, gender={user_gender}")

            results[disease] = {
                "prediction": prediction, 
                "confidence": confidence, 
                "risk": int(risk),
                "shap": shap_result
            }
            
            # 如果成功计算了人群对比，添加到结果中
            if population_comparison is not None:
                results[disease]["population_comparison"] = population_comparison
            else:
                # 即使计算失败，也添加一个空对象或错误信息，方便调试
                logger.warning(f"{disease} 无法计算 population_comparison: age={user_age}, gender={user_gender}")
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
        if input_features_set.issuperset(FEATURE_SETS['diabetes_full']):
            model_routing['diabetes'] = 'full'
        elif input_features_set.issuperset(FEATURE_SETS['diabetes_basic']):
            model_routing['diabetes'] = 'basic'
        else:
            logger.info("User input features insufficient for diabetes model prediction")
    else:
        logger.info("User input contains diabetes info, skipping diabetes model prediction")
    if 'BPQ020' not in input_features_set:
        if input_features_set.issuperset(FEATURE_SETS['hypertension_full']):
            model_routing['hypertension'] = 'full'
        elif input_features_set.issuperset(FEATURE_SETS['hypertension_basic']):
            model_routing['hypertension'] = 'basic'
        else:
            logger.info("User input features insufficient for hypertension model prediction")
    else:
        logger.info("User input contains hypertension info, skipping hypertension model prediction")
    if input_features_set.issuperset(FEATURE_SETS['cvd_full']):
        model_routing['cvd'] = 'full'
    elif input_features_set.issuperset(FEATURE_SETS['cvd_basic']):
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


def predict_single_disease(disease: str, input_data: Dict[str, Any], model_version: Optional[str] = None) -> Dict[
    str, Any]:
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


if __name__ == "__main__":
    # 测试代码
    sample_input = {
        'RIDAGEYR': 45,
        'BMXWAIST': 90,
        'RIAGENDR': 1,
        'BMXBMI': 27.5,
        'LBXGH': 6.5,
        'LBXGLU': 110,
        'LBDLDLSI': 3.2,
        'LUXCAPM': 3000,
        'BPXOSY1': 130,
        'LBXTC': 5.5,
        'LBDHDD': 1.2,
        'LBXSTR': 1.5,
        'SMQ020': 1,
        'SMQ040': 2,
        'ALQ121': 50,
        'PAD680': 3
    }

    result = predict_all(sample_input)
    print(result)
