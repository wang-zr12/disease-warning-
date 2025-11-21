import onnxruntime as ort
from onnx import ModelProto
from config_ import MODEL_DIR, get_model_version, get_all_diseases, train_data_df
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from utils_ import onnx_predict_proba, is_onnx_model
import pickle
from typing import Any, Dict, List, Optional
import numpy as np
import shap
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class MetricCalculator(ABC):
    """指标计算器抽象基类"""

    @abstractmethod
    def compute(self, model, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """计算指标"""
        pass


def is_tree_model(model) -> bool:
    if isinstance(model, ort.InferenceSession):
        return False

    model_name = str(type(model)).lower()
    tree_keywords = [
        "lgbm", "lightgbm", "booster",
        "xgb", "xgboost",
        "catboost",
        "randomforest", "random_forest",
        "gradientboosting", "gradient_boost",
        "sklearn.ensemble"
    ]
    return any(k in model_name for k in tree_keywords)


class SHAPCalculator(MetricCalculator):
    """SHAP值计算器"""

    def __init__(self, background_samples: int = 100, train_data_df=train_data_df):
        """
        初始化SHAP计算器

        Args:
            background_samples: 背景样本数量，用于KernelExplainer
        """
        self.background_samples = background_samples
        self.train_data_df = train_data_df

    def compute(
            self,
            model,
            input_data: Dict[str, Any],
            feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        try:
            if feature_names is None:
                feature_names = sorted(input_data.keys())

            X = np.array([[input_data[f] for f in feature_names]], dtype=float)

            # ==================== 1. 智能 predict_proba（只返回正类概率）===================
            if isinstance(model, ort.InferenceSession):
                def predict_proba_func(data):
                    # ONNX 输出可能是 [neg, pos] 或单值
                    return onnx_predict_proba(model, data)

                logger.info("ONNX 模型 → 包装 predict_proba (正类)")

            else:
                # 所有 sklearn 模型（包括 CalibratedClassifierCV）
                if not hasattr(model, "predict_proba"):
                    raise ValueError("模型必须支持 predict_proba")

                def predict_proba_func(data):
                    return model.predict_proba(data)[:, 1]

                logger.info(f"sklearn 模型 {type(model).__name__} → 原生 predict_proba")

            # ==================== 2. 终极智能 Explainer 选择器 ====================
            explainer = None
            sv_positive = None
            base_value = 0.0

            # 情况1：原始 LightGBM / XGBoost / RandomForest（非 ONNX）
            if not isinstance(model, ort.InferenceSession):
                if any(k in str(type(model)).lower() for k in ["lgbm", "xgb", "randomforest", "gradientboosting"]):
                    logger.info("原生树模型 → TreeExplainer")
                    explainer = shap.TreeExplainer(model)
                    raw = explainer.shap_values(X)
                    sv_positive = np.array(raw[1] if isinstance(raw, list) and len(raw) > 1 else raw).ravel()
                    base_value = float(
                        explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(
                            explainer.expected_value) > 1 else explainer.expected_value)

                # 情况2：CalibratedClassifierCV 或 纯 LogisticRegression → LinearExplainer
                elif isinstance(model, (CalibratedClassifierCV, LogisticRegression)):
                    logger.info(f"{type(model).__name__} → LinearExplainer（<1秒）")
                    background = self._safe_background(train_data_df, feature_names)
                    # Calibrated 也要用包装后的模型
                    explainer = shap.LinearExplainer(model, background)
                    sv_positive = explainer.shap_values(X).ravel()
                    base_value = float(explainer.expected_value)

                else:
                    # 兜底低采样 Kernel
                    logger.info("其他 sklearn 模型 → 低采样 KernelExplainer")
                    background = self._safe_background(train_data_df, feature_names, size=30)
                    explainer = shap.KernelExplainer(predict_proba_func, background)
                    raw = explainer.shap_values(X, nsamples=40, l1_reg="num_features(10)")
                    sv_positive = np.array(raw).ravel()
                    base_value = float(explainer.expected_value)

            # 情况3：ONNX 模型 → 自动检测内部结构
            else:

                logger.info("使用 KernelExplainer（ONNX）")

                # 构造更合理的背景数据集
                background = self._safe_background(train_data_df, feature_names, size=self.background_samples)
                for i, f in enumerate(feature_names):
                    val = float(input_data.get(f, 0))
                    scale = max(abs(val) * 0.2, 1e-3)
                    background[:, i] = np.clip(
                        np.random.normal(val, scale, self.background_samples),
                        -1e6, 1e6
                    )

                explainer = shap.KernelExplainer(predict_proba_func, background)
                shap_values = explainer.shap_values(X, nsamples=100)  # 可调精度

                # KernelExplainer 返回格式处理
                sv = np.asarray(shap_values)
                if sv.ndim == 3:
                    if sv.shape[1] == 2:  # (1, 2, n_features)
                        sv_positive = sv[0, 1, :]
                    elif sv.shape[2] == 2:  # (1, n_features, 2)
                        sv_positive = sv[0, :, 1]
                    else:
                        sv_positive = sv[0]
                else:
                    sv_positive = sv.ravel()

                # 取正类的 expected_value
                exp = explainer.expected_value
                if isinstance(exp, (list, np.ndarray)) and len(exp) > 1:
                    base_value = float(exp[1])
                else:
                    base_value = float(exp) if np.isscalar(exp) else float(exp.ravel()[0])

            # ==================== 3. 统一返回 ====================
            shap_dict = {f: float(v) for f, v in zip(feature_names, sv_positive)}
            sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

            return {
                "shap_values": shap_dict,
                "base_value": float(base_value),
                "feature_importance": [
                    {"feature": f, "importance": v} for f, v in sorted_items
                ]
            }

        except Exception as e:
            logger.error(f"SHAP计算失败: {e}", exc_info=True)
            raise ValueError(f"SHAP计算失败: {e}")

    def _safe_background(self, train_data_df, feature_names, size=None):
        """生成不含 NaN 的背景数据（关键！解决你的 NaN 错误）"""
        size = size or self.background_samples
        if train_data_df is not None and len(train_data_df) > 0:
            try:
                bg = train_data_df[feature_names].sample(n=min(size, len(train_data_df)), random_state=42)
                bg = bg.fillna(bg.median(numeric_only=True))  # 中位数填 NaN
                return bg.values
            except:
                pass
        # 用中位数或 0
        return np.zeros((size, len(feature_names)))


class FeatureImportanceCalculator(MetricCalculator):
    """特征重要性计算器"""

    def compute(
            self,
            model,
            input_data: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        """
        计算特征重要性

        Returns:
            特征重要性字典
        """
        try:
            feature_importance = {}

            # 树模型的特征重要性
            if hasattr(model, 'feature_importances_'):
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                else:
                    feature_names = sorted(input_data.keys())

                importance_dict = {
                    feature: float(importance)
                    for feature, importance in zip(feature_names, model.feature_importances_)
                }

                # 排序
                sorted_importance = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                feature_importance = {
                    "importance": importance_dict,
                    "ranked": [
                        {"feature": f, "importance": v}
                        for f, v in sorted_importance
                    ],
                    "top_5": [
                        {"feature": f, "importance": v}
                        for f, v in sorted_importance[:5]
                    ]
                }

            # 线性模型的系数
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]

                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                else:
                    feature_names = sorted(input_data.keys())

                coef_dict = {
                    feature: float(c)
                    for feature, c in zip(feature_names, coef)
                }

                sorted_coef = sorted(
                    coef_dict.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                feature_importance = {
                    "coefficients": coef_dict,
                    "ranked": [
                        {"feature": f, "coefficient": v}
                        for f, v in sorted_coef
                    ],
                    "intercept": float(model.intercept_) if hasattr(model, 'intercept_') else None
                }

            else:
                raise ValueError("模型不支持特征重要性计算")

            return feature_importance

        except Exception as e:
            logger.error(f"特征重要性计算失败: {str(e)}")
            raise ValueError(f"特征重要性计算失败: {str(e)}")


class MetricsService:
    """指标服务主类"""

    def __init__(self):
        """初始化指标服务"""
        self.calculators = {
            "shap": SHAPCalculator(),
            "feature_importance": FeatureImportanceCalculator(),
        }
        self._model_cache = {}

    def _load_model(self, disease: str, model_version: Optional[str] = None):
        """
        加载模型（带缓存）

        Args:
            disease: 疾病名称
            model_version: 模型版本（None时从配置读取）

        Returns:
            加载的模型对象
        """
        # 如果没有提供model_version，从配置读取
        if model_version is None:
            model_version = get_model_version(disease)

        cache_key = f"{disease}_{model_version or 'default'}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # 构建模型路径
        if model_version:
            model_path = MODEL_DIR / f"{disease}_{model_version}.pkl"
        else:
            model_path = MODEL_DIR / f"{disease}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # 缓存模型
        self._model_cache[cache_key] = model
        logger.info(f"模型已加载并缓存: {cache_key}")

        return model

    def compute_metric(
            self,
            metric_type: str,
            disease: str,
            input_data: Dict[str, Any],
            model_version: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        计算指定类型的指标

        Args:
            metric_type: 指标类型
            disease: 疾病名称，支持 'all'
            input_data: 输入数据
            model_version: 模型版本（None时从配置读取）
            **kwargs: 额外参数

        Returns:
            指标计算结果
        """
        # 处理 disease='all' 的情况
        if disease.lower() == 'all':
            return self._compute_metric_for_all_diseases(
                metric_type, input_data, model_version, **kwargs
            )

        # 单个疾病的指标计算
        return self._compute_metric_single(
            metric_type, disease, input_data, model_version, **kwargs
        )

    def _compute_metric_single(
            self,
            metric_type: str,
            disease: str,
            input_data: Dict[str, Any],
            model_version: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """计算单个疾病的指标"""
        # 验证指标类型
        if metric_type not in self.calculators:
            available_metrics = ", ".join(self.calculators.keys())
            raise ValueError(
                f"不支持的指标类型: {metric_type}. "
                f"可用指标: {available_metrics}"
            )

        # 加载模型（model_version为None时会从配置读取）
        try:
            model = self._load_model(disease, model_version)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"无法加载模型: {str(e)}")

        # 计算指标
        calculator = self.calculators[metric_type]
        try:
            result = calculator.compute(model, input_data, **kwargs)

            # 添加元信息
            result["meta"] = {
                "metric_type": metric_type,
                "disease": disease,
                "model_version": model_version,
                "model_type": type(model).__name__
            }

            return result

        except Exception as e:
            logger.error(f"指标计算失败 ({metric_type}): {str(e)}")
            raise

    def _compute_metric_for_all_diseases(
            self,
            metric_type: str,
            input_data: Dict[str, Any],
            model_version: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """为所有疾病计算指标"""
        all_diseases = get_all_diseases()
        logger.info(f"开始为 {len(all_diseases)} 种疾病计算 {metric_type} 指标")

        results = {}
        errors = {}

        for disease in all_diseases:
            try:
                result = self._compute_metric_single(
                    metric_type, disease, input_data, model_version, **kwargs
                )
                results[disease] = result
                logger.info(f"成功计算 {disease} 的 {metric_type} 指标")
            except Exception as e:
                error_msg = str(e)
                errors[disease] = error_msg
                logger.error(f"计算 {disease} 的 {metric_type} 指标失败: {error_msg}")

        # 构建响应
        response = {
            "mode": "all_diseases",
            "metric_type": metric_type,
            "total_diseases": len(all_diseases),
            "successful_calculations": len(results),
            "failed_calculations": len(errors),
            "results": results
        }

        if errors:
            response["errors"] = errors

        return response

    def compute_all_metrics(
            self,
            disease: str,
            input_data: Dict[str, Any],
            model_version: Optional[str] = None,
            metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        计算所有指标或指定的多个指标

        支持 disease='all'
        """
        if metrics is None:
            metrics = list(self.calculators.keys())

        # 如果是 'all' 疾病，遍历处理
        if disease.lower() == 'all':
            all_diseases = get_all_diseases()
            all_results = {}

            for d in all_diseases:
                results = {}
                errors = {}

                for metric_type in metrics:
                    try:
                        results[metric_type] = self._compute_metric_single(
                            metric_type, d, input_data, model_version
                        )
                    except Exception as e:
                        errors[metric_type] = str(e)

                all_results[d] = {"metrics": results}
                if errors:
                    all_results[d]["errors"] = errors

            return {
                "mode": "all_diseases",
                "results": all_results
            }

        # 单个疾病
        results = {}
        errors = {}

        for metric_type in metrics:
            try:
                results[metric_type] = self._compute_metric_single(
                    metric_type, disease, input_data, model_version
                )
            except Exception as e:
                errors[metric_type] = str(e)
                logger.error(f"计算 {metric_type} 失败: {str(e)}")

        response = {"metrics": results}
        if errors:
            response["errors"] = errors

        return response


# 创建全局实例
metrics_service = MetricsService()
