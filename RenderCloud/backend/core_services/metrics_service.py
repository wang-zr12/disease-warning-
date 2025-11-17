# 在文件开头添加
from config_ import MODEL_DIR, get_model_version, get_all_diseases
from utils_ import ONNXModelWrapper, is_onnx_model
import pickle
from typing import Any, Dict, List, Optional
import numpy as np
import shap
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

class MetricCalculator(ABC):
    """指标计算器抽象基类"""

    @abstractmethod
    def compute(self, model, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """计算指标"""
        pass


class SHAPCalculator(MetricCalculator):
    """SHAP值计算器"""

    def __init__(self, background_samples: int = 100):
        """
        初始化SHAP计算器

        Args:
            background_samples: 背景样本数量，用于KernelExplainer
        """
        self.background_samples = background_samples

    def compute(
            self,
            model,
            input_data: Dict[str, Any],
            feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        计算SHAP值

        Args:
            model: 已加载的模型
            input_data: 输入特征字典
            feature_names: 特征名称列表

        Returns:
            包含SHAP值的字典
        """
        try:
            # 准备输入数据
            if feature_names is None:
                feature_names = sorted(input_data.keys())

            X = np.array([[input_data[f] for f in feature_names]])

            # 如果是ONNX模型，包装一下
            if is_onnx_model(model):
                model = ONNXModelWrapper(model)
                logger.info("使用ONNX模型进行SHAP计算")

            # 根据模型类型选择合适的explainer
            if hasattr(model, 'tree_'):
                # 树模型（决策树、随机森林、LGBM等）
                explainer = shap.TreeExplainer(model)
                logger.info("使用TreeExplainer")
            else:
                # 其他模型使用KernelExplainer
                # 创建背景数据（使用零值或中位数）
                background = np.zeros((self.background_samples, len(feature_names)))

                # 尝试使用更有意义的背景数据
                # 可以基于输入数据的范围生成
                for i, feature in enumerate(feature_names):
                    if feature in input_data:
                        val = input_data[feature]
                        if isinstance(val, (int, float)):
                            # 在当前值附近生成背景样本
                            background[:, i] = np.random.normal(val, abs(val) * 0.1, self.background_samples)

                explainer = shap.KernelExplainer(model.predict_proba, background)
                logger.info("使用KernelExplainer")

            # 计算SHAP值
            shap_values = explainer.shap_values(X)

            # 处理多分类情况
            if isinstance(shap_values, list):
                # 多分类：取正类（通常是第二个类别）的SHAP值
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # 转换为1D数组
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            # 转换为字典格式
            shap_dict = {
                feature: float(value)
                for feature, value in zip(feature_names, shap_values)
            }

            # 排序（按绝对值）
            sorted_features = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # 获取基准值
            base_value = 0.0
            if hasattr(explainer, 'expected_value'):
                expected = explainer.expected_value
                if isinstance(expected, (list, np.ndarray)):
                    # 多分类情况，取正类的期望值
                    base_value = float(expected[1] if len(expected) > 1 else expected[0])
                else:
                    base_value = float(expected)

            return {
                "shap_values": shap_dict,
                "feature_importance": [
                    {"feature": f, "importance": v}
                    for f, v in sorted_features
                ],
                "base_value": base_value
            }

        except Exception as e:
            logger.error(f"SHAP计算失败: {str(e)}", exc_info=True)
            raise ValueError(f"SHAP计算失败: {str(e)}")

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