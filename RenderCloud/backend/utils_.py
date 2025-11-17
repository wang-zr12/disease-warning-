import numpy as np
import onnxruntime as ort
from typing import Union, Any
import logging

logger = logging.getLogger(__name__)
def is_onnx_model(model) -> bool:
    """判断是否为ONNX模型"""
    return isinstance(model, ort.InferenceSession)


class ONNXModelWrapper:
    """
    ONNX模型包装器，提供类似sklearn的接口供SHAP使用
    """

    def __init__(self, onnx_session: ort.InferenceSession):
        """
        初始化ONNX模型包装器

        Args:
            onnx_session: ONNX InferenceSession对象
        """
        if not isinstance(onnx_session, ort.InferenceSession):
            raise TypeError("onnx_session must be an onnxruntime.InferenceSession")

        self.session = onnx_session
        self.input_name = onnx_session.get_inputs()[0].name
        self.output_names = [output.name for output in onnx_session.get_outputs()]

        logger.info(f"ONNX模型输入: {self.input_name}")
        logger.info(f"ONNX模型输出: {self.output_names}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        X = self._prepare_input(X)
        outputs = self.session.run(None, {self.input_name: X})

        # 第一个输出通常是类别预测
        predictions = outputs[0]

        # 处理不同的输出格式
        if isinstance(predictions, np.ndarray):
            return predictions.flatten()
        else:
            return np.array(predictions).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        X = self._prepare_input(X)
        outputs = self.session.run(None, {self.input_name: X})

        # 尝试多种ONNX输出格式
        proba = self._extract_probabilities(outputs)

        return proba

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """
        准备输入数据，确保类型和形状正确

        Args:
            X: 输入数组

        Returns:
            np.ndarray: 准备好的输入数组 (float32)
        """
        # 确保是numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # 确保是2D数组
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 转换为float32（ONNX通常需要）
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        return X

    def _extract_probabilities(self, outputs: list) -> np.ndarray:
        """
        从ONNX输出中提取概率
        处理多种可能的输出格式

        Args:
            outputs: ONNX模型的输出列表

        Returns:
            np.ndarray: 概率数组 (n_samples, n_classes)
        """
        # 情况1: 有多个输出，第二个是概率（sklearn-onnx常见格式）
        if len(outputs) > 1:
            proba = outputs[1]
        else:
            proba = outputs[0]

        # 情况2: 输出是字典列表 (某些转换工具的格式)
        if isinstance(proba, list) and len(proba) > 0 and isinstance(proba[0], dict):
            # 提取字典中的值并转为数组
            proba = np.array([list(p.values()) for p in proba])

        # 情况3: 输出是1D数组（二分类单一概率）
        elif isinstance(proba, np.ndarray) and proba.ndim == 1:
            # 转换为 (n_samples, 2) 格式
            proba = np.column_stack([1 - proba, proba])

        # 情况4: 输出需要转换为numpy数组
        elif not isinstance(proba, np.ndarray):
            proba = np.array(proba)

        # 确保形状正确
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)

        return proba

    def __repr__(self) -> str:
        return f"ONNXModelWrapper(input={self.input_name}, outputs={self.output_names})"


def onnx_predict(model: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """
    便捷函数：使用ONNX模型进行预测

    Args:
        model: ONNX InferenceSession
        X: 输入数据 (n_samples, n_features)

    Returns:
        预测类别 (n_samples,)
    """
    wrapper = ONNXModelWrapper(model)
    return wrapper.predict(X)


def onnx_predict_proba(model: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """
    便捷函数：使用ONNX模型进行概率预测

    Args:
        model: ONNX InferenceSession
        X: 输入数据 (n_samples, n_features)

    Returns:
        预测概率 (n_samples, n_classes)
    """
    wrapper = ONNXModelWrapper(model)
    return wrapper.predict_proba(X)