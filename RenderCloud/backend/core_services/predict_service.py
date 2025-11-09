import pickle
from pathlib import Path

MODEL_DIR = Path("models")


def predict_disease(disease: str, input_data: dict):
    model_path = MODEL_DIR / f"{disease}.pkl"
    if not model_path.exists():
        return {"error": f"Model for {disease} not found"}

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 假设 input_data 是 dict -> 转成模型可接受的数组/向量
    X = [list(input_data.values())]
    pred = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    return {"prediction": pred, "confidence": confidence}
