import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = Path("models")

def train_and_save_model(disease: str, data, labels):
    """
    本地训练模型（不可通过API访问），保存最优模型
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 保存模型
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / f"{disease}.pkl", "wb") as f:
        pickle.dump(model, f)

    # 可以返回性能指标用于对比
    score = model.score(X_test, y_test)
    return {"accuracy": score}
