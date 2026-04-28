from fastapi import FastAPI, Request
import requests
import uvicorn

app = FastAPI(title="Gateway Coordinator")

@app.post("/predict_ckd")
async def predict_ckd(req: Request):
    data = await req.json()
    print("Gateway received:", data)

    # 调用下游微服务测试
    try:
        resp = requests.post("http://10.0.0.35:5001/predict", json=data).json()
        print("CKD service response:", resp)
    except Exception as e:
        print("CKD service failed:", e)
        return {"status": "error", "message": str(e)}

    return {"status": "successful", "from": "gateway"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
