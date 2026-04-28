from fastapi import FastAPI, Request
import uvicorn

app = FastAPI(title="CKD Prediction Service")

@app.post("/predict")
async def predict_ckd(req: Request):
    data = await req.json()
    print("CKD received:", data)
    return {"service": "ckd", "status": "successful"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
