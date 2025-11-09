from fastapi import FastAPI
from routers import visualization, predict, metrics

app = FastAPI(title="Multi-Disease Prediction API")

# 注册路由
app.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])
app.include_router(predict.router, prefix="/prediction", tags=["Prediction"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
