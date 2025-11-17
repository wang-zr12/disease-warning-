from fastapi import FastAPI
from routers import visualization, predict, metrics

app = FastAPI(title="Multi-Disease Prediction API")

# 注册路由
app.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])
app.include_router(predict.router, prefix="/prediction", tags=["Prediction"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])

from fastapi.middleware.cors import CORSMiddleware
# --- CORS 设置开始 ---
# 定义允许的来源
origins = [
    "http://localhost",
    "http://localhost:3000",  # React 默认端口
    "http://localhost:8080",  # Vue 默认端口
    "http://localhost:5173",  # Vite 默认端口
    "http://localhost:8501",  # Streamlit 默认端口
    # 可以在这里加上未来前端部署后的域名
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许上面列表中的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, etc.)
    allow_headers=["*"],  # 允许所有头部
)
# --- CORS 设置结束 ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Disease Warning API!"}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
