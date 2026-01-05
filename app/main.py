from fastapi import FastAPI
from app.routers.analyze import router as analyze_router

app = FastAPI(
    title="Accident AI Server",
    version="1.0.0"
)

app.include_router(analyze_router)