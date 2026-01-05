from fastapi import FastAPI
from app.routers import analyze, damage, drive, pipeline

app = FastAPI(title="Car Damage Detection API")

app.include_router(analyze.router)
app.include_router(damage.router)
app.include_router(drive.router)
app.include_router(pipeline.router)