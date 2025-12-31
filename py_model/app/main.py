from fastapi import FastAPI
from app.routers import damage_detect

app = FastAPI(title="Car Damage Detection API")

app.include_router(damage_detect.router, prefix="/damage")