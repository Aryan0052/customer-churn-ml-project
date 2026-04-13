from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import STATIC_DIR
from app.ml.pipeline import load_or_train_model


app = FastAPI(
    title="Clothing Recommendation API",
    description="Beginner-friendly outfit recommendation website with an ML model and fallback rules.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(router)


@app.on_event("startup")
async def startup_event() -> None:
    # Train once on startup if the saved model is not available yet.
    load_or_train_model()
