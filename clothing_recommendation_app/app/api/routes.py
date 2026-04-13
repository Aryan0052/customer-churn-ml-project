from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import TEMPLATE_DIR
from app.schemas import (
    FeedbackRequest,
    ImageAnalysisResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.feedback import save_feedback
from app.services.image_analyzer import analyze_image_for_styling
from app.services.recommender import recommend_outfits


router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    return recommend_outfits(payload)


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)) -> ImageAnalysisResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    return analyze_image_for_styling(image_bytes)


@router.post("/feedback")
async def collect_feedback(payload: FeedbackRequest) -> dict:
    save_feedback(payload)
    return {"message": "Feedback saved for future model improvement."}
