from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    height: str = Field(..., description="short, average, or tall")
    skin_tone: str = Field(..., description="fair, medium, or dark")
    body_type: str = Field(..., description="slim, athletic, or heavy")
    gender: str = Field(..., description="male, female, or unisex")
    style_preference: str = Field(..., description="casual, formal, or streetwear")
    occasion: str = Field("everyday", description="everyday, office, party, wedding, date, college, travel, or gym")
    season: str = Field("all-season", description="summer, winter, monsoon, or all-season")
    weather: str = Field("mild", description="hot, cold, rainy, or mild")
    budget: str = Field("mid-range", description="budget, mid-range, or premium")
    color_goal: str = Field("balanced", description="subtle, balanced, bold, monochrome, or colorful")


class RecommendationItem(BaseModel):
    title: str
    description: str
    confidence: float
    image_url: str
    source: str
    explanation: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    shopping_items: List[str] = Field(default_factory=list)
    color_palette: List[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    model_version: str
    used_fallback: bool
    style_strategy: str = ""


class ImageAnalysisResponse(BaseModel):
    dominant_colors: List[str]
    brightness: str
    contrast: str
    color_temperature: str
    color_harmony: str
    recommendations: List[RecommendationItem]


class FeedbackRequest(BaseModel):
    request: RecommendationRequest
    recommendation_title: str
    rating: int = Field(..., ge=1, le=5)
    liked: bool
    notes: Optional[str] = None
