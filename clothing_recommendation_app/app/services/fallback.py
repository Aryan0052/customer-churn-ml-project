from __future__ import annotations

from app.schemas import RecommendationItem, RecommendationRequest


PLACEHOLDER_IMAGES = [
    "/static/images/casual-outfit.svg",
    "/static/images/formal-outfit.svg",
    "/static/images/streetwear-outfit.svg",
]

OUTFIT_IMAGES = {
    "casual": "/static/images/casual-outfit.svg",
    "formal": "/static/images/formal-outfit.svg",
    "streetwear": "/static/images/streetwear-outfit.svg",
    "palette": "/static/images/palette-outfit.svg",
    "accessory": "/static/images/accessory-outfit.svg",
}


def build_rule_based_recommendations(payload: RecommendationRequest) -> list[RecommendationItem]:
    recommendations: list[RecommendationItem] = []

    if payload.style_preference == "formal":
        title = "Tailored formal outfit"
        description = (
            f"Try a structured blazer, crisp shirt, and straight trousers in "
            f"colors that flatter {payload.skin_tone} skin."
        )
    elif payload.style_preference == "streetwear":
        title = "Relaxed streetwear outfit"
        description = (
            f"Use an oversized tee, relaxed cargo pants, and layered outerwear "
            f"that balance a {payload.body_type} frame."
        )
    else:
        title = "Easy casual outfit"
        description = (
            "Pair a clean tee or polo with well-fitted jeans and simple sneakers "
            "for a versatile everyday look."
        )

    recommendations.append(
        RecommendationItem(
            title=title,
            description=description,
            confidence=0.68,
            image_url=OUTFIT_IMAGES.get(payload.style_preference, OUTFIT_IMAGES["casual"]),
            source="rule-based",
        )
    )
    recommendations.append(
        RecommendationItem(
            title="Best color direction",
            description=f"Choose shades that complement {payload.skin_tone} skin tone and keep contrast balanced for a {payload.height} height profile.",
            confidence=0.64,
            image_url=OUTFIT_IMAGES["palette"],
            source="rule-based",
        )
    )
    recommendations.append(
        RecommendationItem(
            title="Fit and layering tip",
            description=f"Use proportions that support a {payload.body_type} body type and keep the styling aligned with {payload.style_preference} fashion.",
            confidence=0.61,
            image_url=OUTFIT_IMAGES["accessory"],
            source="rule-based",
        )
    )
    return recommendations
