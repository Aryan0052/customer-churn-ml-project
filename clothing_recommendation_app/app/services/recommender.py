from __future__ import annotations

from collections import defaultdict

import pandas as pd

from app.ml.pipeline import FEATURE_COLUMNS, TARGET_COLUMNS, load_or_train_model
from app.schemas import RecommendationItem, RecommendationRequest, RecommendationResponse
from app.services.fallback import OUTFIT_IMAGES, build_rule_based_recommendations


OCCASION_ITEMS = {
    "office": ["structured blazer", "pressed trousers", "leather loafers"],
    "party": ["statement jacket", "dark denim", "clean sneakers"],
    "wedding": ["tailored suit or ethnic jacket", "polished shoes", "minimal accessories"],
    "date": ["textured overshirt", "well-fitted jeans", "signature fragrance"],
    "college": ["layered tee", "comfortable jeans", "backpack-friendly jacket"],
    "travel": ["wrinkle-resistant overshirt", "stretch trousers", "comfortable sneakers"],
    "gym": ["performance t-shirt", "training joggers", "supportive trainers"],
    "everyday": ["versatile top", "balanced bottoms", "clean footwear"],
}

WEATHER_RULES = {
    "hot": "Choose breathable cotton, linen, relaxed fits, and lighter colors.",
    "cold": "Add thermal layers, wool textures, jackets, and deeper colors.",
    "rainy": "Use quick-dry layers, darker bottoms, waterproof footwear, and avoid long hems.",
    "mild": "Use light layering so the outfit can adapt through the day.",
}

SEASON_RULES = {
    "summer": "Use breathable fabrics and avoid bulky layering.",
    "winter": "Prioritize layering, heavier textures, and closed footwear.",
    "monsoon": "Prefer darker hems, quick-dry fabrics, and water-friendly shoes.",
    "all-season": "Keep the outfit modular so layers can be added or removed.",
}

BUDGET_RULES = {
    "budget": "Focus on versatile basics first: one neutral top, one good bottom, and one clean shoe.",
    "mid-range": "Mix quality basics with one statement item so the outfit feels upgraded.",
    "premium": "Invest in tailoring, fabric quality, leather footwear, and signature accessories.",
}

COLOR_GOALS = {
    "subtle": ["cream", "stone", "charcoal"],
    "balanced": ["navy", "olive", "white"],
    "bold": ["black", "rust", "cobalt"],
    "monochrome": ["black", "charcoal", "grey"],
    "colorful": ["emerald", "burgundy", "sky blue"],
}


def _build_description(target_map: dict[str, str]) -> str:
    return (
        f"{target_map['fit_recommendation'].capitalize()}, "
        f"{target_map['top_recommendation']}, "
        f"colors: {target_map['color_recommendation']}, "
        f"plus {target_map['extra_recommendation']}."
    )


def _advanced_strategy(payload: RecommendationRequest) -> tuple[str, list[str], list[str], str]:
    occasion_items = OCCASION_ITEMS.get(payload.occasion, OCCASION_ITEMS["everyday"])
    weather_tip = WEATHER_RULES.get(payload.weather, WEATHER_RULES["mild"])
    season_tip = SEASON_RULES.get(payload.season, SEASON_RULES["all-season"])
    budget_tip = BUDGET_RULES.get(payload.budget, BUDGET_RULES["mid-range"])
    palette = COLOR_GOALS.get(payload.color_goal, COLOR_GOALS["balanced"])
    strategy = (
        f"For {payload.occasion}, keep the base {payload.style_preference}, adapt it for "
        f"{payload.weather} weather and {payload.season}, then use a {payload.color_goal} color direction."
    )
    explanation = f"{weather_tip} {season_tip} {budget_tip}"
    return strategy, occasion_items, palette, explanation


def _complete_card(
    item: RecommendationItem,
    payload: RecommendationRequest,
    index: int,
) -> RecommendationItem:
    strategy, occasion_items, palette, explanation = _advanced_strategy(payload)
    item.tags = [
        payload.style_preference,
        payload.occasion,
        payload.weather,
        payload.budget,
        payload.color_goal,
    ]
    item.shopping_items = occasion_items
    item.color_palette = palette
    item.explanation = f"{strategy} {explanation}"
    item.confidence = round(max(0.05, item.confidence - (0.03 * index)), 2)
    return item


def _image_for_payload(payload: RecommendationRequest, variation: str = "style") -> str:
    if variation == "palette":
        return OUTFIT_IMAGES["palette"]
    if variation == "accessory":
        return OUTFIT_IMAGES["accessory"]
    return OUTFIT_IMAGES.get(payload.style_preference, OUTFIT_IMAGES["casual"])


def _generate_ranked_recommendations(pipeline, payload: RecommendationRequest) -> list[RecommendationItem]:
    input_frame = pd.DataFrame([payload.model_dump()])[FEATURE_COLUMNS]
    prediction_matrix = pipeline.predict(input_frame)[0]
    transformed_input = pipeline.named_steps["preprocessor"].transform(input_frame)
    probability_sets = pipeline.named_steps["model"].predict_proba(transformed_input)

    target_map = dict(zip(TARGET_COLUMNS, prediction_matrix))
    probability_map: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for target_name, estimator_probabilities, estimator in zip(
        TARGET_COLUMNS,
        probability_sets,
        pipeline.named_steps["model"].estimators_,
    ):
        class_names = estimator.classes_
        scores = estimator_probabilities[0]
        ranked = sorted(zip(class_names, scores), key=lambda row: row[1], reverse=True)
        probability_map[target_name] = ranked

    recommendations: list[RecommendationItem] = []

    base_confidence = sum(
        probability_map[target_name][0][1] for target_name in TARGET_COLUMNS
    ) / len(TARGET_COLUMNS)
    recommendations.append(
        RecommendationItem(
            title="Best overall outfit",
            description=_build_description(target_map),
            confidence=round(float(base_confidence), 2),
            image_url=_image_for_payload(payload),
            source="ml-model",
        )
    )

    second_choice = {
        target_name: ranked[1][0] if len(ranked) > 1 else ranked[0][0]
        for target_name, ranked in probability_map.items()
    }
    second_confidence = sum(
        (ranked[1][1] if len(ranked) > 1 else ranked[0][1])
        for ranked in probability_map.values()
    ) / len(TARGET_COLUMNS)
    recommendations.append(
        RecommendationItem(
            title="Alternative outfit idea",
            description=_build_description(second_choice),
            confidence=round(float(second_confidence), 2),
            image_url=_image_for_payload(payload, "accessory"),
            source="ml-model",
        )
    )

    color_focus = target_map.copy()
    color_focus["color_recommendation"] = (
        probability_map["color_recommendation"][1][0]
        if len(probability_map["color_recommendation"]) > 1
        else probability_map["color_recommendation"][0][0]
    )
    color_confidence = sum(
        probability_map[target_name][0][1] for target_name in TARGET_COLUMNS
    ) / len(TARGET_COLUMNS)
    recommendations.append(
        RecommendationItem(
            title="Color-focused variation",
            description=_build_description(color_focus),
            confidence=round(float(color_confidence * 0.95), 2),
            image_url=_image_for_payload(payload, "palette"),
            source="ml-model",
        )
    )

    return [_complete_card(item, payload, index) for index, item in enumerate(recommendations)]


def recommend_outfits(payload: RecommendationRequest) -> RecommendationResponse:
    used_fallback = False

    try:
        pipeline, metadata = load_or_train_model()
        recommendations = _generate_ranked_recommendations(pipeline, payload)
    except Exception:
        metadata = {"model_version": "rule-based-only"}
        recommendations = [
            _complete_card(item, payload, index)
            for index, item in enumerate(build_rule_based_recommendations(payload))
        ]
        used_fallback = True

    if len(recommendations) < 3:
        recommendations.extend(
            _complete_card(item, payload, index)
            for index, item in enumerate(build_rule_based_recommendations(payload))
        )
        recommendations = recommendations[:3]

    style_strategy, _items, _palette, _explanation = _advanced_strategy(payload)
    return RecommendationResponse(
        recommendations=recommendations,
        model_version=metadata["model_version"],
        used_fallback=used_fallback,
        style_strategy=style_strategy,
    )
