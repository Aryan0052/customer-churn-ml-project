from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd


HEIGHTS = ["short", "average", "tall"]
SKIN_TONES = ["fair", "medium", "dark"]
BODY_TYPES = ["slim", "athletic", "heavy"]
GENDERS = ["male", "female", "unisex"]
STYLES = ["casual", "formal", "streetwear"]


def _choose_fit(height: str, body_type: str, style: str) -> str:
    if style == "formal":
        return "tailored blazer and straight-fit trousers" if body_type != "heavy" else "structured blazer and relaxed trousers"
    if style == "streetwear":
        return "oversized layers with relaxed-fit pants" if height != "short" else "cropped jacket with tapered joggers"
    if body_type == "slim":
        return "slim fit jeans with a clean silhouette"
    if body_type == "athletic":
        return "athletic taper chinos with fitted basics"
    return "relaxed fit jeans with a balanced drape"


def _choose_top(skin_tone: str, style: str, gender: str) -> str:
    if style == "formal":
        return "button-down shirt with a lightweight blazer"
    if style == "streetwear":
        return "oversized t-shirt with a statement jacket"
    if skin_tone == "fair":
        return "earth-tone polo or knit top"
    if skin_tone == "medium":
        return "rich jewel-tone shirt or tee"
    return "crisp pastel tee or textured overshirt"


def _choose_palette(skin_tone: str, style: str) -> str:
    if style == "formal":
        return "navy, charcoal, and white"
    if style == "streetwear":
        return "black, olive, and muted accent colors"
    if skin_tone == "fair":
        return "olive, rust, and soft blue"
    if skin_tone == "medium":
        return "emerald, burgundy, and cream"
    return "mustard, lavender, and sky blue"


def _choose_extra(height: str, body_type: str, gender: str) -> str:
    if height == "short":
        return "low-contrast sneakers and shorter outerwear"
    if height == "tall":
        return "layered outerwear and wide-leg options"
    if body_type == "heavy":
        return "vertical patterns and open layers"
    if gender == "female":
        return "structured accessories and clean footwear"
    return "minimal accessories and classic sneakers"


def build_sample_dataset(output_path: Path) -> pd.DataFrame:
    rows = []

    for height, skin_tone, body_type, gender, style in product(
        HEIGHTS, SKIN_TONES, BODY_TYPES, GENDERS, STYLES
    ):
        base_row = {
            "height": height,
            "skin_tone": skin_tone,
            "body_type": body_type,
            "gender": gender,
            "style_preference": style,
            "fit_recommendation": _choose_fit(height, body_type, style),
            "top_recommendation": _choose_top(skin_tone, style, gender),
            "color_recommendation": _choose_palette(skin_tone, style),
            "extra_recommendation": _choose_extra(height, body_type, gender),
        }
        rows.append(base_row)

        variant = base_row.copy()
        if style == "casual":
            variant["extra_recommendation"] = "clean sneakers and simple layering pieces"
        elif style == "formal":
            variant["fit_recommendation"] = "smart tailored separates with neat proportions"
        else:
            variant["top_recommendation"] = "graphic tee with a roomy overshirt"
        rows.append(variant)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
