from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageStat

from app.schemas import ImageAnalysisResponse, RecommendationItem
from app.services.fallback import OUTFIT_IMAGES


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _color_temperature(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    if red > blue + 20 and green > blue:
        return "warm"
    if blue > red + 20:
        return "cool"
    return "neutral"


def _color_harmony(temperature: str, brightness: str, contrast: str) -> tuple[str, list[str]]:
    if contrast == "high":
        return "high-contrast neutral", ["black", "white", "charcoal", "silver"]
    if brightness == "dark":
        return "lifted contrast", ["cream", "light denim", "tan", "soft white"]
    if temperature == "warm":
        return "warm complementary", ["olive", "cream", "denim blue", "chocolate"]
    if temperature == "cool":
        return "cool analogous", ["navy", "ice blue", "grey", "white"]
    return "balanced neutral", ["navy", "beige", "forest green", "off-white"]


def _brightness_label(value: float) -> str:
    if value < 85:
        return "dark"
    if value > 170:
        return "bright"
    return "balanced"


def _contrast_label(stddev: float) -> str:
    if stddev < 35:
        return "soft"
    if stddev > 70:
        return "high"
    return "medium"


def analyze_image_for_styling(image_bytes: bytes) -> ImageAnalysisResponse:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    preview = image.resize((160, 160))
    palette_preview = preview.quantize(colors=5).convert("RGB")

    stat = ImageStat.Stat(preview.convert("L"))
    brightness = _brightness_label(stat.mean[0])
    contrast = _contrast_label(stat.stddev[0])

    palette_image = preview.resize((1, 1), resample=Image.Resampling.BILINEAR)
    average_rgb = palette_image.getpixel((0, 0))
    color_counts = palette_preview.getcolors(160 * 160) or []
    color_counts = sorted(color_counts, key=lambda item: item[0], reverse=True)
    dominant_colors = [_rgb_to_hex(color) for _count, color in color_counts[:5]]
    temperature = _color_temperature(average_rgb)
    harmony, recommended_palette = _color_harmony(temperature, brightness, contrast)

    if temperature == "warm":
        color_tip = "Style this with olive, cream, denim blue, chocolate, or soft black."
    elif temperature == "cool":
        color_tip = "Style this with charcoal, white, navy, icy blue, or silver-toned accents."
    else:
        color_tip = "Style this with navy, beige, off-white, forest green, or muted earth tones."

    if brightness == "dark":
        balance_tip = "Add one lighter layer or sneaker to keep the outfit from feeling too heavy."
    elif brightness == "bright":
        balance_tip = "Ground the look with darker bottoms or a textured jacket."
    else:
        balance_tip = "Keep the outfit balanced with one statement piece and simple basics."

    if contrast == "high":
        texture_tip = "High contrast works well with clean silhouettes and minimal accessories."
    elif contrast == "soft":
        texture_tip = "Soft contrast looks polished with layered textures like knit, linen, or suede."
    else:
        texture_tip = "Medium contrast is flexible, so try a tonal outfit with one accent color."

    recommendations = [
        RecommendationItem(
            title="Color palette suggestion",
            description=color_tip,
            confidence=0.78,
            image_url=OUTFIT_IMAGES["palette"],
            source="image-analysis",
            explanation=f"The image appears {temperature} with {brightness} brightness, so this palette keeps the styling intentional.",
            tags=["image-color", temperature, harmony],
            shopping_items=["solid top", "neutral bottom", "matching footwear"],
            color_palette=recommended_palette,
        ),
        RecommendationItem(
            title="Outfit balance suggestion",
            description=balance_tip,
            confidence=0.72,
            image_url=OUTFIT_IMAGES["casual"],
            source="image-analysis",
            explanation=f"The image brightness is {brightness}; this tip balances visual weight in the outfit.",
            tags=["balance", brightness],
            shopping_items=["outer layer", "bottom wear", "sneakers"],
            color_palette=recommended_palette[:3],
        ),
        RecommendationItem(
            title="Texture and accessory suggestion",
            description=texture_tip,
            confidence=0.7,
            image_url=OUTFIT_IMAGES["accessory"],
            source="image-analysis",
            explanation=f"The image has {contrast} contrast, so texture and accessory choices should match that energy.",
            tags=["texture", contrast],
            shopping_items=["watch or bracelet", "textured layer", "simple bag"],
            color_palette=recommended_palette,
        ),
    ]

    return ImageAnalysisResponse(
        dominant_colors=dominant_colors,
        brightness=brightness,
        contrast=contrast,
        color_temperature=temperature,
        color_harmony=harmony,
        recommendations=recommendations,
    )
