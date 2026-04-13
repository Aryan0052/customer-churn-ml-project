# Clothing Recommendation Website

This is a beginner-friendly but scalable outfit recommendation website built with FastAPI, scikit-learn, Pillow, HTML, CSS, and JavaScript.

## Folder Structure

```text
clothing_recommendation_app/
├── app/
│   ├── api/routes.py
│   ├── core/config.py
│   ├── ml/dataset.py
│   ├── ml/pipeline.py
│   ├── services/fallback.py
│   ├── services/feedback.py
│   ├── services/recommender.py
│   ├── static/css/styles.css
│   ├── static/js/app.js
│   ├── templates/index.html
│   ├── main.py
│   └── schemas.py
├── data/
├── models/
├── requirements.txt
└── train_model.py
```

## Features

- FastAPI backend with a `/recommend` REST API
- Random Forest based recommendation model
- Automatic preprocessing and categorical encoding with `OneHotEncoder`
- Synthetic sample dataset generator for training
- Rule-based fallback for safe recommendations
- Feedback logging endpoint for future retraining
- Simple frontend with recommendation cards and placeholder images
- Image upload mode that analyzes dominant colors, brightness, contrast, color temperature, and color harmony for styling tips
- Advanced profile controls for occasion, season, weather, budget, and color goal
- Explainable recommendation cards with tags, color palettes, and shopping-item hints

## Run Locally

1. Open a terminal in `clothing_recommendation_app`
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Start the server:

```bash
uvicorn app.main:app --reload
```

5. Open `http://127.0.0.1:8000`

If another app is already using port `8000`, run:

```bash
uvicorn app.main:app --reload --port 8010
```

## Example API Request

```json
{
  "height": "average",
  "skin_tone": "medium",
  "body_type": "athletic",
  "gender": "male",
  "style_preference": "streetwear",
  "occasion": "party",
  "season": "winter",
  "weather": "cold",
  "budget": "premium",
  "color_goal": "bold"
}
```

## Image Upload API

The frontend uses:

```text
POST /analyze-image
```

Send the image as multipart form data with the field name `file`. The response includes dominant colors, brightness, contrast, color temperature, color harmony, and image-based styling recommendations.

## Advanced Features Added

- Occasion-aware styling for everyday, office, party, wedding, date, college, travel, and gym.
- Weather-aware suggestions for hot, cold, rainy, and mild conditions.
- Season-aware styling for summer, winter, monsoon, and all-season outfits.
- Budget-aware guidance for budget, mid-range, and premium wardrobes.
- Color direction controls for subtle, balanced, bold, monochrome, and colorful looks.
- Explanation text so users understand why an outfit was recommended.
- Shopping hints such as blazers, footwear, accessories, layers, and bottoms.
- Image-based color harmony suggestions from uploaded pictures.

## Production Upgrade Ideas

- Use a real catalog and real user behavior data instead of synthetic data
- Save feedback in a database and schedule retraining jobs
- Replace the simple classifier with ranking models or deep learning embeddings
- Add clothing image understanding and product matching
- Add a wardrobe system where users upload their own clothes and the app builds outfits from owned items
- Use a vision model such as CLIP or a fashion item detector to identify shirts, pants, shoes, colors, and patterns
- Add vector search with FAISS or ChromaDB for similar outfit retrieval
- Add user accounts, saved profiles, saved outfits, and recommendation history
- Add product inventory with prices, stock status, filters, and shopping links
- Track model performance, drift, and feedback acceptance rate
- Containerize the app and deploy with CI/CD, logging, and monitoring
