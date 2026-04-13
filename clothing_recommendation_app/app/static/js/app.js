const form = document.getElementById("recommendation-form");
const resultsGrid = document.getElementById("results-grid");
const statusText = document.getElementById("status-text");
const imageForm = document.getElementById("image-form");
const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const imageResultsGrid = document.getElementById("image-results-grid");
const imageStatusText = document.getElementById("image-status-text");
const imageInsights = document.getElementById("image-insights");

function renderRecommendationCards(items, targetGrid, requestPayload = null) {
    targetGrid.innerHTML = "";

    items.forEach((item) => {
        const card = document.createElement("article");
        card.className = "recommendation-card";
        const tags = item.tags?.map((tag) => `<span>${tag}</span>`).join("") || "";
        const palette = item.color_palette?.map((color) => {
            if (color.startsWith("#")) {
                return `<span class="palette-dot" title="${color}" style="background: ${color}"></span>`;
            }
            return `<span class="palette-label">${color}</span>`;
        }).join("") || "";
        const shoppingItems = item.shopping_items?.map((entry) => `<li>${entry}</li>`).join("") || "";

        card.innerHTML = `
            <img src="${item.image_url}" alt="${item.title}" />
            <div class="card-content">
                <div class="card-meta">
                    <strong>${item.title}</strong>
                    <span>${item.source}</span>
                </div>
                <p>${item.description}</p>
                ${item.explanation ? `<p class="card-explanation">${item.explanation}</p>` : ""}
                ${tags ? `<div class="tag-row">${tags}</div>` : ""}
                ${palette ? `<div class="palette-row">${palette}</div>` : ""}
                ${shoppingItems ? `<ul class="shopping-list">${shoppingItems}</ul>` : ""}
                <span class="confidence-chip">Confidence: ${Math.round(item.confidence * 100)}%</span>
                <div class="feedback-actions">
                    <button type="button" data-title="${item.title}" data-liked="true">Helpful</button>
                    <button type="button" data-title="${item.title}" data-liked="false">Not for me</button>
                </div>
            </div>
        `;

        card.querySelectorAll("button").forEach((button) => {
            button.addEventListener("click", () => {
                if (!requestPayload) {
                    imageStatusText.textContent = "Thanks. Image styling feedback noted for future improvement.";
                    return;
                }
                submitFeedback(requestPayload, button.dataset.title, button.dataset.liked === "true");
            });
        });

        targetGrid.appendChild(card);
    });
}

function renderRecommendations(data, requestPayload) {
    renderRecommendationCards(data.recommendations, resultsGrid, requestPayload);
}

async function submitFeedback(requestPayload, recommendationTitle, liked) {
    await fetch("/feedback", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            request: requestPayload,
            recommendation_title: recommendationTitle,
            rating: liked ? 5 : 2,
            liked,
            notes: liked ? "User liked this suggestion." : "User wants a different look.",
        }),
    });

    statusText.textContent = liked
        ? "Thanks. Your positive feedback was saved."
        : "Thanks. Your feedback was saved to improve future recommendations.";
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());
    statusText.textContent = "Generating recommendations...";
    resultsGrid.innerHTML = "";

    try {
        const response = await fetch("/recommend", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        const data = await response.json();
        statusText.textContent = `${data.style_strategy} | Model: ${data.model_version}${data.used_fallback ? " | rule-based fallback used" : ""}`;
        renderRecommendations(data, payload);
    } catch (error) {
        statusText.textContent = "Something went wrong while generating recommendations.";
    }
});

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) {
        imagePreview.style.display = "none";
        return;
    }

    imagePreview.src = URL.createObjectURL(file);
    imagePreview.style.display = "block";
    imageStatusText.textContent = `Selected: ${file.name}`;
});

imageForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files[0];
    if (!file) {
        imageStatusText.textContent = "Please choose an image first.";
        return;
    }

    const payload = new FormData();
    payload.append("file", file);
    imageStatusText.textContent = "Analyzing picture...";
    imageResultsGrid.innerHTML = "";
    imageInsights.innerHTML = "";

    try {
        const response = await fetch("/analyze-image", {
            method: "POST",
            body: payload,
        });

        const data = await response.json();
        imageStatusText.textContent = `Brightness: ${data.brightness} | Contrast: ${data.contrast} | Harmony: ${data.color_harmony}`;
        imageInsights.innerHTML = data.dominant_colors
            .map((color) => `<span class="insight-pill" style="border-left: 18px solid ${color}">${color}</span>`)
            .join("");
        renderRecommendationCards(data.recommendations, imageResultsGrid);
    } catch (error) {
        imageStatusText.textContent = "Something went wrong while analyzing the image.";
    }
});
