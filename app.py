from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "logistic_regression_churn.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_churn.csv"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def load_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def risk_tier(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "High Risk", "Likely to churn"
    if probability >= 0.40:
        return "Medium Risk", "Needs retention attention"
    return "Low Risk", "Likely to stay"


def get_feature_importance(model) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "impact": pd.Series(coefficients).abs(),
        }
    ).sort_values("impact", ascending=False)

    return importance_df


st.set_page_config(
    page_title="RetentionScope",
    page_icon="RS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f4efe7;
            --paper: rgba(255, 250, 244, 0.82);
            --text: #1f2937;
            --muted: #6b7280;
            --line: rgba(148, 163, 184, 0.22);
            --accent: #d96c3f;
            --accent-dark: #b14d24;
            --accent-soft: rgba(217, 108, 63, 0.14);
            --green: #1f7a5c;
            --amber: #a16207;
            --red: #b42318;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(217, 108, 63, 0.18), transparent 26%),
                radial-gradient(circle at top right, rgba(31, 122, 92, 0.16), transparent 24%),
                linear-gradient(180deg, #f9f3ea 0%, var(--bg) 48%, #eee3d1 100%);
            color: var(--text);
            font-family: "Segoe UI", sans-serif;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            font-family: "Trebuchet MS", sans-serif;
            color: var(--text);
            letter-spacing: -0.03em;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .hero-shell {
            background:
                linear-gradient(135deg, rgba(255, 248, 238, 0.92), rgba(255, 255, 255, 0.68)),
                linear-gradient(120deg, rgba(217, 108, 63, 0.08), rgba(31, 122, 92, 0.06));
            border: 1px solid rgba(255, 255, 255, 0.6);
            box-shadow: 0 18px 60px rgba(88, 65, 38, 0.12);
            border-radius: 28px;
            padding: 2.4rem 2.2rem;
            margin-bottom: 1.4rem;
            backdrop-filter: blur(10px);
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-size: 0.84rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: clamp(2.6rem, 5vw, 4.6rem);
            line-height: 0.95;
            margin: 0;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.8;
            color: var(--muted);
            max-width: 720px;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }

        .hero-stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.4rem;
        }

        .hero-stat {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.66);
            border-radius: 18px;
            padding: 1rem 1.1rem;
        }

        .hero-stat-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
        }

        .hero-stat-value {
            margin-top: 0.35rem;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--text);
        }

        .section-card {
            background: var(--paper);
            border: 1px solid rgba(255, 255, 255, 0.7);
            box-shadow: 0 18px 55px rgba(111, 78, 45, 0.08);
            border-radius: 24px;
            padding: 1.35rem 1.2rem 1rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(8px);
        }

        .section-card h3 {
            font-size: 1.25rem;
            margin-bottom: 0.2rem;
        }

        .section-card p {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }

        .stSelectbox label, .stNumberInput label, .stSlider label {
            color: var(--text) !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stNumberInput > div > div {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid var(--line) !important;
            min-height: 3.15rem;
            border-radius: 16px !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45);
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] p,
        div[data-baseweb="select"] input,
        div[data-baseweb="input"] input,
        .stNumberInput input {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] [class*="singleValue"],
        div[data-baseweb="select"] [class*="placeholder"],
        div[data-baseweb="select"] [class*="valueContainer"],
        div[data-baseweb="select"] [class*="input-container"] {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] input::placeholder,
        div[data-baseweb="input"] input::placeholder,
        .stNumberInput input::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] svg,
        .stNumberInput svg {
            fill: var(--text) !important;
            color: var(--text) !important;
        }

        div[data-testid="stSlider"] [role="slider"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 6px rgba(217, 108, 63, 0.16);
        }

        .stButton > button {
            background: linear-gradient(135deg, #d96c3f, #b14d24) !important;
            color: white !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 0.9rem 1.5rem !important;
            min-height: 3.2rem;
            font-weight: 700 !important;
            font-size: 1rem !important;
            box-shadow: 0 16px 32px rgba(177, 77, 36, 0.24);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 20px 36px rgba(177, 77, 36, 0.28);
        }

        .result-shell {
            margin-top: 1.4rem;
            border-radius: 28px;
            padding: 1.6rem;
            border: 1px solid rgba(255, 255, 255, 0.7);
            box-shadow: 0 18px 55px rgba(111, 78, 45, 0.09);
            background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255,249,242,0.78));
        }

        .result-grid {
            display: grid;
            grid-template-columns: 1.4fr 1fr 1fr;
            gap: 1rem;
            margin-top: 1.1rem;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.86);
            border-radius: 20px;
            padding: 1.15rem;
            border: 1px solid rgba(255, 255, 255, 0.7);
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .result-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            line-height: 1;
            margin: 0;
        }

        .tier-high { color: var(--red); }
        .tier-medium { color: var(--amber); }
        .tier-low { color: var(--green); }

        .helper-copy {
            color: var(--muted);
            line-height: 1.7;
            font-size: 0.95rem;
        }

        .spacer-8 {
            height: 0.5rem;
        }

        @media (max-width: 900px) {
            .hero-stats,
            .result-grid {
                grid-template-columns: 1fr;
            }

            .hero-shell {
                padding: 1.6rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()
dataset = load_dataset()
importance_df = get_feature_importance(model)

st.markdown(
    """
    <section class="hero-shell">
        <div class="hero-kicker">RetentionScope | Customer Intelligence</div>
        <h1 class="hero-title">Predict churn before customers quietly leave.</h1>
        <p class="hero-subtitle">
            A portfolio-style retention dashboard that turns customer attributes into a clear churn-risk signal.
            Use it like an internal product page: enter customer details, review the risk level, and identify
            where retention teams should intervene first.
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-label">Prediction Engine</div>
                <div class="hero-stat-value">Logistic Regression</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Decision Focus</div>
                <div class="hero-stat-value">Retention Prioritization</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">Experience</div>
                <div class="hero-stat-value">Interactive Web Demo</div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

intro_left, intro_right = st.columns([1.3, 0.9], gap="large")
with intro_left:
    st.markdown(
        """
        <div class="helper-copy">
            Fill out the profile below to estimate churn probability and classify the customer into a practical
            risk tier. The layout is grouped like a real internal tool so it feels closer to a business dashboard
            than a default form.
        </div>
        """,
        unsafe_allow_html=True,
    )
with intro_right:
    st.info(
        "Best flow: complete the profile, run prediction, then use the probability and tier to suggest retention actions."
    )

tab_dashboard, tab_predictor, tab_model = st.tabs(
    ["Insights Dashboard", "Predict Customer", "Model Signals"]
)

with tab_dashboard:
    st.subheader("Churn Overview")
    if dataset is None:
        st.warning("Processed data not found. Run `python src/data_prep.py` to unlock the dashboard charts.")
    else:
        churn_rate = (dataset["Churn"] == "Yes").mean()
        avg_monthly = dataset["MonthlyCharges"].mean()
        avg_tenure = dataset["tenure"].mean()

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Churn Rate", f"{churn_rate:.1%}")
        metric_b.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
        metric_c.metric("Avg Tenure", f"{avg_tenure:.1f} months")

        chart_col1, chart_col2 = st.columns(2, gap="large")
        with chart_col1:
            st.caption("Churn by contract type")
            contract_view = pd.crosstab(dataset["Contract"], dataset["Churn"])
            st.bar_chart(contract_view)

        with chart_col2:
            st.caption("Average monthly charges by churn status")
            charges_view = (
                dataset.groupby("Churn", as_index=False)["MonthlyCharges"].mean().set_index("Churn")
            )
            st.bar_chart(charges_view)

        chart_col3, chart_col4 = st.columns(2, gap="large")
        with chart_col3:
            st.caption("Churn by internet service")
            internet_view = pd.crosstab(dataset["InternetService"], dataset["Churn"])
            st.bar_chart(internet_view)

        with chart_col4:
            st.caption("Average tenure by churn status")
            tenure_view = dataset.groupby("Churn", as_index=False)["tenure"].mean().set_index("Churn")
            st.bar_chart(tenure_view)

        st.caption("Snapshot of the processed churn dataset")
        st.dataframe(dataset.head(10), use_container_width=True)

with tab_predictor:
    profile_col, services_col, billing_col = st.columns(3, gap="large")

    with profile_col:
        st.markdown(
            """
            <div class="section-card">
                <h3>Customer Profile</h3>
                <p>Basic demographics and relationship context that often shape churn behavior.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with services_col:
        st.markdown(
            """
            <div class="section-card">
                <h3>Service Mix</h3>
                <p>Connectivity, support, and add-on services that influence overall customer stickiness.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with billing_col:
        st.markdown(
            """
            <div class="section-card">
                <h3>Billing and Plan</h3>
                <p>Commercial details that are strongly correlated with cancellation patterns and renewal risk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    metric_col1, metric_col2 = st.columns(2, gap="large")
    st.markdown('<div class="spacer-8"></div>', unsafe_allow_html=True)
    with metric_col1:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
    with metric_col2:
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=10.0)

    predict_now = st.button("Predict Churn Risk")

    if predict_now:
        input_df = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )

        probability = float(model.predict_proba(input_df)[0][1])
        tier, summary = risk_tier(probability)
        prediction = int(model.predict(input_df)[0])
        tier_class = (
            "tier-high" if tier == "High Risk" else "tier-medium" if tier == "Medium Risk" else "tier-low"
        )

        st.markdown(
            f"""
            <section class="result-shell">
                <h2>Prediction Result</h2>
                <p class="helper-copy">
                    This section translates the model output into a business-friendly readout that a support or retention team could act on.
                </p>
                <div class="result-grid">
                    <div class="result-card">
                        <div class="result-label">Risk Tier</div>
                        <p class="result-value {tier_class}">{tier}</p>
                        <p class="helper-copy">{summary}</p>
                    </div>
                    <div class="result-card">
                        <div class="result-label">Churn Probability</div>
                        <p class="result-value">{probability:.1%}</p>
                        <p class="helper-copy">Estimated probability that this customer may churn.</p>
                    </div>
                    <div class="result-card">
                        <div class="result-label">Model Decision</div>
                        <p class="result-value">{'Churn' if prediction == 1 else 'Stay'}</p>
                        <p class="helper-copy">Binary prediction from the trained classifier.</p>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

with tab_model:
    st.subheader("Feature Importance")
    st.caption("Top drivers from the logistic regression baseline, ranked by absolute coefficient magnitude.")

    top_features = importance_df.head(12).copy()
    top_features["signed impact"] = top_features["coefficient"]
    feature_chart = top_features.set_index("feature")[["signed impact"]]
    st.bar_chart(feature_chart)

    st.dataframe(
        top_features[["feature", "coefficient", "impact"]],
        use_container_width=True,
        hide_index=True,
    )
