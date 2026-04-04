# Customer Churn Prediction

This project combines data analysis and machine learning to understand why customers leave a company and to predict which customers are likely to churn.

## Project Goal

- Analyze customer behavior and business patterns related to churn.
- Build a machine learning model to predict whether a customer will churn.
- Present findings clearly with charts, metrics, and business recommendations.

## Recommended Dataset

Use the `Telco Customer Churn` dataset from Kaggle:

[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

After downloading, place the CSV file here:

`data/raw/telco_customer_churn.csv`

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- outputs/
|   |-- metrics/
|   |-- figures/
|   `-- models/
|-- src/
|   |-- data_prep.py
|   |-- eda.py
|   |-- model_comparison.py
|   `-- train_model.py
|-- app.py
|-- report_template.md
|-- requirements.txt
`-- README.md
```

## Workflow

1. Load and inspect the raw dataset.
2. Clean missing values and fix data types.
3. Perform EDA:
   - churn distribution
   - contract type vs churn
   - monthly charges vs churn
   - tenure vs churn
   - payment method vs churn
4. Encode categorical columns.
5. Split the dataset into train and test sets.
6. Train a baseline model.
7. Evaluate the model using:
   - accuracy
   - precision
   - recall
   - F1-score
   - ROC-AUC
8. Interpret important features.
9. Write conclusions and business suggestions.

## Portfolio Flow

Use the project in this order:

1. Download the dataset and place it in `data/raw/`.
2. Run `src/data_prep.py` to create the cleaned dataset.
3. Run `src/eda.py` and review the notebook in `notebooks/`.
4. Run `src/train_model.py` for a baseline model.
5. Run `src/model_comparison.py` to compare models.
6. Launch `app.py` with Streamlit for a demo.
7. Fill in `report_template.md` with your results and insights.

## Good Models To Try

- Logistic Regression
- Random Forest
- XGBoost or LightGBM later

Start with Logistic Regression because it is easy to explain and works well as a baseline.

## How To Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run data preparation:

```powershell
python src/data_prep.py
```

Run EDA and save charts:

```powershell
python src/eda.py
```

Train the model:

```powershell
python src/train_model.py
```

Compare multiple models:

```powershell
python src/model_comparison.py
```

Launch the Streamlit app:

```powershell
streamlit run app.py
```

## Suggested Portfolio Title

`Customer Churn Prediction Using Data Analysis and Machine Learning`

## Questions This Project Answers

- Which customers are most likely to leave?
- Which services or plans are linked with high churn?
- Does contract type reduce churn?
- How important are monthly charges and tenure?
- What actions can a business take to retain customers?

## Next Steps

- Add a Jupyter notebook for EDA.
- Build a dashboard in Power BI or Tableau.
- Improve the model with feature engineering and tuning.
- Deploy the model with Streamlit later if you want.
