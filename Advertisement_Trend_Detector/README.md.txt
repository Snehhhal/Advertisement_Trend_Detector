# Advertisement Trend Detector

## 1. Project Overview

The **Advertisement Trend Detector** is a machine learning based system designed to analyze advertisement campaign data and identify campaign performance patterns.

The system analyzes historical advertisement campaign features and predicts whether a campaign falls into the **High ROI** category.

The project includes data preprocessing, exploratory data analysis, machine learning, explainable AI, fairness analysis, API deployment and an interactive Streamlit dashboard.

---

## 2. Objectives

The main objectives of the project are:

* Analyze advertisement campaign performance.
* Identify advertisement trends from historical data.
* Predict High ROI campaigns using machine learning.
* Understand the factors influencing model predictions.
* Evaluate fairness across demographic groups.
* Provide an interactive dashboard for analysis and prediction.
* Monitor basic changes in the input data.
* Apply responsible AI principles.

---

## 3. Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Classifier

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Plotly

### Explainable AI

* SHAP
* LIME

### Responsible AI

* Fairness metrics
* Privacy considerations
* Consent considerations
* Human oversight

### Deployment

* FastAPI
* Uvicorn
* Docker
* Streamlit

### Documentation and Version Control

* Markdown
* Jupyter Notebook
* Git
* GitHub

---

## 4. Project Workflow

```text
Advertisement Dataset
        |
        v
Data Cleaning
        |
        v
Feature Engineering
        |
        v
Exploratory Data Analysis
        |
        v
Machine Learning Model
        |
        v
Model Evaluation
        |
        v
Explainable AI
        |
        v
Fairness Analysis
        |
        v
FastAPI / Docker Deployment
        |
        v
Streamlit Dashboard
        |
        v
Responsible AI Reporting
```

---

## 5. Dataset

The project uses advertisement campaign data containing numerical and categorical campaign features.

Important numerical features include:

* Impressions
* Clicks
* CTR
* Spend
* CPC
* CPM
* Conversions
* Conversion Rate
* ROI
* Likes
* Shares
* Comments
* Engagement Rate
* Total Engagement
* Cost Per Conversion

Categorical features include:

* Platform
* Advertisement Type
* Campaign Objective
* Target Region
* Target Age Group
* Target Gender
* Day of Week

---

## 6. Data Preprocessing

The preprocessing stage includes:

* Removing duplicate records.
* Converting date values.
* Handling missing numerical values.
* Creating date-related features.
* Creating engagement-related features.
* Creating cost-related features.
* Creating High ROI and High Engagement categories.
* Encoding categorical variables.
* Scaling numerical variables.

The processed dataset is stored in:

```text
data/cleaned_advertisement_dataset.csv
```

---

## 7. Machine Learning Model

The project uses a **Random Forest Classifier** for High ROI prediction.

The model uses a preprocessing pipeline containing:

* Numerical feature preprocessing.
* Categorical feature preprocessing.
* One-hot encoding.
* Random Forest classification.

The trained model is stored at:

```text
model/campaign_roi_model.pkl
```

The model uses an 80/20 train-test split.

---

## 8. Model Evaluation

The Streamlit dashboard provides model evaluation information including:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The dashboard also displays the model's prediction output for campaign inputs.

### Important Model Limitation

The current High ROI target is derived from the ROI value, while ROI is also included as an input feature.

This creates **target leakage**.

Therefore, the very high accuracy obtained by the current implementation should not be interpreted as evidence of equivalent real-world predictive performance.

A future version should remove ROI from the prediction inputs or redesign the target so that the model predicts future campaign performance without directly using the variable that defines the target.

---

## 9. Explainable AI

Explainable AI techniques are included to improve model transparency.

### SHAP

SHAP is used to identify the contribution of features to model predictions.

The dashboard provides:

* Top SHAP features.
* Mean absolute SHAP feature importance.
* SHAP summary plot.

### LIME

LIME was used during the explainable AI experiment to generate local explanations for individual predictions.

These techniques help users understand model behaviour.

---

## 10. Fairness Analysis

Fairness analysis was performed using demographic information such as target gender.

The project considers:

* Demographic Parity Difference.
* Equalized Odds Difference.
* Accuracy by demographic group.

These measurements are used to identify possible differences in model behaviour across demographic groups.

Fairness metrics are diagnostic measures and do not by themselves prove that a model is completely fair.

---

## 11. Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard contains:

### Overview

Provides a summary of the project and dataset.

### Advertisement Trends

Displays campaign-level trends and visualizations.

### High ROI Prediction

Allows users to enter campaign features and obtain a High ROI prediction.

### Model Metrics

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Explainable AI - SHAP

Displays SHAP feature importance and SHAP summary plots.

### Drift Checks

Provides a basic comparison between earlier and later portions of the dataset to identify changes in feature distributions.

### Responsible AI

Provides information about:

* Fairness
* Privacy
* Consent
* Explainability
* Human oversight
* Limitations
* Responsible AI checklist

---

## 12. Running the Streamlit Dashboard

Open PowerShell and navigate to the dashboard directory:

```powershell
cd "C:\Users\shriy\OneDrive\Pictures\Desktop\Advertisement_Trend_Detector\dashboard"
```

Run:

```powershell
py -m streamlit run app.py
```

The dashboard will normally open at:

```text
http://localhost:8501
```

---

## 13. Running the API

The project also contains a FastAPI deployment workflow.

The API provides a `/predict` endpoint that accepts campaign features and returns a prediction.

Example request structure:

```json
{
    "features": {
        "impressions": 10000,
        "clicks": 500,
        "ctr_percent": 5.0
    }
}
```

The complete API configuration and deployment workflow are documented in the project files.

---

## 14. Docker Deployment

The project includes a Docker-based deployment workflow.

The Docker workflow packages:

* FastAPI application.
* Trained machine learning model.
* Python dependencies.

The container can be built and executed using Docker commands according to the deployment configuration.

---

## 15. Project Structure

```text
Advertisement_Trend_Detector/
│
├── data/
│   └── cleaned_advertisement_dataset.csv
│
├── model/
│   └── campaign_roi_model.pkl
│
├── dashboard/
│   └── app.py
│
├── reports/
│   └── Responsible_AI.md
│
└── README.md
```

---

## 16. Responsible AI

Responsible AI practices are included in the project to address:

* Fairness
* Privacy
* Consent
* Explainability
* Transparency
* Human oversight
* Data drift
* Responsible use

A detailed report is available in:

```text
reports/Responsible_AI.md
```

---

## 17. Privacy and Consent

The system should avoid collecting unnecessary personal information.

If future versions use customer-level or personally identifiable information, appropriate permission and consent should be obtained before collecting and processing the information.

Only appropriate and permitted data should be included in a public repository.

---

## 18. Human Oversight

The system is intended as a **decision-support tool**.

Marketing professionals should review important decisions such as:

* Campaign budget allocation.
* Target audience selection.
* Campaign continuation.
* Advertisement strategy.

Model predictions should not be treated as guaranteed outcomes.

---

## 19. Limitations

The main limitations of the current implementation are:

1. Target leakage due to the use of ROI as both an input and a component of the High ROI target.
2. Dependence on the available historical dataset.
3. Possible changes in model behaviour when applied to new datasets.
4. Basic rather than production-level drift monitoring.
5. Fairness analysis depends on the demographic information available in the dataset.
6. Model predictions do not guarantee actual campaign success.

---

## 20. Future Scope

Future improvements may include:

* Removing target leakage.
* Predicting future campaign ROI without using current ROI.
* Adding more recent campaign data.
* Advanced data drift monitoring.
* Real-time campaign monitoring.
* Improved fairness evaluation.
* More detailed model explanations.
* Cloud deployment.
* Integration with advertising platforms.
* Advanced trend forecasting.
* Automated reporting.

---

## 21. Conclusion

The Advertisement Trend Detector combines data preprocessing, machine learning, explainable AI, fairness analysis and interactive visualization to analyze advertisement campaign performance.

The Streamlit dashboard provides a single interface for campaign trends, High ROI prediction, model metrics, SHAP explanations, drift checks and responsible AI information.

The project demonstrates how machine learning can be integrated into an advertisement analytics workflow while also considering transparency, privacy, fairness and human oversight.

---

