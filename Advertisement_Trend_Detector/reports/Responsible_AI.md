# Responsible AI Report

## 1. Introduction

The Advertisement Trend Detector is a machine learning based system that analyzes advertisement campaign data and predicts whether a campaign falls into the **High ROI** category. The system uses campaign features such as impressions, clicks, CTR, spending, conversions, engagement and demographic information.

Responsible AI principles are considered to make the system transparent, fair, privacy-aware and suitable for human-supported decision making.

---

## 2. Objective

The main objectives of responsible AI implementation in this project are:

* To provide understandable model predictions.
* To examine possible fairness issues.
* To protect user and campaign-related information.
* To consider consent and responsible data usage.
* To provide transparency about model limitations.
* To keep humans involved in important marketing decisions.
* To monitor changes in the input data over time.

---

## 3. Model Used

The system uses a **Random Forest Classifier** to predict the `High_ROI` target.

The model is trained using advertisement campaign data after preprocessing numerical and categorical features.

### Numerical features include:

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
* Date-related features

### Categorical features include:

* Platform
* Advertisement Type
* Campaign Objective
* Target Region
* Target Age Group
* Target Gender
* Day of Week

---

## 4. Explainable AI

Explainable AI techniques are used to understand the behaviour of the machine learning model.

### SHAP

SHAP (SHapley Additive exPlanations) is used to identify the contribution of individual features to model predictions.

The Streamlit dashboard provides:

* SHAP feature importance
* Top 15 important features
* SHAP summary plot

Features with larger absolute SHAP values have a greater influence on the model's predictions.

SHAP explanations describe model behaviour. They should not be interpreted as proof that a feature directly causes a particular advertising outcome.

### LIME

LIME was also used during the explainable AI experiment to provide local explanations for individual predictions.

LIME explains a prediction by examining how changes in input features can affect the model's output around a particular example.

---

## 5. Fairness

Fairness is important because advertisement campaign data may contain demographic information such as target gender and target age group.

During the explainable AI experiment, fairness was examined using:

* Demographic Parity Difference
* Equalized Odds Difference
* Accuracy by demographic group

These measures help identify whether model performance or prediction behaviour differs between demographic groups.

Fairness measurements should be treated as diagnostic information rather than proof that the model is completely fair.

The model should be periodically evaluated when new data is introduced.

---

## 6. Privacy

The system should avoid using unnecessary personally identifiable information.

The advertisement dataset used for this project contains campaign-level information rather than requiring individual users' personal information.

Privacy practices include:

* Avoid collecting unnecessary personal information.
* Do not expose private customer information in the dashboard.
* Do not publish confidential campaign information.
* Store datasets and model files securely.
* Restrict access to sensitive information.
* Remove personally identifiable information before using data for analysis whenever possible.

The public project repository should contain only data that is appropriate for public sharing.

---

## 7. Consent

Consent is important when data is collected from users or individuals.

If future versions of the Advertisement Trend Detector use customer-level information, social media user information, survey responses or other personally identifiable information, appropriate consent and data-use permissions should be obtained before collecting and processing the information.

For this project, the system is focused on structured advertisement campaign data.

The project should not assume permission to collect or publish personal data simply because the data is technically accessible.

---

## 8. Transparency

The system provides information about:

* The machine learning model used.
* The input features used for prediction.
* Model evaluation metrics.
* SHAP-based explanations.
* Fairness considerations.
* Data drift checks.
* Model limitations.

The Streamlit dashboard is designed to make the model workflow easier to understand.

Users should be informed that a model prediction is an analytical output and not a guaranteed result.

---

## 9. Human Oversight

The model should support human decision making rather than completely replace marketing professionals.

A marketer or analyst should review important decisions such as:

* Campaign budget allocation
* Advertisement strategy
* Target audience selection
* Campaign continuation or cancellation
* Major changes to advertising strategy

Model predictions should be considered together with business objectives, campaign context and other relevant information.

---

## 10. Data Drift

The Streamlit dashboard includes a basic drift check that compares an earlier portion of the dataset with a later portion.

Drift monitoring can help identify changes in the distribution of campaign features over time.

Examples include changes in:

* Impressions
* Clicks
* Spending
* CTR
* Conversions
* Engagement
* Other campaign characteristics

A change in data distribution does not automatically mean that the model is incorrect. It indicates that the data should be investigated further.

The current dashboard provides a basic drift check and is not a formal statistical drift-monitoring system.

---

## 11. Model Limitations

The model has several limitations.

### 11.1 Target Leakage

The current `High_ROI` target is derived from the `roi` feature, while ROI is also included among the model input features.

This creates **target leakage** because information directly related to the target is available to the model during prediction.

Therefore, the very high test accuracy obtained in the current implementation should not be interpreted as proof of real-world predictive performance.

A future improved version should remove ROI or redesign the target so that the model predicts future campaign performance without directly receiving the target-defining information.

### 11.2 Dataset Dependency

The model learns patterns from the available advertisement dataset. Its behaviour may change when it is applied to campaigns from different platforms, regions, industries or time periods.

### 11.3 Prediction Uncertainty

A model prediction is not a guarantee of campaign success. Real-world advertising performance can be affected by factors that are not included in the dataset.

### 11.4 Fairness Limitations

Fairness metrics depend on the available demographic information and the selected evaluation methods. Additional fairness analysis may be required before using the system in a real-world environment.

### 11.5 Drift Monitoring Limitations

The current drift check provides a basic comparison of data distributions. More advanced monitoring techniques may be required for production deployment.

---

## 12. Responsible Use

The Advertisement Trend Detector should be used as a decision-support system.

Users should:

* Review model predictions before making important decisions.
* Consider the limitations of the dataset.
* Avoid treating predictions as guaranteed outcomes.
* Monitor model performance when new data becomes available.
* Check fairness across relevant groups.
* Protect sensitive information.
* Obtain appropriate consent when personal data is used.
* Keep humans involved in high-impact decisions.

---

## 13. Responsible AI Checklist

| Area            | Implementation                                                |
| --------------- | ------------------------------------------------------------- |
| Fairness        | Demographic fairness metrics and group-wise evaluation        |
| Privacy         | Avoid unnecessary personal information                        |
| Consent         | Obtain appropriate permission when personal data is collected |
| Explainability  | SHAP and LIME                                                 |
| Transparency    | Model, features, metrics and limitations documented           |
| Human Oversight | Human review of important marketing decisions                 |
| Data Drift      | Basic comparison of earlier and later data                    |
| Security        | Avoid exposing confidential campaign information              |
| Limitations     | Target leakage and dataset limitations documented             |

---

## 14. Conclusion

Responsible AI practices help make the Advertisement Trend Detector more transparent and suitable for responsible decision support. The project includes explainability using SHAP and LIME, fairness evaluation, privacy considerations, consent requirements, human oversight and basic data drift monitoring.

The system should be used as an analytical support tool rather than as an automatic decision maker. The identified target leakage is an important limitation and should be addressed in a future version before relying on the model for real-world prediction.
