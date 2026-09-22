import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# ============================================================
# OPTIONAL SHAP
# ============================================================

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advertisement Trend Detector",
    page_icon="📊",
    layout="wide"
)

st.title("Advertisement Trend Detector")
st.caption(
    "Advertisement campaign analysis, High ROI prediction, "
    "Explainable AI and Responsible AI dashboard"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_advertisement_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "campaign_roi_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
except Exception as e:
    st.error("Unable to load the dataset.")
    st.exception(e)
    st.stop()


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Advertisement Trends",
        "High ROI Prediction",
        "Model Metrics",
        "Explainable AI - SHAP",
        "Drift Checks",
        "Responsible AI"
    ]
)


# ============================================================
# COMMON SETTINGS
# ============================================================

TARGET = "High_ROI"

EXCLUDED_COLUMNS = [
    "High_ROI",
    "High_Engagement",
    "campaign_id",
    "date"
]

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


# ============================================================
# 1. OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Advertisement Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Campaigns",
            len(df)
        )

    with col2:
        st.metric(
            "Platforms",
            df["platform"].nunique()
        )

    with col3:
        high_roi_count = int(
            pd.to_numeric(
                df["High_ROI"],
                errors="coerce"
            ).sum()
        )

        st.metric(
            "High ROI Campaigns",
            high_roi_count
        )

    with col4:

        roi_numeric = pd.to_numeric(
            df["roi"],
            errors="coerce"
        )

        avg_roi = roi_numeric.mean()

        st.metric(
            "Average ROI",
            f"{avg_roi:.2f}"
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        width="stretch"
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"Number of rows: **{df.shape[0]}**"
        )

    with col2:
        st.write(
            f"Number of columns: **{df.shape[1]}**"
        )


# ============================================================
# 2. ADVERTISEMENT TRENDS
# ============================================================

elif page == "Advertisement Trends":

    st.header("Advertisement Trends")

    # --------------------------------------------------------
    # Campaigns by Platform
    # --------------------------------------------------------

    st.subheader("Campaigns by Platform")

    platform_counts = (
        df["platform"]
        .value_counts()
        .reset_index()
    )

    platform_counts.columns = [
        "Platform",
        "Campaigns"
    ]

    fig1 = px.bar(
        platform_counts,
        x="Platform",
        y="Campaigns",
        title="Number of Campaigns by Platform"
    )

    st.plotly_chart(
        fig1,
        width="stretch"
    )

    # --------------------------------------------------------
    # High ROI Distribution
    # --------------------------------------------------------

    st.subheader("High ROI Campaign Distribution")

    roi_counts = (
        df["High_ROI"]
        .value_counts()
        .reset_index()
    )

    roi_counts.columns = [
        "High_ROI",
        "Count"
    ]

    roi_counts["ROI Category"] = roi_counts[
        "High_ROI"
    ].map(
        {
            0: "Not High ROI",
            1: "High ROI"
        }
    )

    fig2 = px.pie(
        roi_counts,
        names="ROI Category",
        values="Count",
        title="High ROI vs Not High ROI"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

    # --------------------------------------------------------
    # Average ROI by Platform
    # --------------------------------------------------------

    st.subheader("Average ROI by Platform")

    temp_df = df.copy()

    temp_df["roi"] = pd.to_numeric(
        temp_df["roi"],
        errors="coerce"
    )

    avg_roi_platform = (
        temp_df
        .groupby("platform")["roi"]
        .mean()
        .reset_index()
    )

    fig3 = px.bar(
        avg_roi_platform,
        x="platform",
        y="roi",
        title="Average ROI by Platform",
        labels={
            "platform": "Platform",
            "roi": "Average ROI"
        }
    )

    st.plotly_chart(
        fig3,
        width="stretch"
    )

    # --------------------------------------------------------
    # Engagement Rate by Platform
    # --------------------------------------------------------

    st.subheader(
        "Average Engagement Rate by Platform"
    )

    temp_df[
        "engagement_rate_percent"
    ] = pd.to_numeric(
        temp_df["engagement_rate_percent"],
        errors="coerce"
    )

    avg_engagement = (
        temp_df
        .groupby("platform")[
            "engagement_rate_percent"
        ]
        .mean()
        .reset_index()
    )

    fig4 = px.bar(
        avg_engagement,
        x="platform",
        y="engagement_rate_percent",
        title="Average Engagement Rate by Platform",
        labels={
            "platform": "Platform",
            "engagement_rate_percent":
                "Engagement Rate (%)"
        }
    )

    st.plotly_chart(
        fig4,
        width="stretch"
    )


# ============================================================
# 3. HIGH ROI PREDICTION
# ============================================================

elif page == "High ROI Prediction":

    st.header("High ROI Prediction")

    st.write(
        "Enter advertisement campaign details and the trained "
        "Random Forest model will predict whether the campaign "
        "belongs to the High ROI category."
    )

    input_data = {}

    st.subheader("Campaign Details")

    for column in FEATURE_COLUMNS:

        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        numeric_ratio = (
            numeric_series.notna().mean()
        )

        # ----------------------------------------------------
        # NUMERICAL FEATURE
        # ----------------------------------------------------

        if numeric_ratio >= 0.80:

            median_value = numeric_series.median()

            if pd.isna(median_value):
                median_value = 0.0

            min_value = numeric_series.min()
            max_value = numeric_series.max()

            if pd.isna(min_value):
                min_value = 0.0

            if pd.isna(max_value):
                max_value = 100.0

            if min_value == max_value:
                max_value = min_value + 1

            input_data[column] = st.number_input(
                column,
                value=float(median_value),
                min_value=float(min_value),
                max_value=float(max_value)
            )

        # ----------------------------------------------------
        # CATEGORICAL FEATURE
        # ----------------------------------------------------

        else:

            categories = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            categories = sorted(categories)

            if len(categories) == 0:
                categories = ["Unknown"]

            input_data[column] = st.selectbox(
                column,
                categories
            )

    st.divider()

    if st.button(
        "Predict High ROI",
        type="primary"
    ):

        input_df = pd.DataFrame(
            [input_data]
        )

        try:

            prediction = model.predict(
                input_df
            )[0]

            probabilities = model.predict_proba(
                input_df
            )[0]

            confidence = (
                max(probabilities) * 100
            )

            st.subheader(
                "Prediction Result"
            )

            if prediction == 1:

                st.success(
                    "Prediction: High ROI"
                )

            else:

                st.warning(
                    "Prediction: Not High ROI"
                )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.info(
                "This prediction is decision-support information "
                "and should not be treated as a guaranteed outcome."
            )

        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(e)


# ============================================================
# 4. MODEL METRICS
# ============================================================

elif page == "Model Metrics":

    st.header("Model Performance Metrics")

    st.write(
        "The deployed model is a Random Forest Classifier "
        "trained to predict High_ROI."
    )

    # --------------------------------------------------------
    # Recreate the same train-test split
    # --------------------------------------------------------

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    X = df.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore"
    )

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    try:

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Precision",
                f"{precision * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Recall",
                f"{recall * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{f1 * 100:.2f}%"
            )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Not High ROI",
                "Actual High ROI"
            ],
            columns=[
                "Predicted Not High ROI",
                "Predicted High ROI"
            ]
        )

        st.dataframe(
            cm_df,
            width="stretch"
        )

        st.subheader(
            "Model Configuration"
        )

        st.write(
            "Algorithm: Random Forest Classifier"
        )

        st.write(
            "Number of estimators: 100"
        )

        st.write(
            "Random state: 42"
        )

        st.write(
            "Train-test split: 80% / 20%"
        )

        st.warning(
            "The current Experiment 6 model includes ROI-related "
            "information while High_ROI is derived from ROI. "
            "This can cause target leakage and can make the "
            "reported performance unrealistically high. "
            "Therefore, the accuracy should not be interpreted "
            "as proof of real-world performance."
        )

    except Exception as e:

        st.error(
            "Unable to calculate model metrics."
        )

        st.exception(e)


# ============================================================
# 5. EXPLAINABLE AI - SHAP
# ============================================================

elif page == "Explainable AI - SHAP":

    st.header("Explainable AI - SHAP")

    st.write(
        "SHAP (SHapley Additive exPlanations) is used to "
        "understand which features influence the model prediction."
    )

    if not SHAP_AVAILABLE:

        st.error("SHAP is not installed.")

        st.code("py -m pip install shap")

    else:

        try:

            # ------------------------------------------------
            # Check that the model is a pipeline
            # ------------------------------------------------

            if not hasattr(model, "named_steps"):

                st.error(
                    "The loaded model is not a scikit-learn Pipeline."
                )

                st.stop()

            # ------------------------------------------------
            # Find preprocessing and Random Forest automatically
            # ------------------------------------------------

            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import RandomForestClassifier

            preprocessing = None
            rf_model = None

            st.write("Model pipeline steps:")

            for step_name, step_model in model.named_steps.items():

                st.write(
                    f"- {step_name}: "
                    f"{type(step_model).__name__}"
                )

                # Find ColumnTransformer
                if isinstance(
                    step_model,
                    ColumnTransformer
                ):

                    preprocessing = step_model

                # Find Random Forest
                if isinstance(
                    step_model,
                    RandomForestClassifier
                ):

                    rf_model = step_model

            # ------------------------------------------------
            # Check preprocessing
            # ------------------------------------------------

            if preprocessing is None:

                st.error(
                    "ColumnTransformer preprocessing step "
                    "could not be found in the model."
                )

                st.write(
                    "Available pipeline steps:"
                )

                st.write(
                    list(model.named_steps.keys())
                )

                st.stop()

            # ------------------------------------------------
            # Check Random Forest
            # ------------------------------------------------

            if rf_model is None:

                st.error(
                    "Random Forest classifier could not "
                    "be found in the model."
                )

                st.stop()

            st.success(
                "Preprocessing and Random Forest model "
                "identified successfully."
            )

            # ------------------------------------------------
            # Prepare input data
            # ------------------------------------------------

            X = df.drop(
                columns=EXCLUDED_COLUMNS,
                errors="ignore"
            )

            # Use a smaller sample for faster SHAP calculation
            sample_size = min(
                300,
                len(X)
            )

            X_sample = X.sample(
                sample_size,
                random_state=42
            )

            # ------------------------------------------------
            # Transform the data
            # ------------------------------------------------

            X_transformed = preprocessing.transform(
                X_sample
            )

            # Convert sparse matrix to dense matrix
            if hasattr(
                X_transformed,
                "toarray"
            ):

                X_transformed = (
                    X_transformed.toarray()
                )

            # ------------------------------------------------
            # Get transformed feature names
            # ------------------------------------------------

            try:

                feature_names = (
                    preprocessing
                    .get_feature_names_out()
                )

            except Exception:

                feature_names = [
                    f"Feature_{i}"
                    for i in range(
                        X_transformed.shape[1]
                    )
                ]

            # ------------------------------------------------
            # Create SHAP explainer
            # ------------------------------------------------

            explainer = shap.TreeExplainer(
                rf_model
            )

            shap_values = explainer.shap_values(
                X_transformed
            )

            # ------------------------------------------------
            # Handle different SHAP output formats
            # ------------------------------------------------

            if isinstance(
                shap_values,
                list
            ):

                # Binary classification
                if len(shap_values) > 1:

                    values = shap_values[1]

                else:

                    values = shap_values[0]

            else:

                shap_array = np.asarray(
                    shap_values
                )

                # Newer SHAP versions may return:
                # samples × features × classes

                if shap_array.ndim == 3:

                    values = shap_array[:, :, 1]

                else:

                    values = shap_array

            # ------------------------------------------------
            # Make sure dimensions match
            # ------------------------------------------------

            if values.shape[1] != len(
                feature_names
            ):

                feature_names = [
                    f"Feature_{i}"
                    for i in range(
                        values.shape[1]
                    )
                ]

            # ------------------------------------------------
            # SHAP Feature Importance
            # ------------------------------------------------

            st.subheader(
                "SHAP Feature Importance"
            )

            mean_abs_shap = np.mean(
                np.abs(values),
                axis=0
            )

            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Mean |SHAP|": mean_abs_shap
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    "Mean |SHAP|",
                    ascending=False
                )
                .head(15)
            )

            fig_shap = px.bar(
                importance_df.sort_values(
                    "Mean |SHAP|"
                ),
                x="Mean |SHAP|",
                y="Feature",
                orientation="h",
                title="Top 15 SHAP Features"
            )

            st.plotly_chart(
                fig_shap,
                width="stretch"
            )

            # ------------------------------------------------
            # SHAP Summary Plot
            # ------------------------------------------------

            st.subheader(
                "SHAP Summary Plot"
            )

            import matplotlib.pyplot as plt

            plt.figure(
                figsize=(10, 7)
            )

            shap.summary_plot(
                values,
                X_transformed,
                feature_names=feature_names,
                show=False
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True
            )

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            st.success(
                "SHAP analysis completed successfully."
            )

            st.info(
                "Features with larger mean absolute SHAP "
                "values have a greater influence on the "
                "model's predictions."
            )

            st.caption(
                "SHAP values explain model behaviour. "
                "They show feature contribution to predictions "
                "and should not be interpreted as proof of "
                "causal relationships."
            )

        except Exception as e:

            st.error(
                "SHAP analysis could not be generated."
            )

            st.exception(e)


# ============================================================
# 6. DRIFT CHECKS
# ============================================================

elif page == "Drift Checks":

    st.header("Data Drift Checks")

    st.write(
        "Drift checks compare earlier and later portions of "
        "the available dataset to identify noticeable changes "
        "in feature distributions."
    )

    drift_df = df.copy()

    # --------------------------------------------------------
    # Date-based split when date is available
    # --------------------------------------------------------

    if "date" in drift_df.columns:

        drift_df["date"] = pd.to_datetime(
            drift_df["date"],
            errors="coerce"
        )

        drift_df = drift_df.dropna(
            subset=["date"]
        ).sort_values(
            "date"
        )

    split_index = int(
        len(drift_df) * 0.80
    )

    reference_data = drift_df.iloc[
        :split_index
    ]

    current_data = drift_df.iloc[
        split_index:
    ]

    st.write(
        f"Reference records: **{len(reference_data)}**"
    )

    st.write(
        f"Current records: **{len(current_data)}**"
    )

    st.subheader(
        "Numerical Feature Drift"
    )

    numeric_columns = []

    for column in FEATURE_COLUMNS:

        converted = pd.to_numeric(
            drift_df[column],
            errors="coerce"
        )

        if converted.notna().mean() >= 0.80:
            numeric_columns.append(
                column
            )

    drift_results = []

    for column in numeric_columns:

        reference_values = pd.to_numeric(
            reference_data[column],
            errors="coerce"
        ).dropna()

        current_values = pd.to_numeric(
            current_data[column],
            errors="coerce"
        ).dropna()

        if len(reference_values) == 0:
            continue

        if len(current_values) == 0:
            continue

        reference_mean = (
            reference_values.mean()
        )

        current_mean = (
            current_values.mean()
        )

        reference_std = (
            reference_values.std()
        )

        if pd.isna(reference_std) or reference_std == 0:
            reference_std = 1

        mean_shift = abs(
            current_mean - reference_mean
        ) / abs(reference_std)

        drift_results.append(
            {
                "Feature": column,
                "Reference Mean": reference_mean,
                "Current Mean": current_mean,
                "Standardized Mean Shift":
                    mean_shift
            }
        )

    if drift_results:

        drift_table = pd.DataFrame(
            drift_results
        )

        drift_table = drift_table.sort_values(
            "Standardized Mean Shift",
            ascending=False
        )

        st.dataframe(
            drift_table,
            width="stretch"
        )

        st.subheader(
            "Drift Interpretation"
        )

        st.write(
            "A larger standardized mean shift indicates that "
            "the average value of a feature has changed more "
            "between the reference and current portions of "
            "the dataset."
        )

        st.info(
            "This is a simple monitoring check, not a formal "
            "statistical drift test. Further monitoring can "
            "use PSI, KS tests or dedicated monitoring tools."
        )

    else:

        st.warning(
            "No suitable numerical features were found for "
            "the drift check."
        )


# ============================================================
# 7. RESPONSIBLE AI
# ============================================================

elif page == "Responsible AI":

    st.header("Responsible AI")

    st.write(
        "Responsible AI principles are considered to improve "
        "transparency, fairness, privacy and appropriate use "
        "of the Advertisement Trend Detector."
    )

    # --------------------------------------------------------
    # Fairness
    # --------------------------------------------------------

    st.subheader(
        "1. Fairness"
    )

    st.write(
        "Fairness analysis was performed in Experiment 5 using "
        "target_gender as the sensitive attribute."
    )

    st.write(
        "Demographic Parity Difference, Equalized Odds "
        "Difference and accuracy by gender were evaluated."
    )

    st.write(
        "Fairness metrics should be monitored when new data "
        "or new user groups are introduced."
    )

    # --------------------------------------------------------
    # Privacy
    # --------------------------------------------------------

    st.subheader(
        "2. Privacy"
    )

    st.write(
        "Advertisement datasets should be handled securely "
        "and only information required for the intended "
        "analysis should be collected."
    )

    st.write(
        "Personal or sensitive information should not be "
        "exposed unnecessarily."
    )

    # --------------------------------------------------------
    # Consent
    # --------------------------------------------------------

    st.subheader(
        "3. Consent"
    )

    st.write(
        "If future versions of the system collect user-level "
        "or personally identifiable information, appropriate "
        "consent should be obtained before collecting or "
        "processing that information."
    )

    st.write(
        "Users should be informed about the purpose for which "
        "their data is collected and how it will be used."
    )

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------

    st.subheader(
        "4. Explainability"
    )

    st.write(
        "SHAP and LIME were used during the project to understand "
        "model behaviour and feature influence."
    )

    # --------------------------------------------------------
    # Human Oversight
    # --------------------------------------------------------

    st.subheader(
        "5. Human Oversight"
    )

    st.write(
        "Model predictions should be treated as decision-support "
        "information rather than guaranteed outcomes."
    )

    st.write(
        "Important business decisions should include human "
        "review."
    )

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    st.subheader(
        "6. Model Limitations"
    )

    st.write(
        "The model depends on the available training data and "
        "may not generalize to every advertising environment."
    )

    st.write(
        "Advertising trends can change over time, so the model "
        "should be monitored and evaluated periodically."
    )

    st.warning(
        "Important: The Experiment 6 model uses ROI-related "
        "information while High_ROI is derived from ROI. "
        "This can create target leakage and can make model "
        "performance appear higher than it would be on a "
        "properly designed future-campaign prediction task."
    )

    # --------------------------------------------------------
    # Responsible AI Checklist
    # --------------------------------------------------------

    st.subheader(
        "Responsible AI Checklist"
    )

    checklist = {
        "Fairness considered": True,
        "Privacy considered": True,
        "Consent considered": True,
        "Explainability included": True,
        "Human oversight included": True,
        "Model limitations documented": True,
        "Data drift monitoring included": True
    }

    for item, status in checklist.items():

        if status:
            st.checkbox(
                item,
                value=True,
                disabled=True
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.write(
    "Advertisement Trend Detector"
)

st.sidebar.write(
    "Experiment 8 - Final Portfolio"
)