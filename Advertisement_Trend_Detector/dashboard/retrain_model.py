
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Advertisement Trend Detector",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = "C:\\Users\\shriy\\OneDrive\\Pictures\\Desktop\\Advertisement_Trend_Detector\\data\\data\\cleaned_advertisement_dataset.csv"
MODEL_PATH = "C:\\Users\\shriy\\OneDrive\\Pictures\\Desktop\\Advertisement_Trend_Detector\\model\\campaign_roi_model.pkl"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

try:
    df = load_data()
    model = load_model()

except Exception as e:
    st.error("Unable to load the dataset or trained model.")
    st.write("Error:", str(e))
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("Advertisement Trend Detector")

st.write(
    "A machine learning dashboard for analyzing advertisement "
    "campaign trends and predicting High ROI campaigns."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Advertisement Trends",
        "High ROI Prediction",
        "Model Information"
    ]
)


# =========================================================
# 1. OVERVIEW
# =========================================================

if page == "Overview":

    st.header("Dashboard Overview")

    total_campaigns = len(df)

    high_roi_campaigns = int(
        pd.to_numeric(
            df["High_ROI"],
            errors="coerce"
        ).fillna(0).sum()
    )

    not_high_roi_campaigns = int(
        (
            pd.to_numeric(
                df["High_ROI"],
                errors="coerce"
            ).fillna(0) == 0
        ).sum()
    )

    if "platform" in df.columns:
        number_of_platforms = df["platform"].nunique()
    else:
        number_of_platforms = 0


    # Dashboard metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Campaigns",
            total_campaigns
        )

    with col2:
        st.metric(
            "High ROI Campaigns",
            high_roi_campaigns
        )

    with col3:
        st.metric(
            "Not High ROI",
            not_high_roi_campaigns
        )

    with col4:
        st.metric(
            "Platforms",
            number_of_platforms
        )


    st.divider()


    # Dataset preview

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )


    # Dataset information

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            "Number of Rows:",
            df.shape[0]
        )

    with col2:
        st.write(
            "Number of Columns:",
            df.shape[1]
        )


# =========================================================
# 2. ADVERTISEMENT TRENDS
# =========================================================

elif page == "Advertisement Trends":

    st.header("Advertisement Trends")


    # -----------------------------------------------------
    # Campaigns by Platform
    # -----------------------------------------------------

    if "platform" in df.columns:

        st.subheader("Campaigns by Platform")

        platform_data = (
            df.groupby("platform")
            .size()
            .reset_index(name="Campaigns")
        )

        fig = px.bar(
            platform_data,
            x="platform",
            y="Campaigns",
            title="Number of Advertisement Campaigns by Platform"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


    # -----------------------------------------------------
    # High ROI Distribution
    # -----------------------------------------------------

    if "High_ROI" in df.columns:

        st.subheader("High ROI Distribution")

        roi_data = (
            pd.to_numeric(
                df["High_ROI"],
                errors="coerce"
            )
            .fillna(0)
            .value_counts()
            .reset_index()
        )

        roi_data.columns = [
            "High_ROI",
            "Count"
        ]

        roi_data["High_ROI"] = roi_data[
            "High_ROI"
        ].map({
            0: "Not High ROI",
            1: "High ROI"
        })

        fig2 = px.pie(
            roi_data,
            names="High_ROI",
            values="Count",
            title="High ROI vs Not High ROI"
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )


    # -----------------------------------------------------
    # Average ROI by Platform
    # -----------------------------------------------------

    if (
        "platform" in df.columns
        and "roi" in df.columns
    ):

        st.subheader("Average ROI by Platform")

        roi_temp = df.copy()

        roi_temp["roi"] = pd.to_numeric(
            roi_temp["roi"],
            errors="coerce"
        )

        roi_platform = (
            roi_temp
            .groupby("platform")["roi"]
            .mean()
            .reset_index()
        )

        roi_platform.columns = [
            "Platform",
            "Average ROI"
        ]

        fig3 = px.bar(
            roi_platform,
            x="Platform",
            y="Average ROI",
            title="Average ROI by Platform"
        )

        st.plotly_chart(
            fig3,
            width="stretch"
        )


    # -----------------------------------------------------
    # Average Engagement Rate
    # -----------------------------------------------------

    if (
        "platform" in df.columns
        and "engagement_rate_percent" in df.columns
    ):

        st.subheader(
            "Average Engagement Rate by Platform"
        )

        engagement_temp = df.copy()

        engagement_temp[
            "engagement_rate_percent"
        ] = pd.to_numeric(
            engagement_temp[
                "engagement_rate_percent"
            ],
            errors="coerce"
        )

        engagement_platform = (
            engagement_temp
            .groupby("platform")[
                "engagement_rate_percent"
            ]
            .mean()
            .reset_index()
        )

        engagement_platform.columns = [
            "Platform",
            "Average Engagement Rate"
        ]

        fig4 = px.bar(
            engagement_platform,
            x="Platform",
            y="Average Engagement Rate",
            title="Average Engagement Rate by Platform"
        )

        st.plotly_chart(
            fig4,
            width="stretch"
        )


# =========================================================
# 3. HIGH ROI PREDICTION
# =========================================================

elif page == "High ROI Prediction":

    st.header("Campaign High ROI Prediction")

    st.write(
        "Enter campaign information below to predict whether "
        "the campaign is likely to have High ROI."
    )

    st.info(
        "The prediction uses the Random Forest model created "
        "in Experiment 6."
    )


    # -----------------------------------------------------
    # FEATURES USED FOR PREDICTION
    # -----------------------------------------------------

    input_columns = [
        col
        for col in df.columns
        if col not in [
            "High_ROI",
            "High_Engagement",
            "campaign_id",
            "date"
        ]
    ]


    input_data = {}


    st.subheader("Campaign Details")


    left_column, right_column = st.columns(2)


    # -----------------------------------------------------
    # CREATE INPUT FIELDS
    # -----------------------------------------------------

    for index, column in enumerate(input_columns):

        target_column = (
            left_column
            if index % 2 == 0
            else right_column
        )


        with target_column:

            # Convert values to numeric
            numeric_values = pd.to_numeric(
                df[column],
                errors="coerce"
            )


            numeric_count = numeric_values.notna().sum()


            # -------------------------------------------------
            # NUMERIC COLUMN
            # -------------------------------------------------

            if numeric_count > 0.8 * len(df):

                median_value = numeric_values.median()


                if pd.isna(median_value):
                    median_value = 0.0


                # Choose suitable step size

                if abs(median_value) >= 100:
                    step_value = 1.0

                elif abs(median_value) >= 1:
                    step_value = 0.01

                else:
                    step_value = 0.001


                input_data[column] = st.number_input(
                    column,
                    value=float(median_value),
                    step=float(step_value)
                )


            # -------------------------------------------------
            # CATEGORICAL COLUMN
            # -------------------------------------------------

            else:

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )


                values = sorted(values)


                if len(values) > 0:

                    input_data[column] = st.selectbox(
                        column,
                        values
                    )

                else:

                    input_data[column] = ""


    st.divider()


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button(
        "Predict High ROI",
        type="primary"
    ):

        try:

            # Create input DataFrame

            input_df = pd.DataFrame(
                [input_data]
            )


            # -------------------------------------------------
            # Make prediction
            # -------------------------------------------------

            prediction = model.predict(
                input_df
            )[0]


            # -------------------------------------------------
            # Prediction probability
            # -------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = (
                    model.predict_proba(
                        input_df
                    )[0].max()
                )

            else:

                probability = None


            # -------------------------------------------------
            # Display result
            # -------------------------------------------------

            st.subheader(
                "Prediction Result"
            )


            if prediction == 1:

                st.success(
                    "Prediction: HIGH ROI"
                )

            else:

                st.warning(
                    "Prediction: NOT HIGH ROI"
                )


            # -------------------------------------------------
            # Display confidence
            # -------------------------------------------------

            if probability is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{probability * 100:.2f}%"
                )


        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.write(
                "Error:",
                str(e)
            )


# =========================================================
# 4. MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.header("Model Information")


    st.subheader(
        "Machine Learning Model"
    )

    st.write(
        "Random Forest Classifier"
    )


    st.subheader(
        "Prediction Target"
    )

    st.write(
        "High_ROI"
    )


    st.subheader(
        "Target Meaning"
    )

    st.write(
        "1 = High ROI"
    )

    st.write(
        "0 = Not High ROI"
    )


    st.subheader(
        "Model Pipeline"
    )

    st.write(
        "The model uses preprocessing for numerical and "
        "categorical features followed by a Random Forest "
        "classifier."
    )


    st.subheader(
        "Training Information"
    )

    st.write(
        "Training/Test Split: 80/20"
    )

    st.write(
        "Random Forest Estimators: 100"
    )

    st.write(
        "Random State: 42"
    )


    st.subheader(
        "Model Accuracy"
    )

    st.write(
        "Test Accuracy: 100%"
    )


    st.subheader(
        "Dataset"
    )

    st.write(
        f"Number of records: {df.shape[0]}"
    )

    st.write(
        f"Number of columns: {df.shape[1]}"
    )


    st.success(
        "The trained model was loaded successfully."
    )
