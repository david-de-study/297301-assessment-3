import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

if 'baseline_forecast' not in st.session_state:
    st.session_state['baseline_forecast'] = 0

if 'ml_forecast' not in st.session_state:
    st.session_state['ml_forecast'] = 0
    
st.set_page_config(page_title="Retail Inventory Prediction", layout="wide")
st.title("Retail Inventory & Decision Support Dashboard")

# Load datasets from cache
@st.cache_data 
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

# Cache loading model and preprocessor
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except Exception as e:
        st.error(f"⚠️ Failed to load model files. Ensure model.joblib and preprocessor.joblib are in the same directory.\nError: {e}")
        return None, None

df = load_data()
model, preprocessor = load_ml_components()

# Sidebar input boxes
st.sidebar.header("Control Panel")

# Product Category
selected_family = st.sidebar.selectbox(
    "Product Category", 
    sorted(df['family'].unique()),
    key="product_category"
)

st.sidebar.divider()

# Previous Week Date
st.sidebar.subheader("Reference Week")
selected_date = st.sidebar.date_input(
    "Previous Week Date",
    value=df['date'].max(),
    min_value=df['date'].min(),
    max_value=df['date'].max(),
    key="ref_week_date"
)

st.sidebar.divider()

# Lagged Sales
st.sidebar.subheader("Lagged Sales (Units)")
lag_sales_1 = st.sidebar.number_input("Week 1 (Most Recent)", min_value=0, value=0, step=1, key="sales_1")
lag_sales_2 = st.sidebar.number_input("Week 2", min_value=0, value=0, step=1, key="sales_2")
lag_sales_3 = st.sidebar.number_input("Week 3", min_value=0, value=0, step=1, key="sales_3")

st.sidebar.divider()

# Lagged Transactions
st.sidebar.subheader("Lagged Transactions")
lag_trans_1 = st.sidebar.number_input("Week 1 (Most Recent)", min_value=0, value=0, step=1, key="trans_1")
lag_trans_2 = st.sidebar.number_input("Week 2", min_value=0, value=0, step=1, key="trans_2")
lag_trans_3 = st.sidebar.number_input("Week 3", min_value=0, value=0, step=1, key="trans_3")

st.sidebar.divider()

# On Promotion Count
st.sidebar.subheader("On Promotion Count")
lag_promo_1 = st.sidebar.number_input("Week 1 (Most Recent)", min_value=0, value=0, step=1, key="promo_1")
lag_promo_2 = st.sidebar.number_input("Week 2", min_value=0, value=0, step=1, key="promo_2")
lag_promo_3 = st.sidebar.number_input("Week 3", min_value=0, value=0, step=1, key="promo_3")

col1, col2, col3, col4 = st.columns(4)

st.caption("""
**Findings**: The naive previous-week baseline performs well on outlier values that don't fall within typical ranges. 
The ML model captures non-linear effects of promotions, transaction patterns, and seasonality for scenario planning.
""")

st.divider()

# Prediction and Comparison Section
# st.subheader("2. Next Week Forecast Comparison")

if st.button("Generate Forecast", type="primary", key="forecast_btn"):
    if model is None or preprocessor is None:
        st.error("Model not loaded. Cannot run prediction.")
    else:
        with st.spinner('Running Gradient Boosting model inference...'):
            time.sleep(0.5)
            
            sales_2avg = (lag_sales_1 + lag_sales_2) / 2
            trans_2avg = (lag_trans_1 + lag_trans_2) / 2
            sales_3avg = (lag_sales_1 + lag_sales_2 + lag_sales_3) / 3
            trans_3avg = (lag_trans_1 + lag_trans_2 + lag_trans_3) / 3
            sales_4avg = (lag_sales_1 + lag_sales_2 + lag_sales_3 + lag_sales_3) / 4
            trans_4avg = (lag_trans_1 + lag_trans_2 + lag_trans_3 + lag_trans_3) / 4
            
            month = selected_date.month
            week_of_year = selected_date.isocalendar()[1]
            
            feature_data = {
                'store_nbr': 1,
                'family': selected_family,
                'previous_transactions_1': lag_trans_1,
                'previous_unit_sales_1': lag_sales_1,
                'previous_onpromotion_1': lag_promo_1,
                'previous_transactions_2': lag_trans_2,
                'previous_unit_sales_2': lag_sales_2,
                'previous_onpromotion_2': lag_promo_2,
                'previous_transactions_3': lag_trans_3,
                'previous_unit_sales_3': lag_sales_3,
                'previous_onpromotion_3': lag_promo_3,
                'previous_transactions_4_avg': trans_4avg,
                'previous_unit_sales_4_avg': sales_4avg,
                'previous_transactions_3_avg': trans_3avg,
                'previous_unit_sales_3_avg': sales_3avg,
                'previous_transactions_2_avg': trans_2avg,
                'previous_unit_sales_2_avg': sales_2avg,
                'month': month,
                'week_of_year': week_of_year
            }
            
            input_df = pd.DataFrame([feature_data])
            
            try:
                X_processed = preprocessor.transform(input_df)
                ml_forecast = model.predict(X_processed)[0]
                st.session_state['ml_forecast'] = max(0, ml_forecast)
            except Exception as e:
                st.warning(f"Model prediction error, using fallback. Error: {str(e)[:80]}")
                # st.session_state['ml_forecast'] = lag_sales_1
            
            st.session_state['baseline_forecast'] = lag_sales_1

            col1.metric(
                label="Baseline Prediction (Using previous value)", 
                value=f"{int(st.session_state.baseline_forecast)} Units",
                delta="Naive reference method",
                delta_color="off"
            )

            col2.metric(
                label="ML Prediction", 
                value=f"{int(st.session_state.ml_forecast)} Units",
                delta="Naive reference method"
            )
            
            # Chart Display
            result_df = pd.DataFrame([
                {'Method': 'Baseline (Previous Week)', 'Forecast (Units)': int( st.session_state.baseline_forecast)},
                {'Method': 'ML Model (Gradient Boosting)', 'Forecast (Units)': int( st.session_state.ml_forecast)}
            ])
            
            st.bar_chart(result_df, x='Method', y='Forecast (Units)')
            
            # Business Analysis
            st.success("✅ Forecast generated successfully.")
            
            st.info(f"🧠 **Decision Support Report**\n\n"
                    f"- **Baseline Forecast**: **{int( st.session_state.baseline_forecast)} units** (naive previous-week projection)\n"
                    f"- **ML Model Forecast**: **{int( st.session_state.ml_forecast)} units** (Gradient Boosting with lag features & seasonality)\n\n"
                    f"**Recommendation**: Use the baseline for routine replenishment. "
                    f"Reference the ML model when planning around promotions, seasonal shifts, or unusual transaction patterns.")
