# app.py
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import joblib
from mlflow import MlflowClient

import os
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}")

CLS_VERSION = 1
REG_VERSION = 1

client = MlflowClient()

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# ---------------------------
# 1. Load models + artifacts
# ---------------------------
def load_artifacts():
    cls_model = mlflow.xgboost.load_model(
        f"models:/RealEstate_GoodInvestment_Classifier/{CLS_VERSION}"
    )
    reg_model = mlflow.xgboost.load_model(
        f"models:/RealEstate_FuturePrice_Regressor/{REG_VERSION}"
    )

    # Make sure these paths point to the files you saved in training
    num_scaler = joblib.load("num_scaler.pkl")              # fitted RobustScaler
    cat_ohe = joblib.load("cat_ohe.pkl")                    # fitted OneHotEncoder
    te = joblib.load("target_encoder_locality.pkl")         # fitted target encoder
    lookup = pd.read_csv("lookup_values.csv").drop_duplicates()

    return cls_model, reg_model, num_scaler, cat_ohe, te, lookup


cls_model, reg_model, num_scaler, cat_ohe, te, lookup = load_artifacts()

numeric_features = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "school_density_score",
    "hospital_density_score",
    "floor_position_ratio",
    "age_score",
    "amenity_score",
    "ready_to_move",
]

categorical_low = [
    "Property_Type",
    "Furnished_Status",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Owner_Type",
    "Availability_Status",
]

# EXACT columns/order as X_base in training (before scaling/OHE)
base_cols = numeric_features + categorical_low + ["Locality_target_encoded"]

# ---------------------------
# 2. Sidebar filters (optional)
# ---------------------------
st.title("🏠 Real Estate Investment Advisor")

st.sidebar.header("Filter Properties")
min_price = st.sidebar.slider("Min Price (Lakhs)", 10, 500, 50)
max_price = st.sidebar.slider("Max Price (Lakhs)", 10, 500, 200)
min_bhk = st.sidebar.slider("Min BHK", 1, 6, 2)
max_bhk = st.sidebar.slider("Max BHK", 1, 6, 4)
min_size = st.sidebar.slider("Min Size (SqFt)", 300, 5000, 800)
max_size = st.sidebar.slider("Max Size (SqFt)", 300, 5000, 2500)

# ---------------------------
# 3. Input form
# ---------------------------
st.subheader("Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("City", sorted(lookup["City"].unique()))
    locality = st.selectbox("Locality", sorted(lookup["Locality"].unique()))
    bhk = st.slider("BHK", 1, 6, 3)
    size = st.number_input("Size (SqFt)", 300, 10000, 1200)

with col2:
    price_lakhs = st.number_input(
        "Current Price (Lakhs)", 10.0, 1000.0, 100.0, step=1.0
    )
    floor_no = st.slider("Floor No.", 0, 50, 2)
    total_floors = st.slider("Total Floors", 1, 60, 10)
    age = st.slider("Age of Property (yrs)", 0, 40, 8)

with col3:
    property_type = st.selectbox(
        "Property Type", sorted(lookup["Property_Type"].unique())
    )
    furnished = st.selectbox(
        "Furnished Status", sorted(lookup["Furnished_Status"].unique())
    )
    transport = st.selectbox(
        "Public Transport Accessibility",
        sorted(lookup["Public_Transport_Accessibility"].unique()),
    )
    parking = st.selectbox(
        "Parking Space", sorted(lookup["Parking_Space"].unique())
    )
    security = st.selectbox("Security", sorted(lookup["Security"].unique()))
    owner_type = st.selectbox("Owner Type", sorted(lookup["Owner_Type"].unique()))
    availability = st.selectbox(
        "Availability Status", sorted(lookup["Availability_Status"].unique())
    )

nearby_schools = st.slider("Nearby Schools", 0, 10, 3)
nearby_hospitals = st.slider("Nearby Hospitals", 0, 10, 2)

# ---------------------------
# 4. Build single-row DataFrame
# ---------------------------
def build_feature_row():
    row = {
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price_lakhs,
        "Price_per_SqFt": price_lakhs * 100000 / size,
        "Year_Built": 2025 - age,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Property_Type": property_type,
        "Furnished_Status": furnished,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Locality": locality,
    }

    df_input = pd.DataFrame([row])

    # Engineered features (must match training)
    df_input["school_density_score"] = (
        df_input["Nearby_Schools"] / (df_input["Size_in_SqFt"] / 1000)
    )
    df_input["hospital_density_score"] = (
        df_input["Nearby_Hospitals"] / (df_input["Size_in_SqFt"] / 1000)
    )
    df_input["floor_position_ratio"] = (
        df_input["Floor_No"] / df_input["Total_Floors"]
    )
    df_input["age_score"] = 1 / (1 + df_input["Age_of_Property"])
    df_input["amenity_score"] = (
        (df_input["Parking_Space"] == "Yes").astype(int)
        + (df_input["Security"] == "Yes").astype(int)
        + (df_input["Furnished_Status"] != "Unfurnished").astype(int)
    )
    df_input["ready_to_move"] = (
        df_input["Availability_Status"] == "Ready_to_Move"
    ).astype(int)

    # Target encode locality with fitted encoder
    df_input["Locality_target_encoded"] = te.transform(df_input[["Locality"]])

    return df_input


# ---------------------------
# 5. Prediction
# ---------------------------
if st.button("Analyze Investment"):
    df_input = build_feature_row()

    # 1) Scale base numeric features (10 raw numerics)
    num_raw_cols = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]

    num_scaled = pd.DataFrame(
        num_scaler.transform(df_input[num_raw_cols]),
        columns=[
            "BHK_scaled",
            "Size_in_SqFt_scaled",
            "Price_in_Lakhs_scaled",
            "Price_per_SqFt_scaled",
            "Year_Built_scaled",
            "Floor_No_scaled",
            "Total_Floors_scaled",
            "Age_of_Property_scaled",
            "Nearby_Schools_scaled",
            "Nearby_Hospitals_scaled",
        ],
    )

    # 2) One-hot encode low-cardinality categoricals (7 → 11)
    cat_encoded = pd.DataFrame(
        cat_ohe.transform(df_input[categorical_low]),
        columns=[
            "Property_Type_Independent House",
            "Property_Type_Villa",
            "Furnished_Status_Semi-furnished",
            "Furnished_Status_Unfurnished",
            "Public_Transport_Accessibility_Low",
            "Public_Transport_Accessibility_Medium",
            "Parking_Space_Yes",
            "Security_Yes",
            "Owner_Type_Builder",
            "Owner_Type_Owner",
            "Availability_Status_Under_Construction",
        ],
    )

    # 3) Final feature matrix – EXACT order used in training (28 cols)
    X_input = pd.concat(
        [
            num_scaled,
            cat_encoded,
            df_input[["Locality_target_encoded"]],
            df_input[
                [
                    "school_density_score",
                    "hospital_density_score",
                    "floor_position_ratio",
                    "age_score",
                    "amenity_score",
                    "ready_to_move",
                ]
            ],
        ],
        axis=1,
    )

    # 4) Classification: good investment probability
    cls_proba = float(cls_model.predict_proba(X_input)[0, 1])
    cls_label = "Good Investment" if cls_proba >= 0.3 else "Not Ideal"

    # 5) Regression: future price, then ROI
    future_log_price = float(reg_model.predict(X_input)[0])
    future_price_5y = np.expm1(future_log_price)
    roi_pct = (future_price_5y / price_lakhs - 1) * 100

    # 6) Display metrics
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(
            "Investment Decision",
            cls_label,
            f"{cls_proba * 100:.1f}% confidence",
        )
    with colB:
        st.metric(
            "Estimated Price (5Y)",
            f"₹{future_price_5y:,.1f} Lakhs",
        )
    with colC:
        st.metric(
            "Expected 5Y ROI",
            f"{roi_pct:.1f}%",
        )

    # Optional debug prints
    st.write("✅ df_input shape:", df_input.shape)
    st.write("✅ X_input shape:", X_input.shape)
    st.write("🎯 FINAL cls_proba:", cls_proba)
