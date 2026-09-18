"""Feature preparation for Individual Task 2, Part 2.

Reproduces the two pipelines from Task 1 exactly, so the reflection in Task 2
analyses the same models that were reported, not a rebuilt approximation. The
only addition is that each loader also returns the sensitive attributes needed
for the fairness analysis, which Task 1 never isolated.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
SVM_TRAIN_CAP = 15_000

# Data was downloaded for Task 1 and is not duplicated here.
DATA_ROOT = Path(__file__).resolve().parents[2] / "Assignment 1"
OLIST_DIR = DATA_ROOT / "olist"
FOOD_DIR = DATA_ROOT / "food_delivery"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
IMG_DIR = Path(__file__).resolve().parent / "img"


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _top_n(series, n, other="other"):
    keep = series.value_counts().nlargest(n).index
    return series.where(series.isin(keep), other)


def load_olist():
    """Returns X, y, and a frame of attributes held out of the model."""
    def read(name):
        return pd.read_csv(OLIST_DIR / f"olist_{name}_dataset.csv")

    orders, order_items = read("orders"), read("order_items")
    order_payments, order_reviews = read("order_payments"), read("order_reviews")
    customers, sellers = read("customers"), read("sellers")
    products, geolocation = read("products"), read("geolocation")
    category_translation = pd.read_csv(
        OLIST_DIR / "product_category_name_translation.csv", encoding="utf-8-sig")

    date_cols = ["order_purchase_timestamp", "order_approved_at",
                 "order_delivered_carrier_date", "order_delivered_customer_date",
                 "order_estimated_delivery_date"]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])

    df = orders[orders["order_status"] == "delivered"].copy()
    df = df.dropna(subset=["order_delivered_customer_date", "order_delivered_carrier_date"])
    df["late"] = (df["order_delivered_customer_date"]
                  > df["order_estimated_delivery_date"]).astype(int)

    HOUR, DAY = 3600.0, 86400.0
    df["approval_lag_hours"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / HOUR
    df["seller_processing_days"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / DAY
    df["promised_window_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / DAY
    df["window_used_at_handover"] = df["seller_processing_days"] / df["promised_window_days"]
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year

    df = df[~(df["seller_processing_days"] < 0)].copy()

    items = order_items.merge(products, on="product_id", how="left")
    items = items.merge(category_translation, on="product_category_name", how="left")
    items["product_volume_cm3"] = (items["product_length_cm"] * items["product_height_cm"]
                                   * items["product_width_cm"])
    item_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"), n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"), total_freight=("freight_value", "sum"),
        max_product_weight_g=("product_weight_g", "max"),
        total_product_volume_cm3=("product_volume_cm3", "sum"),
        n_photos=("product_photos_qty", "mean"))
    primary_item = (items.sort_values("price", ascending=False).groupby("order_id")
                    .agg(seller_id=("seller_id", "first"),
                         product_category=("product_category_name_english", "first")))
    item_agg = item_agg.join(primary_item).reset_index()

    payment_agg = order_payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"), max_installments=("payment_installments", "max"),
        n_payments=("payment_sequential", "count"))
    dominant_payment = (order_payments.sort_values("payment_value", ascending=False)
                        .groupby("order_id")["payment_type"].first().rename("payment_type"))
    payment_agg = payment_agg.join(dominant_payment).reset_index()

    rows_before = len(df)
    df = df.merge(item_agg, on="order_id", how="inner")
    assert len(df) <= rows_before, "order_items merge duplicated orders"
    df = df.merge(payment_agg, on="order_id", how="left")

    BRAZIL_LAT, BRAZIL_LNG = (-33.75, 5.27), (-73.99, -34.79)
    outside = ~(geolocation["geolocation_lat"].between(*BRAZIL_LAT)
                & geolocation["geolocation_lng"].between(*BRAZIL_LNG))
    centroids = (geolocation[~outside]
                 .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
                 .mean().reset_index())

    df = df.merge(customers[["customer_id", "customer_zip_code_prefix", "customer_state"]],
                  on="customer_id", how="left")
    df = df.merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]],
                  on="seller_id", how="left")
    for side in ["customer", "seller"]:
        df = df.merge(centroids.rename(columns={
            "geolocation_zip_code_prefix": f"{side}_zip_code_prefix",
            "geolocation_lat": f"{side}_lat", "geolocation_lng": f"{side}_lng"}),
            on=f"{side}_zip_code_prefix", how="left")

    df["distance_km"] = haversine_km(df["seller_lat"], df["seller_lng"],
                                     df["customer_lat"], df["customer_lng"])
    df["is_interstate"] = (df["customer_state"] != df["seller_state"]).astype(int)

    # Kept before the top-n collapse, since the fairness analysis needs the real
    # state rather than the "other" bucket the model sees.
    true_customer_state = df["customer_state"].copy()

    df["product_category"] = _top_n(df["product_category"].fillna("unknown"), 15)
    df["customer_state"] = _top_n(df["customer_state"], 12)
    df["seller_state"] = _top_n(df["seller_state"], 12)
    df["payment_type"] = df["payment_type"].fillna("unknown")

    review_scores = order_reviews.groupby("order_id")["review_score"].mean()
    df = df.merge(review_scores, on="order_id", how="left")

    numeric = ["seller_processing_days", "approval_lag_hours", "promised_window_days",
               "window_used_at_handover", "distance_km", "is_interstate",
               "total_freight", "total_price", "payment_value", "max_installments",
               "n_items", "n_sellers", "max_product_weight_g", "total_product_volume_cm3",
               "n_photos", "purchase_month", "purchase_weekday", "purchase_hour"]
    categorical = ["product_category", "payment_type", "customer_state", "seller_state"]

    X = pd.get_dummies(df[numeric + categorical], columns=categorical)
    y = df["late"].astype(int)

    sensitive = pd.DataFrame({
        "customer_state": true_customer_state.values,
        "region": true_customer_state.map(_BR_REGION).fillna("unknown").values,
        "seller_id": df["seller_id"].values,
        "purchase_year": df["purchase_year"].values,
        "distance_km": df["distance_km"].values,
        "order_value": df["total_price"].values,
        "review_score": df["review_score"].values,
    }, index=X.index)

    return X, y, sensitive


# Brazil's five official macro-regions. The North and Northeast carry the
# country's lowest incomes and thinnest logistics coverage, which is why
# region is the grouping of interest rather than an arbitrary state split.
_BR_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North",
    "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast",
    "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}


def load_food():
    """Returns X, y, and a frame of attributes held out of the fairness question."""
    food = pd.read_csv(FOOD_DIR / "train.csv")
    food_clean = food.copy()
    for col in food_clean.select_dtypes(include="object").columns:
        food_clean[col] = food_clean[col].str.strip()
    food_clean = food_clean.replace("NaN", np.nan)

    food_clean["Weatherconditions"] = food_clean["Weatherconditions"].str.replace(
        "conditions ", "", regex=False)
    food_clean["Weatherconditions"] = food_clean["Weatherconditions"].replace("NaN", np.nan)
    food_clean["time_taken_min"] = food_clean["Time_taken(min)"].str.extract(r"(\d+)").astype(float)

    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        food_clean[col] = pd.to_numeric(food_clean[col], errors="coerce")

    coord_cols = ["Restaurant_latitude", "Restaurant_longitude",
                  "Delivery_location_latitude", "Delivery_location_longitude"]
    for col in coord_cols:
        food_clean[col] = food_clean[col].abs()
    food_clean = food_clean[~(food_clean[coord_cols] < 1).any(axis=1)].copy()

    food_clean["distance_km"] = haversine_km(
        food_clean["Restaurant_latitude"], food_clean["Restaurant_longitude"],
        food_clean["Delivery_location_latitude"], food_clean["Delivery_location_longitude"])
    food_clean["order_date"] = pd.to_datetime(food_clean["Order_Date"], format="%d-%m-%Y",
                                              errors="coerce")
    food_clean["order_weekday"] = food_clean["order_date"].dt.dayofweek
    food_clean["is_weekend"] = (food_clean["order_weekday"] >= 5).astype(int)

    ordered = pd.to_timedelta(food_clean["Time_Orderd"], errors="coerce")
    picked = pd.to_timedelta(food_clean["Time_Order_picked"], errors="coerce")
    prep = (picked - ordered).dt.total_seconds() / 60.0
    food_clean["prep_minutes"] = prep.where(prep >= 0, prep + 24 * 60)
    food_clean["order_hour"] = ordered.dt.total_seconds() // 3600

    food_clean = food_clean.dropna(subset=["time_taken_min"]).copy()

    numeric = ["distance_km", "prep_minutes", "Delivery_person_Age",
               "Delivery_person_Ratings", "Vehicle_condition", "multiple_deliveries",
               "order_weekday", "is_weekend", "order_hour"]
    categorical = ["Road_traffic_density", "Weatherconditions", "Type_of_order",
                   "Type_of_vehicle", "Festival", "City"]

    X = pd.get_dummies(food_clean[numeric + categorical], columns=categorical)
    y = food_clean["time_taken_min"]

    # Age bands follow the age discrimination literature's convention of
    # treating the youngest and oldest workers as the groups at risk.
    age = food_clean["Delivery_person_Age"]
    sensitive = pd.DataFrame({
        "city_type": food_clean["City"].fillna("unknown").values,
        "age_band": pd.cut(age, [0, 24, 29, 34, 100],
                           labels=["under 25", "25-29", "30-34", "35+"]).astype(str).values,
        "courier_id": food_clean["Delivery_person_ID"].values,
        "rating_band": pd.cut(food_clean["Delivery_person_Ratings"], [0, 4.2, 4.6, 4.8, 5.0],
                              labels=["<=4.2", "4.2-4.6", "4.6-4.8", "4.8-5.0"]).astype(str).values,
        "vehicle": food_clean["Type_of_vehicle"].fillna("unknown").values,
        "festival": food_clean["Festival"].fillna("unknown").values,
    }, index=X.index)

    return X, y, sensitive
