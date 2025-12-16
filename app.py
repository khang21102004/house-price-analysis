import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Dashboard", layout="wide")

@st.cache_data
def load_data(path="kc_house_data.csv"):
    df = pd.read_csv(path)
    # tạo log_price + parse date (nếu bạn muốn)
    df["log_price"] = np.log1p(df["price"])
    return df

@st.cache_resource
def load_model(path="model.pkl"):
    return joblib.load(path)

df = load_data()
model = load_model()

page = st.sidebar.radio("Chọn trang", ["Overview", "EDA", "Predict", "Model Explain"])

# ===== Filters chung (tuỳ chọn) =====
st.sidebar.subheader("Bộ lọc")
if "zipcode" in df.columns:
    zips = sorted(df["zipcode"].unique().tolist())
    zip_choice = st.sidebar.selectbox("Zipcode", ["All"] + zips)
else:
    zip_choice = "All"

df_f = df.copy()
if zip_choice != "All":
    df_f = df_f[df_f["zipcode"] == zip_choice]

# ===== PAGE 1: Overview =====
if page == "Overview":
    st.title("📊 Tổng quan dữ liệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số mẫu", f"{len(df_f):,}")
    c2.metric("Giá trung vị", f"${df_f['price'].median():,.0f}")
    c3.metric("Giá trung bình", f"${df_f['price'].mean():,.0f}")
    c4.metric("Giá max", f"${df_f['price'].max():,.0f}")

    st.subheader("Phân phối giá")
    fig = plt.figure()
    plt.hist(df_f["price"], bins=50)
    plt.xlabel("Price")
    plt.ylabel("Count")
    st.pyplot(fig)

# ===== PAGE 2: EDA =====
elif page == "EDA":
    st.title("🔎 EDA & Insight")

    st.subheader("sqft_living vs price")
    fig = plt.figure()
    plt.scatter(df_f["sqft_living"], df_f["price"], s=10, alpha=0.3)
    plt.xlabel("sqft_living")
    plt.ylabel("price")
    st.pyplot(fig)

    st.subheader("Giá theo waterfront")
    if "waterfront" in df_f.columns:
        fig = plt.figure()
        df_f.boxplot(column="price", by="waterfront", grid=False)
        plt.title("Price by waterfront")
        plt.suptitle("")
        st.pyplot(fig)

# ===== PAGE 3: Predict =====
elif page == "Predict":
    st.title("🏠 Dự đoán giá nhà")

    # form nhập tối thiểu (bạn có thể mở rộng)
    col1, col2 = st.columns(2)
    with col1:
        bedrooms = st.number_input("bedrooms", 0, 20, 3)
        bathrooms = st.number_input("bathrooms", 0.0, 10.0, 2.0)
        sqft_living = st.number_input("sqft_living", 300, 20000, 1800)
        sqft_lot = st.number_input("sqft_lot", 500, 200000, 5000)

    with col2:
        grade = st.slider("grade", 1, 13, 7)
        condition = st.slider("condition", 1, 5, 3)
        waterfront = st.selectbox("waterfront", [0, 1])
        view = st.slider("view", 0, 4, 0)

    zipcode = st.text_input("zipcode", "98178")
    lat = st.number_input("lat", 47.0, 48.0, 47.5)
    long = st.number_input("long", -123.5, -121.0, -122.2)
    yr_built = st.number_input("yr_built", 1800, 2025, 1980)
    yr_renovated = st.number_input("yr_renovated", 0, 2025, 0)
    year_sold = st.number_input("year_sold", 2014, 2016, 2015)
    month_sold = st.slider("month_sold", 1, 12, 6)

    # feature engineering giống lúc train
    house_age = max(0, year_sold - yr_built)
    is_renovated = 1 if yr_renovated > 0 else 0
    sqft_living_ratio = sqft_living / (sqft_lot + 1)

    # default cho các cột còn lại
    sqft_above = int(sqft_living * 0.7)
    sqft_basement = int(sqft_living * 0.3)
    sqft_living15 = int(sqft_living * 0.9)
    sqft_lot15 = int(sqft_lot * 0.9)

    try:
        zipcode_int = int(zipcode)
    except:
        zipcode_int = int(df["zipcode"].mode()[0])

    input_df = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": 1.0,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": zipcode_int,
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
        "year_sold": year_sold,
        "month_sold": month_sold,
        "house_age": house_age,
        "is_renovated": is_renovated,
        "sqft_living_ratio": sqft_living_ratio
    }])

    if st.button("Predict"):
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)
        st.success(f"Giá dự đoán: ${pred_price:,.0f}")

# ===== PAGE 4: Model Explain =====
else:
    st.title("🧠 Giải thích mô hình")
    st.write("Gợi ý: hiển thị feature importance và residual plots (lấy từ Section 4).")
