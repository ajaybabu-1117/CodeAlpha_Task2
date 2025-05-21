import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Data Preprocessor", layout="centered")
st.title("🔧 Data Preprocessing Interface")

# 1️⃣ Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Uploaded Successfully")
    st.write("### Preview of Dataset", df.head())

    # 2️⃣ Target Column Input
    target_column = st.text_input("🎯 Enter the target column name (case-sensitive):")

    if target_column:
        if target_column not in df.columns:
            st.error(f"❌ Column '{target_column}' not found in dataset. Please try again.")
        else:
            st.info("🚀 Starting preprocessing...")

            # Handle missing values
            for col in df.select_dtypes(include=np.number).columns:
                df[col].fillna(df[col].median(), inplace=True)
            for col in df.select_dtypes(include='object').columns:
                df[col].fillna(df[col].mode()[0], inplace=True)

            # Handle outliers
            for col in df.select_dtypes(include=np.number).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]

            # Encode categoricals
            df = pd.get_dummies(df, drop_first=True)

            # Scale numerical features
            scaler = MinMaxScaler()
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = scaler.fit_transform(df[num_cols])

            # Split data
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Output results
            st.success("✅ Preprocessing Done!")
            st.write("### 🔍 Final Data Preview", df.head())
            st.write(f"📊 X_train shape: {X_train.shape}")
            st.write(f"📊 X_test shape: {X_test.shape}")
            st.write(f"🎯 y_train shape: {y_train.shape}")
            st.write(f"🎯 y_test shape: {y_test.shape}")
