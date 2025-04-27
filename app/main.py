import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.train_model import train_and_save_models
from core.predict_model import predict_with_manufacturer

MODEL_DIR = "model"
BACKUP_DIR = "backup"
DATA_DIR = "data"
NEZARADENE = "Nezaradené"

st.set_page_config(page_title="🧙‍♂️ Product Wizard", layout="wide")
st.title("🧙‍♂️ Product Wizard")

st.sidebar.markdown("""
## 🔮 About Product Wizard
Upload products, predict categories, retrain models, and explore insights — fully automated!
""")

# --- Prediction Section ---
st.header("1️⃣ Predict Single Product")
manufacturer = st.text_input("Manufacturer")
description = st.text_input("Product Description")

if st.button("Predict"):
    if manufacturer and description:
        prediction = predict_with_manufacturer(manufacturer, description, model_dir=MODEL_DIR)
        if prediction:
            st.success(f"Predicted category: **{prediction}**")
        else:
            st.error(f"No model found for manufacturer '{manufacturer}'. Please retrain first.")
    else:
        st.error("Please provide both Manufacturer and Description.")

# --- Batch Categorization ---
st.header("2️⃣ Batch Categorization (CSV or Excel Upload)")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if {"manufacturer", "description"}.issubset(df.columns):
            results = []
            alt_results = []

            for idx, row in df.iterrows():
                manufacturer = row["manufacturer"]
                description = row["description"]
                preds = predict_with_manufacturer(manufacturer, description, model_dir=MODEL_DIR, return_top2=True)

                if preds:
                    results.append(preds[0])
                    alt_results.append(preds[1] if len(preds) > 1 else NEZARADENE)
                else:
                    results.append("No model")
                    alt_results.append("")

            df["predicted_category"] = results
            df["alternative_category"] = alt_results

            st.success("✅ Categorization complete!")
            st.dataframe(df)

            # --- Charts ---
            st.subheader("📊 Category Distribution")
            cat_counts = df["predicted_category"].value_counts()

            fig1, ax1 = plt.subplots()
            ax1.pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            cat_counts.plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Number of Products")
            ax2.set_xlabel("Category")
            st.pyplot(fig2)

            # --- Download Link ---
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Categorized CSV",
                data=csv_data,
                file_name="categorized_products.csv",
                mime="text/csv"
            )

        else:
            st.error("Uploaded file must contain 'manufacturer' and 'description' columns.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- Retrain Model Section ---
st.header("3️⃣ Retrain Models (per Manufacturer)")
retrain_file = st.file_uploader("Upload CSV for Retraining", type=["csv"], key="retrain")

if retrain_file:
    df_retrain = pd.read_csv(retrain_file)
    if {"manufacturer", "description", "category"}.issubset(df_retrain.columns):
        train_and_save_models(df_retrain, model_dir=MODEL_DIR, backup_dir=BACKUP_DIR)
        st.success("✅ Models retrained and backups created successfully!")
    else:
        st.error("Retraining file must have columns: 'manufacturer', 'description', 'category'")

# --- Similarity Matching Section ---
st.header("4️⃣ Categorize via Allowed Categories (Similarity Matching)")

col1, col2 = st.columns(2)

with col1:
    products_file = st.file_uploader("Upload Products File (CSV: manufacturer, description)", type=["csv"], key="products_upload")

with col2:
    allowed_file = st.file_uploader("Upload Allowed Categories File (CSV: category)", type=["csv"], key="allowed_upload")

if products_file and allowed_file:
    try:
        products_df = pd.read_csv(products_file)
        allowed_df = pd.read_csv(allowed_file)

        if {"manufacturer", "description"}.issubset(products_df.columns) and {"category"}.issubset(allowed_df.columns):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            st.sidebar.subheader("🎛️ Similarity Matching Settings")
            similarity_threshold = st.sidebar.slider(
                "Select threshold for 'Nezaradené' assignment (0 = very tolerant, 1 = very strict)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01
            )

            allowed_categories = allowed_df["category"].dropna().unique().tolist()
            product_texts = (products_df["manufacturer"].fillna('') + " " + products_df["description"].fillna('')).tolist()

            vectorizer = TfidfVectorizer()
            all_texts = product_texts + allowed_categories
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            products_embeddings = tfidf_matrix[:len(product_texts)]
            allowed_embeddings = tfidf_matrix[len(product_texts):]

            sims = cosine_similarity(products_embeddings, allowed_embeddings)

            best_categories = []
            second_best_categories = []
            for sim_vector in sims:
                sorted_idx = np.argsort(sim_vector)[::-1]
                if sim_vector[sorted_idx[0]] > similarity_threshold:
                    best_categories.append(allowed_categories[sorted_idx[0]])
                    second_best_categories.append(allowed_categories[sorted_idx[1]] if len(sorted_idx) > 1 else "Nezaradené")
                else:
                    best_categories.append("Nezaradené")
                    second_best_categories.append("")

            products_df["predicted_category"] = best_categories
            products_df["alternative_category"] = second_best_categories

            st.success("✅ Similarity categorization complete!")
            st.dataframe(products_df)

            # --- Charts ---
            st.subheader("📊 Similarity Category Distribution")
            cat_counts = products_df["predicted_category"].value_counts()

            fig1, ax1 = plt.subplots()
            ax1.pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            cat_counts.plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Number of Products")
            ax2.set_xlabel("Category")
            st.pyplot(fig2)

            # --- Download Link ---
            csv_data = products_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Similarity Categorized CSV",
                data=csv_data,
                file_name="similarity_categorized_products.csv",
                mime="text/csv"
            )

        else:
            st.error("CSV must have correct columns: Products (manufacturer, description), Allowed Categories (category).")

    except Exception as e:
        st.error(f"Error: {e}")
