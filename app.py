import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder
import io
import joblib   # ✅ ADDED

from functions import (
    load_data, split_data, clean_data, train_models,
    get_feature_importance, detect_remove_outliers, smart_scale,
    tune_top_models_cal, tune_top_models_reg,
    param_grids_cal, param_grids_reg,
    generate_EDA_report, transform_normally_distribuation,
    TREE_MODELS, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
)

st.set_page_config(
    page_title="AutoML Model Selector",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #2b2b2b, #1e1e1e);
    color: white;
}
.header-box {
    background: rgba(45,45,45,0.95);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 15px 40px rgba(0,0,0,0.8);
    text-align: center;
    margin-bottom: 30px;
}
.title {
    font-size: 42px;
    font-weight: bold;
    color: #f5c542;
    animation: glow 2s infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 10px #f5c542; }
    to   { text-shadow: 0 0 30px #ffd84d; }
}
.stButton > button {
    background: #f5c542;
    color: black;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 22px;
    transition: 0.3s;
}
.stButton > button:hover {
    transform: scale(1.08);
    background: #ffd84d;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div class="title">🤖 AutoML Model Selector</div>
</div>
""", unsafe_allow_html=True)

_missing_libs = []
if not XGBOOST_AVAILABLE:
    _missing_libs.append("`xgboost`")
if not LIGHTGBM_AVAILABLE:
    _missing_libs.append("`lightgbm`")
if _missing_libs:
    st.warning(
        f"⚠️ {' aur '.join(_missing_libs)} install nahi hai. "
        "Inhe install karo: `pip install xgboost lightgbm` — "
        "ye models leaderboard mein nahi dikhenge abhi."
    )

uploaded_file = st.file_uploader("📂 Upload CSV Dataset (max 500MB)", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    with st.expander("⚙️ Advanced: Specify Ordinal Columns (optional)"):
        st.caption("Select columns that have a natural order, e.g. Low < Medium < High.")
        cat_cols_available = df.select_dtypes(include=["object", "category"]).columns.tolist()
        ordinal_cols = st.multiselect("Ordinal columns", cat_cols_available)

        ordinal_orders = {}
        for col in ordinal_cols:
            unique_vals = df[col].dropna().unique().tolist()
            user_order = st.text_input(
                f"Order for '{col}' (comma-separated, lowest → highest)",
                value=", ".join([str(v) for v in unique_vals]),
                key=f"order_{col}"
            )
            ordinal_orders[col] = [v.strip() for v in user_order.split(",")]

    X_train, X_test, y_train, y_test = split_data(df, target_col)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())
    with col2:
        st.subheader("📏 Dataset Info")
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

    with st.spinner("Generating EDA report..."):
        missing_df, df_describe = generate_EDA_report(df)

        st.subheader("Missing Values %")
        st.dataframe(missing_df)

        st.subheader("Descriptive Statistics")
        st.dataframe(df_describe)

        num_cols = df.select_dtypes(include=["number"]).columns
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            for col in num_cols:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.kdeplot(data=df, x=col, fill=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                pdf.savefig(fig)
                plt.close(fig)
        pdf_buffer.seek(0)
        st.download_button(
            "⬇ Download KDE Distribution PDF",
            pdf_buffer,
            file_name="numerical_distributions.pdf",
            mime="application/pdf"
        )

    with st.spinner("Converting skewed columns to normal distribution..."):
        X_train, X_test, skewed_cols = transform_normally_distribuation(X_train, X_test)

    pdf_buffer_clean = io.BytesIO()
    with PdfPages(pdf_buffer_clean) as pdf:
        for col in X_train.select_dtypes(include=["number"]).columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.kdeplot(data=X_train, x=col, fill=True, ax=ax)
            ax.set_title(f"Distribution of {col} (After Normalising)")
            pdf.savefig(fig)
            plt.close(fig)
    pdf_buffer_clean.seek(0)
    st.download_button(
        "⬇ Download KDE PDF (After Normalising)",
        pdf_buffer_clean,
        file_name="kde_after_normalising.pdf",
        mime="application/pdf"
    )
    st.success(f"✅ {len(skewed_cols)} skewed column(s) normalised")

    if y_train.dtype == object or str(y_train.dtype) == "category":
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train), index=y_train.index)
        y_test = pd.Series(le.transform(y_test), index=y_test.index)

    if st.button("🚀 Run AutoML"):

        with st.spinner("🧹 Cleaning data with smart strategies..."):
            X_train, X_test, y_train, clean_report = clean_data(
                X_train, X_test, y_train,
                ordinal_cols=ordinal_cols if ordinal_cols else None,
                ordinal_orders=ordinal_orders if ordinal_orders else None
            )

        st.success("✅ Data Cleaned Successfully")

        with st.spinner("🚨 Removing outliers..."):
            X_train, y_train, removed = detect_remove_outliers(X_train, y_train)

        if removed:
            st.success(f"✅ Outliers removed. New shape: {X_train.shape}")
        else:
            st.info("ℹ️ No outliers removed")

        with st.expander("🔍 Imputation Strategies Used"):
            imp_df = pd.DataFrame(list(clean_report["imputation"].items()), columns=["Column", "Strategy"])
            st.dataframe(imp_df)

        with st.expander("🔍 Encoding Strategies Used"):
            enc_df = pd.DataFrame(list(clean_report["encoding"].items()), columns=["Column", "Encoder"])
            st.dataframe(enc_df)

        df_clean = pd.concat([X_train, y_train], axis=1)
        st.subheader("🧹 Cleaned Dataset Preview")
        st.dataframe(df_clean.head())
        st.write("Cleaned Shape:", df_clean.shape)

        cleaned_csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Cleaned Data", cleaned_csv, file_name=f"cleaned_{uploaded_file.name}")

        with st.spinner("🤖 Training models..."):
            best_model_info, results, models, preprocessor = train_models(
                X_train, y_train,
                ordinal_cols=ordinal_cols if ordinal_cols else None
            )

        best_model_name = best_model_info[0]
        st.info(f"🏆 Best baseline model: {best_model_name}")

        result_df = pd.DataFrame(results, columns=["Model", "Score"])
        st.subheader("📋 Model Leaderboard")
        st.dataframe(result_df.sort_values("Score", ascending=False))

        with st.spinner("⚙️ Tuning hyperparameters..."):
            if y_train.nunique() < 10:
                tuned_results = tune_top_models_cal(
                    X_train, y_train, results, models, param_grids_cal, preprocessor
                )
            else:
                tuned_results = tune_top_models_reg(
                    X_train, y_train, results, models, param_grids_reg, preprocessor
                )

        st.subheader("🔥 Tuned Top Models")

        tuned_display = [
            {
                "Model": r["Model"],
                "Tuned Score": round(r["Tuned Score"], 4),
                "Best Params": str(r["Best Params"])
            }
            for r in tuned_results
        ]

        st.dataframe(
            pd.DataFrame(tuned_display).sort_values("Tuned Score", ascending=False)
        )

        st.subheader("📊 Feature Importance")
        best_est = tuned_results[0]["Best Estimator"] if tuned_results else None

        imp = get_feature_importance(X_train, y_train, best_model_name, best_est)

        fig, ax = plt.subplots()
        ax.barh(imp["Feature"][:10], imp["Importance"][:10])
        ax.invert_yaxis()
        st.pyplot(fig)

        # =============================
        # 🔥 DOWNLOAD TOP 2 MODELS
        # =============================
        st.subheader("⬇ Download Top Models")

        for i, res in enumerate(tuned_results[:2]):
            model_name = res["Model"]
            model = res["Best Estimator"]

            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)

            st.download_button(
                label=f"⬇ Download {model_name}",
                data=buffer,
                file_name=f"{model_name}_model.pkl",
                mime="application/octet-stream",
                key=f"download_{i}"
            )