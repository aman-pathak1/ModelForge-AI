import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*valid feature names.*"
)
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, AdaBoostRegressor,
                               ExtraTreesRegressor, ExtraTreesClassifier,
                               GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from category_encoders import TargetEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# -----------------------------
# Tree-based models — no scaling needed
# -----------------------------
TREE_MODELS = {
    "DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier",
    "ExtraTreesClassifier", "AdaBoostClassifier",
    "DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor",
    "ExtraTreesRegressor", "AdaBoostRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor"
}

# -----------------------------
# Load Data
# -----------------------------
def load_data(file_name):
    df = pd.read_csv(file_name, encoding='latin1')
    return df

# -----------------------------
# Train Test Split
# -----------------------------
def split_data(df, target_col):
    cols_to_drop = []
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["id", "name", "unnamed", "index"]):
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# -----------------------------
# Smart Imputation Strategy
# Numerical:
#   - If outliers detected (IQR method) → median
#   - If skewed (|skew| > 1)            → median
#   - Otherwise                         → mean
# Categorical:
#   - Always most_frequent
# -----------------------------
def _choose_num_impute_strategy(series):
    """Pick mean or median based on skew and outliers."""
    clean = series.dropna()
    if len(clean) == 0:
        return "mean"

    skewness = abs(clean.skew())

    Q1 = clean.quantile(0.25)
    Q3 = clean.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_pct = ((clean < lower) | (clean > upper)).mean()

    if skewness > 1 or outlier_pct > 0.05:
        return "median"
    return "mean"

def smart_impute(X_train, X_test):
    """
    Impute each column with the best strategy detected from training data.
    Returns imputed X_train, X_test and a dict of strategies used.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    strategies_used = {}

    # Numerical — column-by-column
    for col in num_cols:
        if X_train[col].isnull().sum() == 0:
            strategies_used[col] = "no missing"
            continue
        strategy = _choose_num_impute_strategy(X_train[col])
        strategies_used[col] = strategy
        imputer = SimpleImputer(strategy=strategy)
        X_train[[col]] = imputer.fit_transform(X_train[[col]])
        X_test[[col]]  = imputer.transform(X_test[[col]])

    # Categorical — most_frequent
    for col in cat_cols:
        if X_train[col].isnull().sum() == 0:
            strategies_used[col] = "no missing"
            continue
        strategies_used[col] = "most_frequent"
        imputer = SimpleImputer(strategy="most_frequent")
        X_train[[col]] = imputer.fit_transform(X_train[[col]])
        X_test[[col]]  = imputer.transform(X_test[[col]])

    return X_train, X_test, strategies_used

# -----------------------------
# Smart Encoding Strategy
# - Ordinal cols (user-specified) → OrdinalEncoder
# - Low cardinality (≤10)        → OHE  (but if OHE would create >50 cols → OrdinalEncoder instead)
# - Medium cardinality (10–50)   → TargetEncoder
# - High cardinality (>50)       → FrequencyEncoder
# -----------------------------
def smart_encode(X_train, X_test, y_train, ordinal_cols=None, ordinal_orders=None):
    """
    ordinal_cols  : list of column names that are ordinal (user-specified)
    ordinal_orders: dict {col: [ordered_list]} e.g. {"size": ["S","M","L","XL"]}
                    If not provided for an ordinal col, categories are inferred.
    Returns encoded X_train, X_test and encoding report dict.
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    if ordinal_cols is None:
        ordinal_cols = []
    if ordinal_orders is None:
        ordinal_orders = {}

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    encoding_report = {}

    # 1. Ordinal encoding (user-specified columns)
    for col in ordinal_cols:
        if col not in cat_cols:
            continue
        if col in ordinal_orders:
            categories = [ordinal_orders[col]]
        else:
            categories = "auto"
        enc = OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[[col]] = enc.fit_transform(X_train[[col]])
        X_test[[col]]  = enc.transform(X_test[[col]])
        encoding_report[col] = "OrdinalEncoder"

    remaining_cats = [c for c in cat_cols if c not in ordinal_cols]

    ohe_cols      = []
    ohe_as_ord    = []   # low cardinality but OHE would explode columns
    target_cols   = []
    freq_cols     = []

    for col in remaining_cats:
        n_unique = X_train[col].nunique()
        if n_unique <= 10:
            # Check if OHE would cause column explosion
            if n_unique <= 10 and n_unique * 1 <= 15:   # safe threshold
                ohe_cols.append(col)
            else:
                ohe_as_ord.append(col)
        elif n_unique <= 50:
            target_cols.append(col)
        else:
            freq_cols.append(col)

    # 2. One-Hot Encoding
    if ohe_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
        train_enc = ohe.fit_transform(X_train[ohe_cols])
        test_enc  = ohe.transform(X_test[ohe_cols])

        new_cols = ohe.get_feature_names_out(ohe_cols)

        X_train = pd.concat([
            X_train.drop(columns=ohe_cols),
            pd.DataFrame(train_enc, columns=new_cols, index=X_train.index)
        ], axis=1)
        X_test = pd.concat([
            X_test.drop(columns=ohe_cols),
            pd.DataFrame(test_enc, columns=new_cols, index=X_test.index)
        ], axis=1)

        for col in ohe_cols:
            encoding_report[col] = "OneHotEncoder"

    # 3. OrdinalEncoder for low-cardinality cols where OHE would explode
    for col in ohe_as_ord:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[[col]] = enc.fit_transform(X_train[[col]])
        X_test[[col]]  = enc.transform(X_test[[col]])
        encoding_report[col] = "OrdinalEncoder (cardinality-safe)"

    # 4. Target Encoding (medium cardinality)
    if target_cols:
        te = TargetEncoder(cols=target_cols)
        X_train[target_cols] = te.fit_transform(X_train[target_cols], y_train)
        X_test[target_cols]  = te.transform(X_test[target_cols])
        for col in target_cols:
            encoding_report[col] = "TargetEncoder"

    # 5. Frequency Encoding (high cardinality)
    for col in freq_cols:
        freq_map = X_train[col].value_counts(normalize=True)
        X_train[col] = X_train[col].map(freq_map).fillna(0)
        X_test[col]  = X_test[col].map(freq_map).fillna(0)
        encoding_report[col] = "FrequencyEncoder"

    return X_train, X_test, encoding_report

# -----------------------------
# Smart Outlier Detection
# Only apply to non-tree models
# Uses IQR method
# -----------------------------
def detect_remove_outliers(X, y, best_model_name=None):
    """
    Remove outliers using IQR. Skipped if best model is tree-based.
    Returns X_clean, y_clean, and whether outliers were removed.
    """
    if best_model_name and best_model_name in TREE_MODELS:
        return X, y, False  # Tree models don't need outlier removal

    num_cols = X.select_dtypes(include=["number"]).columns
    Q1 = X[num_cols].quantile(0.25)
    Q3 = X[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = ~((X[num_cols] < lower) | (X[num_cols] > upper)).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    return X_clean, y_clean, True

# -----------------------------
# Smart Scaling
# Skip scaling for tree-based models
# -----------------------------
def smart_scale(X_train, X_test, best_model_name=None):
    """
    Apply StandardScaler only for non-tree models.
    Returns scaled X_train, X_test and whether scaling was applied.
    """
    if best_model_name and best_model_name in TREE_MODELS:
        return X_train, X_test, False  # No scaling needed

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, True

# -----------------------------
# Full Data Cleaning Pipeline
# (replaces old clean_data)
# -----------------------------
def clean_data(X_train, X_test, y_train, ordinal_cols=None, ordinal_orders=None):
    """
    Full pipeline:
    1. Smart imputation
    2. Smart encoding
    Returns cleaned X_train, X_test, y_train + report dict
    """
    report = {}

    # Step 1: Smart imputation
    X_train, X_test, impute_report = smart_impute(X_train, X_test)
    report["imputation"] = impute_report

    # Step 2: Smart encoding
    X_train, X_test, encode_report = smart_encode(
        X_train, X_test, y_train,
        ordinal_cols=ordinal_cols,
        ordinal_orders=ordinal_orders
    )
    report["encoding"] = encode_report

    return X_train, X_test, y_train, report

# -----------------------------
# Model Training
# Har model ke liye internally scaling decide hoti hai:
# Tree models → no scaling, baaki → StandardScaler
# -----------------------------
from sklearn.preprocessing import LabelEncoder
def train_models(X_train, y_train, ordinal_cols=None):


    X = X_train
    y = y_train.copy()

    # Encode target
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    # Problem type
    problem = "classification" if y.nunique() < 10 else "regression"

    # =========================
    # PREPROCESSOR
    # =========================
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if ordinal_cols is None:
        ordinal_cols = []

    nominal_cols = [c for c in cat_cols if c not in ordinal_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    nom_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("ord", ord_pipe, ordinal_cols),
        ("nom", nom_pipe, nominal_cols)
    ])

    # =========================
    # MODELS
    # =========================
    if problem == "classification":
        models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(),
        "SVC": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "NaiveBayes": GaussianNB(),
        "ExtraTreesClassifier": ExtraTreesClassifier()
         }
        if XGBOOST_AVAILABLE:
            models["XGBClassifier"] = XGBClassifier(eval_metric="logloss", verbosity=0)

        if LIGHTGBM_AVAILABLE:
            models["LGBMClassifier"] = LGBMClassifier(verbosity=-1)

        scoring = "accuracy"

    else:
        models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "ExtraTreesRegressor": ExtraTreesRegressor()
        }
        scoring = "r2"
    results = []

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        score = cross_val_score(pipe, X, y, cv=5, scoring=scoring).mean()
        results.append((name, score))

    best = max(results, key=lambda x: x[1])

    return best, results, models, preprocessor
# -----------------------------
# Feature Importance
# -----------------------------
def get_feature_importance(X_train, y_train, best_model_name=None, tuned_model=None):
    """
    Use the tuned best model if it has feature_importances_.
    Fallback to a fresh RandomForest.
    """
    if tuned_model is not None and hasattr(tuned_model, "feature_importances_"):
        model = tuned_model
        fitted = True
    else:
        model = RandomForestClassifier() if y_train.nunique() < 10 else RandomForestRegressor()
        fitted = False

    if not fitted:
        model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return importance_df

# -----------------------------
# Param grids for tuning
# -----------------------------
param_grids_cal = {
    "RandomForestClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    },
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1]
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "ExtraTreesClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0]
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9]
    },
    "XGBClassifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6, 8]
    },
    "LGBMClassifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63]
    }
}

param_grids_reg = {
    "Ridge": {"alpha": [0.1, 1, 10]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1], "max_iter": [10000]},
    "ElasticNet": {
        "alpha": [0.01, 0.1, 1],
        "l1_ratio": [0.2, 0.5, 0.8],
        "max_iter": [10000]
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    },
    "RandomForestRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    },
    "GradientBoostingRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1]
    },
    "SVR": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "KNNRegressor": {"n_neighbors": [3, 5, 7]},
    "ExtraTreesRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    },
    "AdaBoostRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0]
    },
    "XGBRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6, 8]
    },
    "LGBMRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63]
    }
}

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
def tune_top_models(X_train, y_train, results, models, param_grids, preprocessor, scoring):

    sorted_models = sorted(results, key=lambda x: x[1], reverse=True)
    top_models = [m[0] for m in sorted_models if m[0] in param_grids][:2]

    tuned_results = []

    for name in top_models:

        model = models[name]

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # 🔥 param grid fix
        param_grid = {
            f"model__{k}": v for k, v in param_grids[name].items()
        }

        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        tuned_results.append({
            "Model": name,
            "Tuned Score": grid.best_score_,
            "Best Params": grid.best_params_,
            "Best Estimator": grid.best_estimator_
        })

    return tuned_results
def tune_top_models_cal(X_train, y_train, results, models, param_grids, preprocessor):
    return tune_top_models(
        X_train, y_train,
        results, models,
        param_grids,
        preprocessor,
        scoring="accuracy"
    )


def tune_top_models_reg(X_train, y_train, results, models, param_grids, preprocessor):
    if y_train.nunique() < 10:
        scoring = "accuracy"
    else:
        scoring = "r2"
    return tune_top_models(
        X_train, y_train,
        results, models,
        param_grids,
        preprocessor,
        scoring=scoring
    )
# -----------------------------
# EDA Report
# -----------------------------
def generate_EDA_report(df):
    missing_percentage = df.isnull().mean() * 100
    missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
    missing_df = pd.DataFrame({
        "Column": missing_percentage_sorted.index,
        "Missing_Percentage": missing_percentage_sorted.values
    })
    df_describe = df.describe().transpose()
    return missing_df, df_describe

# -----------------------------
# Skew Transformation
# -----------------------------
def transform_normally_distribuation(X_train, X_test):
    num_cols = X_train.select_dtypes(include=["number"]).columns
    skewed_cols = X_train[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
    skewed_cols = skewed_cols[abs(skewed_cols) > 0.5].index

    if len(skewed_cols) == 0:
        return X_train, X_test, skewed_cols

    # ── Filter out columns that will crash PowerTransformer ─────────────────
    safe_cols = []
    for col in skewed_cols:
        col_data = X_train[col].dropna()

        # Skip constant or near-constant columns
        if col_data.nunique() <= 1:
            continue
        if col_data.std() < 1e-6:
            continue

        # Skip columns where all values are identical after clipping
        q_low  = col_data.quantile(0.01)
        q_high = col_data.quantile(0.99)
        if q_low == q_high:
            continue

        # Try fitting PowerTransformer on just this column — catch any failure
        try:
            pt_test = PowerTransformer(method="yeo-johnson")
            pt_test.fit(col_data.values.reshape(-1, 1))
            safe_cols.append(col)
        except Exception:
            continue  # silently skip problematic columns

    if len(safe_cols) == 0:
        return X_train, X_test, pd.Index([])

    skewed_cols = pd.Index(safe_cols)
    # ────────────────────────────────────────────────────────────────────────

    pt = PowerTransformer(method="yeo-johnson")
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[skewed_cols] = pt.fit_transform(X_train[skewed_cols])
    X_test[skewed_cols]  = pt.transform(X_test[skewed_cols])

    return X_train, X_test, skewed_cols

import sys
import sklearn
import matplotlib
import seaborn
import joblib
from matplotlib.backends import backend_pdf

# Optional libraries safely
try:
    import xgboost
    xgb_version = xgboost.__version__
except:
    xgb_version = "Not Installed"

try:
    import lightgbm
    lgbm_version = lightgbm.__version__
except:
    lgbm_version = "Not Installed"

try:
    import category_encoders
    ce_version = category_encoders.__version__
except:
    ce_version = "Not Installed"

print("\n===== LIBRARY VERSIONS =====")
print("Python:", sys.version)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", seaborn.__version__)
print("joblib:", joblib.__version__)
print("xgboost:", xgb_version)
print("lightgbm:", lgbm_version)
print("category_encoders:", ce_version)
print("============================\n")