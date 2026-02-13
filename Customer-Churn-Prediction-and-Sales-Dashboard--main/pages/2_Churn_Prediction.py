import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Customer Churn Prediction — Identify + Explain")
st.caption("Train churn model, identify a customer, predict churn probability, and explain the reasons clearly.")

# -----------------------------
# Helpers
# -----------------------------
COMMON_TARGETS = [
    "churn","Churn","CHURN","Exited","exited","target","Target","label","Label",
    "is_churn","IsChurn","customer_churn"
]
COMMON_ID_COLS = ["CustomerID", "customer_id", "id", "ID", "CustID", "cust_id"]

def find_target_column(df: pd.DataFrame) -> str | None:
    for c in COMMON_TARGETS:
        if c in df.columns:
            return c
    # fallback: last binary column
    for c in df.columns[::-1]:
        u = df[c].dropna().unique()
        if len(u) <= 2:
            return c
    return None

def find_id_column(df: pd.DataFrame) -> str | None:
    for c in COMMON_ID_COLS:
        if c in df.columns:
            return c
    return None

def to_binary_target(y: pd.Series) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)
    if pd.api.types.is_numeric_dtype(y):
        return (y.astype(float) > 0).astype(int)
    y_str = y.astype(str).str.strip().str.lower()
    return y_str.isin(["1","yes","y","true","churn","exited"]).astype(int)

def safe_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def local_reason_table(model, x_row: pd.DataFrame, feature_names: list[str], top_k: int = 10):
    """
    Local explanation:
    - LogisticRegression: coef_ can show direction
    - RandomForest: use feature_importances_ as weights (no direction)
    """
    values = x_row.values.ravel()

    if hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        impact = values * coef
        df = pd.DataFrame({
            "feature": feature_names,
            "value": values,
            "coef": coef,
            "impact": impact,
            "abs_impact": np.abs(impact)
        }).sort_values("abs_impact", ascending=False).head(top_k)

        df["effect"] = np.where(df["impact"] > 0, "↑ increases churn risk", "↓ decreases churn risk")
        return df[["feature","value","coef","impact","effect"]]

    elif hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_)
        score = np.abs(values) * imp
        df = pd.DataFrame({
            "feature": feature_names,
            "value": values,
            "importance_weight": imp,
            "reason_score": score
        }).sort_values("reason_score", ascending=False).head(top_k)
        return df[["feature","value","importance_weight","reason_score"]]

    return pd.DataFrame()

def risk_bucket(prob: float):
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Medium"
    return "High"

def retention_actions(risk: str):
    if risk == "High":
        return [
            "Call / message customer personally",
            "Offer discount / loyalty plan upgrade",
            "Provide priority support",
            "Check payment issues / service complaints",
            "Share personalized offers (based on usage)"
        ]
    if risk == "Medium":
        return [
            "Send engagement email / app notification",
            "Recommend features / onboarding help",
            "Small coupon / bonus points",
            "Monitor usage weekly"
        ]
    return [
        "Keep regular engagement",
        "Upsell only if customer is active",
        "Collect feedback periodically"
    ]

# -----------------------------
# Require processed data
# -----------------------------
if "data_processor" not in st.session_state or st.session_state.data_processor.processed_features is None:
    st.error("❌ No processed data available. Please upload and preprocess your data first.")
    st.page_link("pages/1_Data_Upload.py", label="📁 Go to Data Upload", icon="📁")
    st.stop()

df = st.session_state.data_processor.processed_features.copy()
TARGET_COL = find_target_column(df)
ID_COL = find_id_column(df)

if TARGET_COL is None:
    st.error("❌ Could not detect churn target column.")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Keep a copy of original (for customer profile)
df_original = df.copy()

# Prepare X and y
y = to_binary_target(df[TARGET_COL])
X_all = df.drop(columns=[TARGET_COL])
X = safe_numeric_features(X_all)

if X.shape[1] == 0:
    st.error("❌ No numeric features found. Model needs numeric columns after preprocessing.")
    st.stop()

# -----------------------------
# Session state
# -----------------------------
if "churn_bundle" not in st.session_state:
    st.session_state.churn_bundle = {}

# -----------------------------
# Sidebar: train settings
# -----------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Model", ["Random Forest (Recommended)", "Logistic Regression"])
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

train_btn = st.sidebar.button("🚀 Train Model", type="primary")

if train_btn:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_choice.startswith("Random Forest"):
        model = RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        )
    else:
        model = LogisticRegression(max_iter=3000, class_weight="balanced")

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "model_name": model_choice,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Global permutation importance (explain model)
    with st.spinner("Calculating feature importance..."):
        try:
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=random_state, n_jobs=-1)
            global_imp = pd.DataFrame({
                "feature": X.columns,
                "importance": perm.importances_mean
            }).sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            global_imp = None

    st.session_state.churn_bundle = {
        "model": model,
        "metrics": metrics,
        "features": list(X.columns),
        "global_importance": global_imp
    }

    st.success("✅ Model trained and explanation generated!")
    st.rerun()

# -----------------------------
# Need trained model
# -----------------------------
if not st.session_state.churn_bundle:
    st.warning("Train a model from the sidebar to start predictions.")
    st.stop()

bundle = st.session_state.churn_bundle
model = bundle["model"]
metrics = bundle["metrics"]
feature_names = bundle["features"]
global_imp = bundle["global_importance"]

# -----------------------------
# Display performance
# -----------------------------
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", metrics["model_name"])
m2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m3.metric("Precision", f"{metrics['precision']:.3f}")
m4.metric("F1 Score", f"{metrics['f1']:.3f}")

# -----------------------------
# Identify customer section
# -----------------------------
st.markdown("---")
st.header("🧾 Identify a Customer to Predict Churn")

left, right = st.columns([1.2, 1])

with left:
    if ID_COL:
        st.subheader("Search by Customer ID")
        entered_id = st.text_input(f"Enter {ID_COL}")
        if entered_id.strip():
            matches = df_original[df_original[ID_COL].astype(str) == entered_id.strip()]
            if len(matches) == 0:
                st.warning("No customer found with that ID.")
                customer_row_index = None
            else:
                customer_row_index = int(matches.index[0])
                st.success(f"Customer found at row index: {customer_row_index}")
        else:
            customer_row_index = None
    else:
        st.info("")
        customer_row_index = None

    st.subheader("Or select by Row Index")
    idx = st.number_input("Customer Row Index", 0, int(len(X) - 1), 0, 1)

    if customer_row_index is None:
        customer_row_index = int(idx)

with right:
    st.subheader("Customer Profile (Raw Data)")
    profile = df_original.loc[customer_row_index]
    # show only a small set (avoid huge wide tables)
    st.dataframe(profile.to_frame("value").head(25), use_container_width=True)

# -----------------------------
# Predict + Explain
# -----------------------------
st.markdown("---")
st.header("🎯 Churn Prediction Result")

row = X.iloc[[customer_row_index]]
prob = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None
pred = int(model.predict(row)[0])
pred_label = "⚠️ Churn" if pred == 1 else "✅ Not Churn"
risk = risk_bucket(prob) if prob is not None else "Unknown"

c1, c2, c3 = st.columns(3)
c1.metric("Prediction", pred_label)
if prob is not None:
    c2.metric("Churn Probability", f"{prob:.3f}")
    c3.metric("Risk Level", risk)
else:
    c2.info("Probability not supported by this model.")
    c3.empty()

# Explanation
st.subheader("🔍 Explanation: Top Reasons for This Customer")

reasons = local_reason_table(model, row, feature_names, top_k=10)
if reasons.empty:
    st.info("Explanation not available for this model.")
else:
    st.dataframe(reasons.round(4), use_container_width=True)

# Actions
st.subheader("💡 What should we do? (Retention Suggestions)")
for act in retention_actions(risk):
    st.write(f"• {act}")

# -----------------------------
# Global importance (model-wide)
# -----------------------------
st.markdown("---")
st.header("🌍 Model-wide Important Features (Global)")

if global_imp is None:
    st.info("Global importance could not be calculated (try training again).")
else:
    st.dataframe(global_imp.head(20).round(6), use_container_width=True)

# -----------------------------
# Batch Predictions + Export
# -----------------------------
st.markdown("---")
st.header("📦 Batch Predictions (All Customers)")

with st.spinner("Generating predictions for all customers..."):
    all_pred = model.predict(X).astype(int)
    all_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.full(len(X), np.nan)

pred_df = pd.DataFrame({
    "customer_index": X.index,
    "churn_prediction": all_pred,
    "churn_probability": all_prob
}).sort_values("churn_probability", ascending=False)

st.dataframe(pred_df.head(50).round(4), use_container_width=True)

csv_data = pred_df.to_csv(index=False)
st.download_button(
    "📥 Download All Predictions (CSV)",
    data=csv_data,
    file_name="churn_predictions.csv",
    mime="text/csv"
)
