# ============================================================
# Customer Churn Prediction App
# Dataset: Telco Customer Churn (Kaggle)
# Model: Random Forest Classifier
# UI: Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ChurnIQ | Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM UI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stSidebar"], button, input, select, textarea {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
h1, h2, h3, .result-title, .hero h1 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}
.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(99,102,241,.12), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(16,185,129,.10), transparent 25%),
        #0b1020;
    color: #f8fafc;
}
[data-testid="stHeader"] { background: rgba(11,16,32,.85); }
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,.08);
}
.hero {
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(30,41,59,.95), rgba(15,23,42,.82));
    border: 1px solid rgba(148,163,184,.18);
    box-shadow: 0 20px 50px rgba(0,0,0,.25);
    margin-bottom: 22px;
}
.hero h1 {
    margin: 0;
    font-size: 3.4rem;
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -0.03em;
}
.hero h1 .accent {
    background: linear-gradient(90deg, #818cf8 0%, #34d399 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}
.hero .tagline {
    margin: 12px 0 0;
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
}
.hero p { margin: 8px 0 0; color: #94a3b8; font-size: 1.02rem; max-width: 720px; }
.pill {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    background: rgba(99,102,241,.16);
    border: 1px solid rgba(129,140,248,.28);
    color: #c7d2fe;
    font-size: .78rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.result-high {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(127,29,29,.92), rgba(153,27,27,.70));
    border: 1px solid rgba(248,113,113,.35);
    text-align: center;
}
.result-low {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(6,78,59,.92), rgba(5,150,105,.65));
    border: 1px solid rgba(52,211,153,.30);
    text-align: center;
}
.result-title { font-size: 1.35rem; font-weight: 800; }
.result-sub { color: #d1d5db; font-size: .9rem; }
.small-note { color: #94a3b8; font-size: .82rem; }
div[data-testid="stMetric"] {
    background: rgba(15,23,42,.75);
    border: 1px solid rgba(148,163,184,.12);
    padding: 14px;
    border-radius: 15px;
}
.stButton > button { border-radius: 12px; font-weight: 700; min-height: 46px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_data():
    for filename in [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "Telco-Customer-Churn.csv",
    ]:
        try:
            return pd.read_csv(filename), filename
        except FileNotFoundError:
            pass
    return None, None


# ============================================================
# MODEL TRAINING
# ============================================================
@st.cache_resource
def train_model(df):
    data = df.copy()

    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["TotalCharges"].median()
    )

    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    encoders = {}

    categorical_columns = data.select_dtypes(
        include="object"
    ).columns

    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)

    return {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": X.columns.tolist(),
        "df": data,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "fpr": fpr_arr,
        "tpr": tpr_arr,
    }


# ============================================================
# LOAD DATA / UPLOAD FALLBACK
# ============================================================
df_raw, source_file = load_data()

if df_raw is None:
    st.markdown("""
    <div class="hero">
        <span class="pill">DATASET REQUIRED</span>
        <h1>📊 Churn<span class="accent">IQ</span></h1>
        <p>Customer churn prediction powered by Random Forest.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload the Telco Customer Churn CSV",
        type=["csv"]
    )

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        source_file = uploaded.name
    else:
        st.warning(
            "Put WA_Fn-UseC_-Telco-Customer-Churn.csv in the "
            "same folder as this app, or upload it above."
        )
        st.stop()

bundle = train_model(df_raw)

model = bundle["model"]
scaler = bundle["scaler"]
encoders = bundle["encoders"]
feature_names = bundle["feature_names"]
df = bundle["df"]
accuracy = bundle["accuracy"]
auc = bundle["auc"]
cm = bundle["cm"]
report = bundle["report"]
fpr = bundle["fpr"]
tpr = bundle["tpr"]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📊 ChurnIQ")
    st.caption("Customer retention intelligence")
    st.markdown("---")

    st.markdown("### 🌲 Model")
    st.write("Random Forest Classifier")
    st.write(f"{model.n_estimators} trees")
    st.write(f"Maximum depth: {model.max_depth}")
    st.write("Train/Test: 80% / 20%")

    st.markdown("---")
    st.markdown("### 📦 Dataset")
    st.write(f"{len(df):,} customers")
    st.write(f"{len(feature_names)} model features")

    st.markdown("---")
    st.markdown(
        '<div class="small-note">Python • Scikit-learn • Pandas • Plotly • Streamlit</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HEADER + KPIs
# ============================================================
st.markdown("""
<div class="hero">
    <span class="pill">MACHINE LEARNING • CUSTOMER RETENTION</span>
    <h1>📊 Churn<span class="accent">IQ</span></h1>
    <div class="tagline">Know who's leaving before they do.</div>
    <p>
        Predict customer churn risk, evaluate the Random Forest model,
        and explore the factors behind customer retention.
    </p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Features Used", f"{len(feature_names)}")
k3.metric("Accuracy", f"{accuracy * 100:.2f}%")
k4.metric("AUC-ROC", f"{auc:.3f}")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    ["🔮 Predict Churn", "📈 Model Performance", "📊 Data Insights"]
)


# ============================================================
# TAB 1 — PREDICTION
# ============================================================
with tab1:
    st.subheader("🔮 Customer Risk Assessment")
    st.caption(
        "Enter all customer attributes used by the trained model."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 👤 Customer Profile")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with c2:
        st.markdown("### 📱 Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    with c3:
        st.markdown("### 💳 Billing")
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        )
        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=65.0,
            step=1.0
        )
        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=800.0,
            step=10.0
        )

    st.markdown("---")

    if st.button(
        "🔮 Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):
        # Encode user input with the EXACT encoders learned during training.
        raw_values = {
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
        }

        encoded = {}

        # Gender is lowercase in the original dataset.
        if "gender" in encoders:
            encoded["gender"] = int(
                encoders["gender"].transform([gender])[0]
            )

        for col, value in raw_values.items():
            if col == "gender":
                continue
            if col in encoders:
                encoded[col] = int(
                    encoders[col].transform([value])[0]
                )

        values = {
            "gender": encoded["gender"],
            "SeniorCitizen": (
                int(encoders["SeniorCitizen"].transform(
                    ["Yes" if senior == "Yes" else "No"])[0])
                if "SeniorCitizen" in encoders
                else (1 if senior == "Yes" else 0)
            ),
            "Partner": encoded["Partner"],
            "Dependents": encoded["Dependents"],
            "tenure": tenure,
            "PhoneService": encoded["PhoneService"],
            "MultipleLines": encoded["MultipleLines"],
            "InternetService": encoded["InternetService"],
            "OnlineSecurity": encoded["OnlineSecurity"],
            "OnlineBackup": encoded["OnlineBackup"],
            "DeviceProtection": encoded["DeviceProtection"],
            "TechSupport": encoded["TechSupport"],
            "StreamingTV": encoded["StreamingTV"],
            "StreamingMovies": encoded["StreamingMovies"],
            "Contract": encoded["Contract"],
            "PaperlessBilling": encoded["PaperlessBilling"],
            "PaymentMethod": encoded["PaymentMethod"],
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        input_data = pd.DataFrame([values])

        # Force exactly the same feature order used during training.
        input_data = input_data[feature_names]

        input_scaled = scaler.transform(input_data)

        prediction = int(model.predict(input_scaled)[0])
        probability = float(
            model.predict_proba(input_scaled)[0][1]
        )
        retention_probability = 1 - probability

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        r1, r2, r3 = st.columns([1.1, 1.4, 1])

        with r1:
            if prediction == 1:
                st.markdown("""
                <div class="result-high">
                    <div class="result-title">⚠️ HIGH CHURN RISK</div>
                    <div class="result-sub">
                        Customer is predicted to churn.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-low">
                    <div class="result-title">✅ LOW CHURN RISK</div>
                    <div class="result-sub">
                        Customer is predicted to stay.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability * 100, 1),
                number={"suffix": "%"},
                title={"text": "Churn Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": "#ef4444"
                        if probability >= 0.5 else "#10b981"
                    },
                    "steps": [
                        {"range": [0, 50], "color": "#064e3b"},
                        {"range": [50, 100], "color": "#7f1d1d"},
                    ],
                }
            ))
            fig.update_layout(
                height=260,
                margin=dict(t=45, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f8fafc", "family": "Plus Jakarta Sans"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with r3:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.1f}%"
            )
            st.metric(
                "Retention Probability",
                f"{retention_probability * 100:.1f}%"
            )

        st.markdown("### 💡 Retention Actions")

        if prediction == 1:
            actions = []

            if contract == "Month-to-month":
                actions.append(
                    "Consider offering a longer-contract incentive."
                )
            if monthly > 80:
                actions.append(
                    "Review the customer's pricing plan."
                )
            if tenure < 12:
                actions.append(
                    "Provide additional onboarding/support."
                )
            if internet == "Fiber optic" and online_security == "No":
                actions.append(
                    "Consider offering a security add-on."
                )
            if not actions:
                actions.append(
                    "Prioritize this customer for a retention review."
                )

            for action in actions:
                st.warning("• " + action)
        else:
            st.success(
                "This customer is predicted to have a lower churn risk."
            )
            st.info(
                "Continue engagement and consider loyalty benefits."
            )


# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================
with tab2:
    st.subheader("📈 Random Forest Model Performance")
    st.caption("Metrics are calculated on the held-out test set.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    m2.metric("AUC-ROC", f"{auc:.3f}")
    m3.metric("Trees", f"{model.n_estimators}")
    m4.metric("Max Depth", f"{model.max_depth}")

    st.markdown("---")

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("### Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Not Churn", "Churn"],
            y=["Not Churn", "Churn"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig_cm.update_layout(
            template="plotly_dark",
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with p2:
        st.markdown("### ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Random Forest (AUC={auc:.3f})"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random baseline",
            line=dict(dash="dash")
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=390,
            template="plotly_dark"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Classification Report")
    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🌲 Feature Importance")

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig_imp = px.bar(
        feat_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Used by Random Forest"
    )
    fig_imp.update_layout(
        height=520,
        template="plotly_dark"
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================
# TAB 3 — DATA INSIGHTS
# ============================================================
with tab3:
    st.subheader("📊 Dataset Insights")

    total_customers = len(df)
    churned = int(df["Churn"].sum())
    not_churned = total_customers - churned
    churn_rate = churned / total_customers * 100

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Customers", f"{total_customers:,}")
    d2.metric("Churned Customers", f"{churned:,}")
    d3.metric("Overall Churn Rate", f"{churn_rate:.1f}%")

    st.markdown("---")

    a, b = st.columns(2)

    with a:
        st.markdown("### Churn Distribution")
        counts = df["Churn"].value_counts().sort_index()
        fig = px.pie(
            values=counts.values,
            names=["Not Churned", "Churned"],
            hole=0.55,
            title="Customer Churn Distribution"
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with b:
        st.markdown("### Tenure vs Churn")
        tenure_df = df.copy()
        tenure_df["Churn Label"] = tenure_df["Churn"].map(
            {0: "Not Churned", 1: "Churned"}
        )
        fig = px.histogram(
            tenure_df,
            x="tenure",
            color="Churn Label",
            nbins=24,
            barmode="overlay",
            opacity=0.75,
            title="Tenure Distribution by Churn"
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)

    with c:
        st.markdown("### Monthly Charges vs Churn")
        charge_df = df.copy()
        charge_df["Churn Label"] = charge_df["Churn"].map(
            {0: "Not Churned", 1: "Churned"}
        )
        fig = px.box(
            charge_df,
            x="Churn Label",
            y="MonthlyCharges",
            color="Churn Label",
            title="Monthly Charges by Churn Status"
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with d:
        st.markdown("### Contract Type vs Churn")
        contract_df = df.copy()
        contract_df["Contract"] = encoders["Contract"].inverse_transform(
            contract_df["Contract"]
        )
        contract_churn = (
            contract_df.groupby("Contract")["Churn"]
            .mean()
            .mul(100)
            .reset_index()
        )
        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Churn",
            text="Churn",
            title="Average Churn Rate by Contract"
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            yaxis_title="Churn Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button(
        "⬇️ Download Processed Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="processed_telco_churn.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("ℹ️ About This Project")

    x1, x2 = st.columns(2)

    with x1:
        st.markdown(f"""
        **Dataset:** Telco Customer Churn  
        **Task:** Binary classification  
        **Algorithm:** Random Forest Classifier  
        **Trees:** {model.n_estimators}  
        **Maximum depth:** {model.max_depth}  
        **Train/Test:** 80% / 20%
        """)

    with x2:
        st.markdown("""
        **Preprocessing**
        - Customer ID removed
        - `TotalCharges` converted to numeric
        - Missing `TotalCharges` handled
        - Categorical variables encoded
        - Features standardized

        **Evaluation**
        - Accuracy
        - AUC-ROC
        - Confusion Matrix
        - Classification Report
        - ROC Curve
        - Feature Importance
        """)

    st.caption("Source file: " + str(source_file))
