# app.py
import os
import time
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score, f1_score
)

# tensorflow import deferred until needed (slower to import)
# from tensorflow.keras.models import load_model

# ---------- Config / Paths ----------
XGB_MODEL_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\xplane_xgboost.pkl"
LSTM_MODEL_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\xplane_lstm.h5"
SCALER_PATH = r"C:\Users\T8630\Desktop\xplane_predictive_project\models\lstm_scaler.pkl"  # optional - if you saved one
DEFAULT_LSTM_TIMESTEPS = 50

st.set_page_config(page_title="XPlane Predictive Maintenance", layout="wide")

# ---------- Caching helpers ----------
@st.cache_resource
def load_xgb_model(path=XGB_MODEL_PATH):
    if not os.path.exists(path):
        return None, 0.5
    data = joblib.load(path)
    # support multiple save formats
    if isinstance(data, dict):
        model = data.get("model", data.get("model_object", None)) or data
        threshold = data.get("threshold", 0.5)
    elif isinstance(data, tuple) or isinstance(data, list):
        try:
            model, threshold = data[0], data[1]
        except Exception:
            model, threshold = data, 0.5
    else:
        model, threshold = data, 0.5

    # some sklearn wrappers have feature_names_in_:
    feature_names = None
    try:
        feature_names = list(getattr(model, "feature_names_in_", [])) or getattr(model, "get_booster", lambda: None)
        # if using the sklearn wrapper, feature_names_in_ should work
        if isinstance(feature_names, dict) or callable(feature_names):
            feature_names = None
    except Exception:
        feature_names = None

    return model, float(threshold)


@st.cache_resource
def load_lstm_model(path=LSTM_MODEL_PATH):
    if not os.path.exists(path):
        return None
    from tensorflow.keras.models import load_model
    model = load_model(path)
    return model


@st.cache_resource
def load_scaler(path=SCALER_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ---------- Helpers ----------
def clean_tabular_for_xgb(df, model=None):
    """Drop unnamed columns and keep only numeric. Align columns to model if possible."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # try to coerce object columns to numeric where possible
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
    df = df.select_dtypes(include=[np.number])
    if model is not None:
        # try to align to model expected features if available
        expected = None
        try:
            expected = list(model.feature_names_in_)
        except Exception:
            try:
                booster = model.get_booster()
                expected = getattr(booster, "feature_names", None)
            except Exception:
                expected = None
        if expected:
            # add missing columns with zeros, keep only expected order
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[expected]
    return df


def sliding_windows(X, timesteps=50):
    """Create sliding windows from 2D array X (n_samples x n_features)."""
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
    if len(Xs) == 0:
        return np.empty((0, timesteps, X.shape[1]))
    return np.stack(Xs, axis=0)


def plot_confusion(cm, labels=["0","1"], title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_roc(y_true, y_proba, label="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    ax.plot([0,1], [0,1], color="grey", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig, roc_auc


# ---------- UI ----------
st.title("✈️ X-Plane Predictive Maintenance — Interactive App")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload X-Plane CSV (processed features)", type=["csv"], help="CSV with columns including 'failure' (optional) and engine features.")
    sample_button = st.button("Load example sample (if available)")
    model_choice = st.selectbox("Model", ["XGBoost (tabular)", "LSTM (sequence)", "Both"], index=0)
    realtime_sim = st.checkbox("Simulate live stream (playback)", value=False)
    st.markdown("---")
    st.write("Model files found:")
    xgb_model_exists = os.path.exists(XGB_MODEL_PATH)
    lstm_model_exists = os.path.exists(LSTM_MODEL_PATH)
    st.write(f"- XGBoost: {'✅' if xgb_model_exists else '❌'} {XGB_MODEL_PATH}")
    st.write(f"- LSTM: {'✅' if lstm_model_exists else '❌'} {LSTM_MODEL_PATH}")
    st.markdown("---")
    st.write("App tips:")
    st.write("- Upload `xplane_features.csv` from `data/processed/` for best results.")
    st.write("- For LSTM choose reasonable timesteps (e.g., 30–100).")

# load models
xgb_model, saved_threshold = load_xgb_model()
lstm_model = load_lstm_model()
saved_scaler = load_scaler()

# Main area layout
col1, col2 = st.columns([2,1])

# Left column: data preview & visualization
with col1:
    st.subheader("Data")
    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded file — rows: {df.shape[0]}, columns: {df.shape[1]}")
        except Exception as e:
            st.error("Could not read CSV file: " + str(e))
    elif sample_button:
        # try to load processed features sample from project
        sample_path = r"C:\Users\T8630\Desktop\xplane_predictive_project\data\processed\xplane_features.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            st.success(f"Loaded sample: {sample_path} — rows: {df.shape[0]}")
        else:
            st.warning("No sample file found at data/processed/xplane_features.csv")

    if df is not None:
        st.dataframe(df.head(200))

        st.subheader("Plot signals")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        choose = st.multiselect("Select features to plot (time-series)", numeric_cols[:10], default=numeric_cols[:4] if len(numeric_cols)>=4 else numeric_cols)
        if choose:
            ts_index = None
            if "timestep" in df.columns:
                ts_index = df["timestep"]
            st.line_chart(df[choose])

        if realtime_sim:
            st.markdown("**Live playback controls**")
            speed = st.slider("Playback speed (delay sec per row)", 0.0, 0.5, 0.05, 0.01)
            run_button = st.button("Start playback")
            if run_button:
                placeholder = st.empty()
                try:
                    for i, _row in df.iterrows():
                        sub = placeholder.container()
                        sub.write(df.iloc[i:i+1])
                        time.sleep(speed)
                except Exception as e:
                    st.error("Playback stopped: " + str(e))

# Right column: modeling and inference
with col2:
    st.subheader("Inference & Model Controls")

    if model_choice in ("XGBoost (tabular)", "Both"):
        st.markdown("**XGBoost controls**")
        if xgb_model is None:
            st.warning("XGBoost model not found. Train and place model at: " + XGB_MODEL_PATH)
        else:
            st.write(f"Saved threshold: {saved_threshold:.2f}")
            threshold = st.slider("Classification threshold (XGBoost)", 0.0, 1.0, float(saved_threshold), 0.01)
            if st.button("Run XGBoost Inference"):
                try:
                    X_tab = clean_tabular_for_xgb(df, model=xgb_model) if df is not None else None
                    if X_tab is None or X_tab.shape[0] == 0:
                        st.error("No numeric features available for XGBoost inference.")
                    else:
                        # if scaler exists, use it
                        scaler = saved_scaler
                        if scaler is None:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_tab)
                        else:
                            X_scaled = scaler.transform(X_tab)

                        y_proba = xgb_model.predict_proba(X_scaled)[:, 1]
                        y_pred = (y_proba >= threshold).astype(int)

                        out = X_tab.copy()
                        out["failure_proba"] = y_proba
                        out["failure_pred"] = y_pred
                        st.success("Inference complete — preview below")
                        st.dataframe(out.head(50))

                        # if ground truth present
                        if "failure" in df.columns:
                            y_true = df["failure"].values[:len(y_pred)]
                            acc = accuracy_score(y_true, y_pred)
                            st.write(f"Accuracy: {acc:.4f}")
                            st.write("Precision:", precision_score(y_true, y_pred, zero_division=0))
                            st.write("Recall:", recall_score(y_true, y_pred, zero_division=0))
                            st.write("F1:", f1_score(y_true, y_pred, zero_division=0))
                            cm = confusion_matrix(y_true, y_pred)
                            st.pyplot(plot_confusion(cm, title="XGBoost Confusion Matrix"))
                            fig_roc, aucval = plot_roc(y_true, y_proba, label="XGBoost")
                            st.pyplot(fig_roc)
                        # download predictions
                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download predictions (CSV)", csv, file_name="xgb_predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error("XGBoost inference failed: " + str(e))

    if model_choice in ("LSTM (sequence)", "Both"):
        st.markdown("**LSTM controls**")
        if lstm_model is None:
            st.warning("LSTM model not found. Train and place model at: " + LSTM_MODEL_PATH)
        else:
            timesteps = st.number_input("LSTM window (timesteps)", min_value=5, max_value=500, value=DEFAULT_LSTM_TIMESTEPS, step=1)
            lstm_thresh = st.slider("LSTM classification threshold", 0.0, 1.0, 0.5, 0.01)

            if st.button("Run LSTM Inference"):
                if df is None:
                    st.error("❌ No data loaded. Please upload a CSV first.")
                else:
                    try:
                        # prepare numeric df
                        df_num = df.select_dtypes(include=[np.number]).copy()
                        if "failure" in df_num.columns:
                            failures = df_num["failure"].values
                            df_num = df_num.drop(columns=["failure"])
                        else:
                            failures = None

                        # scaling
                        scaler = saved_scaler
                        if scaler is None:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(df_num.values)
                        else:
                            X_scaled = scaler.transform(df_num.values)

                        # create sequences
                        X_seq = sliding_windows(X_scaled, timesteps=timesteps)
                        if X_seq.shape[0] == 0:
                            st.error("Not enough rows to create sequences with the chosen timesteps.")
                        else:
                            # predict (model expects shape (n, timesteps, features))
                            y_proba = lstm_model.predict(X_seq).ravel()
                            y_pred = (y_proba >= lstm_thresh).astype(int)

                            # align true labels (label at timestep index = i+timesteps)
                            if failures is not None:
                                y_true = failures[timesteps: timesteps + len(y_pred)]
                                acc = accuracy_score(y_true, y_pred)
                                st.write(f"Accuracy: {acc:.4f}")
                                st.write("Precision:", precision_score(y_true, y_pred, zero_division=0))
                                st.write("Recall:", recall_score(y_true, y_pred, zero_division=0))
                                st.write("F1:", f1_score(y_true, y_pred, zero_division=0))
                                cm = confusion_matrix(y_true, y_pred)
                                st.pyplot(plot_confusion(cm, title="LSTM Confusion Matrix"))
                                fig_roc, aucval = plot_roc(y_true, y_proba, label="LSTM")
                                st.pyplot(fig_roc)

                            out = pd.DataFrame({
                                "proba": y_proba,
                                "pred": y_pred
                            })
                            st.dataframe(out.head(100))
                            csv = out.to_csv(index=False).encode("utf-8")
                            st.download_button("Download LSTM predictions (CSV)", csv, file_name="lstm_predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error("LSTM inference failed: " + str(e))

st.markdown("---")
st.caption("Streamlit app for X-Plane predictive maintenance — interactive demo. Save models into the `models/` folder and processed CSV into `data/processed/`.")