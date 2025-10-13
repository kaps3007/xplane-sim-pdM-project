# ======================================================
# ✈️ X-PLANE PREDICTIVE MAINTENANCE STREAMLIT APP (Unified + Enhanced)
# ======================================================
import os
import time
from datetime import datetime
import io
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

# ---------- CONFIG / PATHS ----------
XGB_MODEL_PATH = r"models\xplane_xgboost.pkl"
LSTM_MODEL_PATH = r"models\xplane_lstm.h5"
SCALER_PATH = r"models\lstm_scaler.pkl"
DATA_PATH = r"data\processed\xplane_features.csv"
DEFAULT_LSTM_TIMESTEPS = 50
LOG_OUT_PATH = r"data\live_log.csv"
df = pd.read_csv(DATA_PATH)
p1_max = df["power_1hp"].max()
p2_max = df["power_2hp"].max()
t1_max = df["thrst_1lb"].max()
t2_max = df["thrst_2lb"].max()
r1_max = df["rpm_1engin"].max()
r2_max = df["rpm_2engin"].max()
e1_max = df["EGT_1__deg"].max()
e2_max = df["EGT_2__deg"].max()
ot1_max = df["OILT1__deg"].max()
ot2_max = df["OILT2__deg"].max()
fp1_max = df["FUEP1__psi"].max()
fp2_max = df["FUEP2__psi"].max()


# ---------- APP CONFIG ----------
st.set_page_config(page_title="✈️ X-Plane Predictive Maintenance", layout="wide")
# ---- Auto-hide Sidebar Script ----
st.markdown("""
    <style>
        /* Sidebar transition and visibility settings */
        [data-testid="stSidebar"] {
            transition: all 0.6s ease-in-out;
        }
        .sidebar-hidden [data-testid="stSidebar"] {
            margin-left: -330px; /* hides the sidebar */
        }
        .sidebar-hidden [data-testid="stSidebarNav"] {
            opacity: 0;
        }
    </style>

    <script>
        let sidebarTimer;
        let root = window.parent.document.querySelector('[data-testid="stAppViewContainer"]').parentElement;

        function hideSidebar() {
            if (!root.classList.contains('sidebar-hidden')) {
                root.classList.add('sidebar-hidden');
            }
        }

        function showSidebar() {
            if (root.classList.contains('sidebar-hidden')) {
                root.classList.remove('sidebar-hidden');
            }
            clearTimeout(sidebarTimer);
            sidebarTimer = setTimeout(hideSidebar, 10000); // 10 seconds of inactivity
        }

        // Reset timer on any mouse movement
        window.addEventListener('mousemove', showSidebar);
        window.addEventListener('mousedown', showSidebar);
        window.addEventListener('scroll', showSidebar);
        showSidebar();  // start timer immediately
    </script>
""", unsafe_allow_html=True)

# ---------- CACHED HELPERS ----------
@st.cache_resource
def load_xgb_model(path=XGB_MODEL_PATH):
    if not os.path.exists(path):
        return None, 0.5
    data = joblib.load(path)
    if isinstance(data, dict):
        model = data.get("model", data.get("model_object", None)) or data
        threshold = data.get("threshold", 0.5)
    elif isinstance(data, (tuple, list)):
        model, threshold = data[0], data[1] if len(data) > 1 else 0.5
    else:
        model, threshold = data, 0.5
    return model, float(threshold)

@st.cache_resource
def load_lstm_model(path=LSTM_MODEL_PATH):
    if not os.path.exists(path):
        return None
    return load_model(path)

@st.cache_resource
def load_scaler(path=SCALER_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ---------- UTILITIES ----------
def live_stream(file_path=DATA_PATH):
    if not os.path.exists(file_path):
        return
    for row in pd.read_csv(file_path, chunksize=1):
        yield row

def clean_features_for_model(row_df, drop_cols=("failure",)):
    df = row_df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.select_dtypes(include=[np.number])

def sliding_windows(X, timesteps=50):
    if X is None or len(X) == 0:
        return np.empty((0, timesteps, 0))
    Xs = [X[i:i+timesteps] for i in range(len(X)-timesteps)]
    return np.stack(Xs, axis=0) if Xs else np.empty((0, timesteps, X.shape[1]))

def plot_confusion(cm, labels=["0", "1"], title="Confusion Matrix"):
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
            ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_roc(y_true, y_proba, label="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1], color="grey", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig, roc_auc

def identify_top_contributors(xgb_model, scaler, features_df_or_dict, top_k=3):
    """
    features_df_or_dict can be a single-row DataFrame or a dict mapping feature->value.
    Returns list of top contributors or None.
    """
    if xgb_model is None or scaler is None:
        return None
    try:
        feat_names = list(xgb_model.feature_names_in_)
    except Exception:
        feat_names = None
    if feat_names is None:
        return None
    importances = getattr(xgb_model, "feature_importances_", np.ones(len(feat_names)))
    mean, scale = getattr(scaler, "mean_", None), getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        return None
    # build row_vals
    if isinstance(features_df_or_dict, pd.DataFrame):
        row_map = features_df_or_dict.iloc[0].to_dict()
    else:
        row_map = dict(features_df_or_dict)
    row_vals = np.array([float(row_map.get(col, 0)) for col in feat_names])
    z = (row_vals - mean) / np.where(scale==0, 1e-6, scale)
    scores = np.abs(z) * np.abs(importances)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"feature": feat_names[i], "value": row_vals[i], "score": scores[i]} for i in top_idx]

# ---------- UI ----------
st.title("✈️ X-Plane Predictive Maintenance Dashboard")
with st.sidebar.expander("📘 About This Dashboard"):
    st.markdown("""
    ### ✈️ X-Plane Predictive Maintenance Dashboard
    This dashboard simulates **real-time engine health monitoring** for aircraft systems using live data from X-Plane 11.

    #### 🧩 Parameters:
    - **RPM**: Engine revolutions per minute — reflects power output.
    - **N1 / N2**: Turbine speeds (low & high pressure turbine speed).
    - **EGT**: Exhaust Gas Temperature — a key early failure indicator.
    - **Oil Temp / Pressure**: Critical for lubrication and cooling.
    - **Fuel Pressure**: Indicates consistent flow; sudden drops can hint at pump or line faults.

    #### 🎯 Failure Probability Threshold Meter (default values, can be modified from the slider below):
    - 🟢 0.00 – 0.50 → Stable (Engine healthy)
    - 🟡 0.51 – 0.70 → Low Risk (Potential warning signs)
    - 🔴 0.71 – 1.00 → High Risk (Immediate inspection advised)
    - **Note:** Thresholds can be adjusted based on operational preferences.

    #### 🕹️ Telemetry Simulation (for common people) --NEW--:
    - Adjust key engine parameters (throttle, RPM, EGT, oil temp/pressure) to see real-time impact on failure probability.
    - Option to sync 2nd engine parameters with the 1st for simplified testing.

    #### 💡 Powered by:
    - **XGBoost** (for static feature-based health scoring)
    - **LSTM (Long Short Term Memory Neural Network)** (for temporal failure prediction)

    **Goal:** Predict failures before they happen — transforming maintenance from Reactive to Predictive.
    """)

st.sidebar.header("Mode Selection")
# Added new simulation mode item to the list
mode = st.sidebar.radio("Choose mode", ["📡 Real-Time Streaming", "📊 Interactive Batch Analysis", "🎮 What-If Simulation"])

xgb_model, saved_threshold = load_xgb_model()
lstm_model = load_lstm_model()
scaler = load_scaler()

# Create a reusable placeholder for the gauge
gauge_ph = st.empty()

def render_gauge(prob, g_thresh, y_thresh):
    """Animated cinematic gauge with glowing background + 3s red-zone alarm."""
    prob = float(np.clip(prob, 0.0, 1.0))

    # Initialize alarm state if not exists
    if "alarm_triggered" not in st.session_state:
        st.session_state.alarm_triggered = False

    # Determine zone colors + glow intensity
    if prob <= g_thresh:
        bar_color = "#15FF00"       # bright green
        bg_color = "rgba(0, 200, 0, 0.5)"
        pulse_strength = 0.1
        st.session_state.alarm_triggered = False  # Reset alarm
    elif prob <= y_thresh:
        bar_color = "#FFD700"       # amber
        bg_color = "rgba(255, 215, 0, 0.25)"
        pulse_strength = 0.3
        st.session_state.alarm_triggered = False  # Reset alarm
    else:
        bar_color = "#FF4C4C"       # red
        bg_color = "rgba(255, 0, 0, 0.3)"
        pulse_strength = 0.6

        # 🚨 Trigger alarm only once per red-zone entry
        if not st.session_state.alarm_triggered:
            st.session_state.alarm_triggered = True
            st.markdown(
                """
                <script>
                setTimeout(() => {
                    const a = document.getElementById('alert-sound');
                    if (a) { a.pause(); a.currentTime = 0; a.remove(); }
                }, 3000);
                </script>
                """,
                unsafe_allow_html=True
            )

    # Background pulse animation
    pulse_phase = (time.time() * 2.5) % (2 * np.pi)
    pulse_alpha = 0.25 + pulse_strength * (0.5 + 0.5 * np.sin(pulse_phase))
    glow_rgba = f"rgba(255, 0, 0, {pulse_alpha:.2f})" if prob > y_thresh else bg_color

    # Build the gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={'font': {'color': 'white', 'size': 44}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Failure Probability", 'font': {'size': 22, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': bar_color, 'thickness': 0.35},
            'borderwidth': 3,
            'bordercolor': "#000000",
            'steps': [
                {'range': [0, g_thresh], 'color': '#003300'},
                {'range': [g_thresh, y_thresh], 'color': '#705900'},
                {'range': [y_thresh, 1.0], 'color': '#4D0000'}
            ],
            'threshold': {
                'line': {'color': "#000000", 'width': 5},
                'thickness': 0.8,
                'value': prob
            }
        }
    ))

    # Set layout with dynamic glow background
    fig.update_layout(
        height=360,
        margin=dict(t=60, b=40, l=40, r=40),
        paper_bgcolor=glow_rgba,
        plot_bgcolor="#0E1117",
        font={'color': 'white'},
        transition={'duration': 500, 'easing': 'cubic-in-out'}
    )

    gauge_ph.plotly_chart(fig, use_container_width=True, key=f"gauge_{time.time_ns()}")


def zone_label(prob, g_thresh, y_thresh):
    if prob <= g_thresh:
        return "🟢 STABLE","green","Engine is operating normally! 😊"
    elif prob <= y_thresh:
        return "🟡 LOW RISK","gold","Model has detected minor anomalies!"
    else:
        st.markdown("""
        <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
        </audio>
    """, unsafe_allow_html=True)
        return "🔴 HIGH RISK","red","Potential failure detected! Consider replacing the part before failure!"

# ---------- REAL-TIME STREAMING ----------
if mode == "📡 Real-Time Streaming":
    st.subheader("📡 Real-Time Predictive Maintenance Dashboard")

    st.sidebar.subheader("🔧 Stream Controls")
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 0.5, 10.0, 1.0, 0.5)
    start_stream = st.sidebar.button("▶ Start Live Streaming")
    stop_stream = st.sidebar.button("■ Stop Live Streaming")

    st.sidebar.subheader("🎯 Risk Zone Thresholds")
    green_threshold = st.sidebar.slider("🟢 Green Zone", 0.0, 1.0, 0.5, 0.01)
    yellow_threshold = st.sidebar.slider("🟡 Yellow Zone", green_threshold, 1.0, 0.75, 0.01)

    # Logging Controls
    st.sidebar.subheader("📥 Logging")
    if "live_log_df" not in st.session_state:
        st.session_state.live_log_df = pd.DataFrame(columns=["timestamp","xgb_prob","lstm_prob","combined_prob","zone"])
    log_button = st.sidebar.button("Toggle Logging")
    if log_button:
        st.session_state["log_enabled"] = not st.session_state.get("log_enabled", False)
        st.success("Logging Enabled" if st.session_state["log_enabled"] else "Logging Disabled")

    # Layout
    col_left, col_right = st.columns([2,1])
    with col_left:
        gauge_ph = st.empty()
        chart_xgb = st.line_chart(pd.DataFrame(columns=["xgb_prob"]))
        chart_lstm = st.line_chart(pd.DataFrame(columns=["lstm_prob"]))
    with col_right:
        status_area = st.empty()
        faulty_area = st.empty()

    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False

    if start_stream:
        st.session_state.stream_running = True
    if stop_stream:
        st.session_state.stream_running = False

    if st.session_state.stream_running:
        last_prob = 0.0
        for row in live_stream():
            if not st.session_state.stream_running:
                break
            features = clean_features_for_model(row)
            try:
                xgb_prob = float(xgb_model.predict_proba(features)[0][1]) if xgb_model is not None else 0.0
            except Exception:
                xgb_prob = 0.0
            try:
                scaled = scaler.transform(features) if scaler is not None else np.zeros_like(features)
                if "seq_buf" not in st.session_state:
                    st.session_state.seq_buf = []
                st.session_state.seq_buf.append(scaled.flatten())
                if len(st.session_state.seq_buf) >= DEFAULT_LSTM_TIMESTEPS and lstm_model is not None:
                    arr = np.array(st.session_state.seq_buf[-DEFAULT_LSTM_TIMESTEPS:]).reshape(1,DEFAULT_LSTM_TIMESTEPS,features.shape[1])
                    lstm_prob = float(lstm_model.predict(arr, verbose=0)[0][0])
                else:
                    lstm_prob = 0.0
            except Exception:
                lstm_prob = 0.0

            combined = xgb_prob + lstm_prob
            smooth = last_prob + (combined - last_prob)
            last_prob = smooth

            render_gauge(smooth, green_threshold, yellow_threshold)
            chart_xgb.add_rows(pd.DataFrame({"xgb_prob":[xgb_prob]}))
            chart_lstm.add_rows(pd.DataFrame({"lstm_prob":[lstm_prob]}))

            zone_txt, color, desc = zone_label(smooth, green_threshold, yellow_threshold)
            try:
                engine_rpm = float(row['rpm_1engin'].values[0]) if 'rpm_1engin' in row.columns else 0.0
                n1 = float(row['N1__1_pcnt'].values[0]) if 'N1__1_pcnt' in row.columns else 0.0
                n2 = float(row['N1__2_pcnt'].values[0]) if 'N1__2_pcnt' in row.columns else 0.0
                oil_temp1 = float(row['OILT1__deg'].values[0]) if 'OILT1__deg' in row.columns else 0.0
                oil_temp2 = float(row['OILT2__deg'].values[0]) if 'OILT2__deg' in row.columns else 0.0
                egt1 = float(row['EGT_1__deg'].values[0]) if 'EGT_1__deg' in row.columns else 0.0
                egt2 = float(row['EGT_2__deg'].values[0]) if 'EGT_2__deg' in row.columns else 0.0
                fuel_pressure = float(row['FUEP1__psi'].values[0]) if 'FUEP1__psi' in row.columns else 0.0
            except Exception:
                engine_rpm = n1 = n2 = oil_temp1 = oil_temp2 = egt1 = egt2 = fuel_pressure = 0.0
            status_area.markdown(f"""
                <div style="padding:13px;border-radius:12px;background:rgba(255,255,255,0.05);
                    border-left:6px solid {color};box-shadow:0 0 25px {color}80;">
                <h3 style="margin:0;color:{color};font-size:22px">{zone_txt}</h3>
                <p style="margin:4px 0;font-size:16px;color:white">{desc}</p>
                <p style="margin:4px 0;color:lightgray">
                Combined Probability: <b style="color:{color}">{smooth:.3f}</b></p>

                <hr style="border:1px solid rgba(255,255,255,0.1)">
                <h4 style="color:white;margin-bottom:4px;">Telemetry Data</h4>
                <ul style="list-style:none;padding-left:8px;color:#dcdcdc;font-size:15px;line-height:1.5;">
                    <li><b>Engine RPM:</b> {engine_rpm:.2f}</li>
                    <li><b>N1:</b> {n1:.2f}% | <b>N2:</b> {n2:.2f}%</li>
                    <li><b>Oil Temp (Eng 1):</b> {oil_temp1:.2f} °C | <b>Oil Temp (Eng 2):</b> {oil_temp2:.2f} °C</li>
                    <li><b>EGT (Eng 1):</b> {egt1:.2f} °C | <b>EGT (Eng 2):</b> {egt2:.2f} °C</li>
                    <li><b>Fuel Pressure:</b> {fuel_pressure:.2f} psi</li>
                    <li><b>Failure Probability (XGBoost – Top Graph):</b> {xgb_prob:.2f}</li>
                    <li><b>Failure Probability (LSTM – Bottom Graph):</b> {lstm_prob:.2f}</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)

            if st.session_state.get("log_enabled", False):
                st.session_state.live_log_df = pd.concat([st.session_state.live_log_df, pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(),
                    "xgb_prob": xgb_prob,
                    "lstm_prob": lstm_prob,
                    "combined_prob": smooth,
                    "zone": zone_txt
                }])], ignore_index=True)
                st.session_state.live_log_df.tail(1).to_csv(LOG_OUT_PATH, mode="a", header=not os.path.exists(LOG_OUT_PATH), index=False)

            time.sleep(refresh_rate)
        st.info("Stream stopped.")
    
# ---------- BATCH ANALYSIS ----------
if mode == "📊 Interactive Batch Analysis":
    st.title("📊 Interactive Batch Analysis")
    uploaded = st.file_uploader("Upload processed X-Plane CSV", type=["csv"])
    model_choice = st.selectbox("Select Model", ["XGBoost", "LSTM", "Both"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        st.dataframe(df.head())
        if model_choice in ("XGBoost","Both"):
            st.subheader("XGBoost Analysis")
            X = clean_features_for_model(df)
            y = df["failure"] if "failure" in df.columns else None
            try:
                proba = xgb_model.predict_proba(X)[:,1] if xgb_model is not None else np.zeros(len(X))
            except Exception:
                proba = np.zeros(len(X))
            preds = (proba>=0.5).astype(int)
            if y is not None:
                cm = confusion_matrix(y, preds)
                st.pyplot(plot_confusion(cm,["NoFail","Fail"],"XGB Confusion"))
                fig_roc, aucv = plot_roc(y, proba, "XGBoost")
                st.pyplot(fig_roc)
                st.success(f"ROC-AUC: {aucv:.3f}")
        if model_choice in ("LSTM","Both"):
            st.subheader("LSTM Analysis")
            df_num = df.select_dtypes(include=[np.number])
            y = df["failure"] if "failure" in df.columns else None
            expected_features = getattr(scaler, "feature_names_in_", None) if scaler is not None else None
            if expected_features is not None:
                # Add missing columns
                for col in expected_features:
                    if col not in df_num.columns:
                        df_num[col] = 0.0
                # Drop extra columns not seen during training
                df_num = df_num[expected_features]
            else:
                # fallback: ensure consistent number of columns
                if scaler is not None and getattr(scaler, "mean_", None) is not None:
                    df_num = df_num.iloc[:, :scaler.mean_.shape[0]]
            X_scaled = scaler.transform(df_num) if scaler is not None else df_num.values
            X_seq = sliding_windows(X_scaled, DEFAULT_LSTM_TIMESTEPS)
            try:
                proba = lstm_model.predict(X_seq).ravel() if lstm_model is not None and X_seq.size else np.array([])
            except Exception:
                proba = np.array([])
            preds = (proba>=0.5).astype(int) if len(proba) else np.array([])
            if y is not None and len(proba):
                y_true = y[DEFAULT_LSTM_TIMESTEPS:DEFAULT_LSTM_TIMESTEPS+len(preds)]
                cm = confusion_matrix(y_true, preds)
                st.pyplot(plot_confusion(cm,["NoFail","Fail"],"LSTM Confusion"))
                fig_roc, aucv = plot_roc(y_true, proba, "LSTM")
                st.pyplot(fig_roc)
                st.success(f"ROC-AUC: {aucv:.3f}")

# ---------- SIMULATION / WHAT-IF MODE ----------
elif mode == "🎮 What-If Simulation":
    st.title("🎮 What-If Simulation Mode — Play with key parameters")
    st.markdown("Adjust the key engine and flight parameters below — background telemetry values remain constant.")

    # --- Determine feature names (prefer scaler) ---
    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is None and os.path.exists(DATA_PATH):
        try:
            sample_df = pd.read_csv(DATA_PATH, nrows=100).select_dtypes(include=[np.number])
            feature_names = list(sample_df.columns)
        except Exception:
            feature_names = []
    if not feature_names:
        feature_names = [
            "power_1hp", "power_2hp", "thrst_1lb", "thrst_2lb",
            "rpm_1engin", "rpm_2engin",
            "N1__1_pcnt", "N1__2_pcnt", "N2__1_pcnt", "N2__2_pcnt",
            "EGT_1__deg", "EGT_2__deg",
            "OILT1__deg", "OILT2__deg",
            "FUEP1__psi", "FUEP2__psi"
        ]

    # --- Constant features (auto-fixed) ---
    constant_features = [
        "fact_sec", "fsim_sec", "frame_time", "__cpu_time", "_gpu_time_",
        "_grndratio", "_flitratio", "_Vind_kias", "_Vind_keas",
        "Vtrue_ktas", "Vtrue_ktgs", "_Vind__mph", "Vtruemphas", "Vtruemphgs"
    ]

    # --- Sidebar controls ---
    st.sidebar.markdown("### 🧭 Simulation Controls")
    sim_g_threshold = st.sidebar.slider("🟢 Green Threshold", 0.0, 1.0, 0.5, 0.01)
    sim_y_threshold = st.sidebar.slider("🟡 Yellow Threshold", sim_g_threshold, 1.0, 0.75, 0.01)
    sim_randomize = st.sidebar.button("🔀 Randomize Values")
    sim_reset = st.sidebar.button("♻️ Reset to Defaults")
    sim_apply_noise = st.sidebar.checkbox("Auto-noise (oscillate values)", value=False)
    sim_noise_amp = st.sidebar.slider("Noise amplitude", 0.0, 1.0, 0.05, 0.01)

    # --- Prepare defaults and sample stats ---
    sample_stats = {}
    if os.path.exists(DATA_PATH):
        try:
            df_sample = pd.read_csv(DATA_PATH, nrows=500).select_dtypes(include=[np.number])
            for fn in feature_names:
                if fn in df_sample.columns:
                    col = df_sample[fn].dropna()
                    if not col.empty:
                        sample_stats[fn] = {
                            "min": float(col.quantile(0.01)),
                            "max": float(col.quantile(0.99)),
                            "mean": float(col.mean())
                        }
        except Exception:
            sample_stats = {}

    if "sim_values" not in st.session_state:
        st.session_state.sim_values = {
            fn: sample_stats.get(fn, {"mean": 0.0})["mean"] for fn in feature_names
        }

    for fn in constant_features:
        if fn not in st.session_state.sim_values:
            st.session_state.sim_values[fn] = 0.0

    # --- Engine sync checkbox ---
    st.markdown("### ⚙️ Engine Pairing Options")
    keep_constant = st.checkbox("Keep 2nd engine values constant (mirror engine 1)?", value=True)

    # --- Editable parameters ---
    st.markdown("### ✈️ Adjustable Parameters")
    sim_cols = st.columns(2)
    idx = 0
    for fn in feature_names:
        # Skip 2nd engine values if syncing is on
        if keep_constant and any(fn.endswith(suffix) for suffix in ["_2hp", "_2lb", "2engin", "__2_pcnt", "_2__deg", "2__psi"]):
            continue

        stats = sample_stats.get(fn, {"mean": 0.0})
        default = st.session_state.sim_values.get(fn, stats["mean"])
        step = 0.01

        with sim_cols[idx % 2]:
            val = st.number_input(fn, value=float(default), step=step, key=f"sim_{fn}")
            st.session_state.sim_values[fn] = val
        idx += 1

    # --- Mirror values automatically if checkbox enabled ---
    if keep_constant:
        mirror_pairs = [
            ("power_1hp", "power_2hp"),
            ("thrst_1lb", "thrst_2lb"),
            ("rpm_1engin", "rpm_2engin"),
            ("N1__1_pcnt", "N1__2_pcnt"),
            ("N2__1_pcnt", "N2__2_pcnt"),
            ("EGT_1__deg", "EGT_2__deg"),
            ("OILT1__deg", "OILT2__deg"),
            ("FUEP1__psi", "FUEP2__psi"),
        ]
        for a, b in mirror_pairs:
            if a in st.session_state.sim_values:
                st.session_state.sim_values[b] = st.session_state.sim_values[a]

    # --- Randomize / Reset ---
    if sim_randomize:
        for fn in feature_names:
            if fn in constant_features:
                continue
            stats = sample_stats.get(fn, {"min": 0.0, "max": 1.0})
            st.session_state.sim_values[fn] = float(np.random.uniform(stats["min"], stats["max"]))
        st.rerun()

    if sim_reset:
        for fn in feature_names:
            stats = sample_stats.get(fn, {"mean": 0.0})
            st.session_state.sim_values[fn] = stats.get("mean", 0.0)
        st.rerun()

    # --- Noise effect (optional) ---
    if sim_apply_noise:
        for fn in feature_names:
            base = st.session_state.sim_values[fn]
            noise = sim_noise_amp * np.sin(time.time() * (0.5 + (hash(fn) % 7)))
            st.session_state.sim_values[fn] = float(base + noise)

    # --- Combine all values into DataFrame ---
    all_features = feature_names + constant_features
    sim_df = pd.DataFrame([{k: st.session_state.sim_values.get(k, 0.0) for k in all_features}])

    with st.expander("🔎 Current Simulation Input"):
        st.table(sim_df.T.rename(columns={0: "value"}))

    # --- Model Predictions ---
    try:
        X_for_xgb = clean_features_for_model(sim_df)
        xgb_prob = float(xgb_model.predict_proba(X_for_xgb)[0][1]) if xgb_model else 0.0
    except Exception:
        xgb_prob = 0.0

    try:
        X_num = sim_df.select_dtypes(include=[np.number])
        if scaler is not None:
            expected = getattr(scaler, "feature_names_in_", None)
            if expected is not None:
                for col in expected:
                    if col not in X_num.columns:
                        X_num[col] = 0.0
                X_num = X_num[expected]
        scaled_row = scaler.transform(X_num) if scaler else X_num.values
        if "sim_seq_buf" not in st.session_state:
            st.session_state.sim_seq_buf = []
        st.session_state.sim_seq_buf.append(scaled_row.flatten())
        st.session_state.sim_seq_buf = st.session_state.sim_seq_buf[-DEFAULT_LSTM_TIMESTEPS:]
        lstm_prob = float(lstm_model.predict(
            np.array(st.session_state.sim_seq_buf[-DEFAULT_LSTM_TIMESTEPS:])
            .reshape(1, DEFAULT_LSTM_TIMESTEPS, -1), verbose=0)[0][0]
        ) if (len(st.session_state.sim_seq_buf) >= DEFAULT_LSTM_TIMESTEPS and lstm_model) else 0.0
    except Exception:
        lstm_prob = 0.0

    combined_sim = xgb_prob + lstm_prob

    # --- Sidebar gauge ---
    with st.sidebar:
        st.title("🎯 Failure Probability")
        render_gauge(combined_sim, sim_g_threshold, sim_y_threshold)
        st.title(combined_sim)

    # --- Metrics ---
    st.metric("XGBoost Prob", f"{xgb_prob:.3f}")
    st.metric("LSTM Prob", f"{lstm_prob:.3f}")
    st.metric("Combined", f"{combined_sim:.3f}")

# ---------- FOOTER ----------
st.markdown("---")
if "live_log_df" in st.session_state and not st.session_state.live_log_df.empty:
    csv_bytes = st.session_state.live_log_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Log (CSV)", csv_bytes, "live_log.csv", "text/csv")
st.caption("🛫 Unified Predictive Maintenance Dashboard | XGBoost + LSTM | Real-time + Batch Analysis + Logging + Fault Insights")
st.sidebar.caption("Made by Kapilesh Simha with ❤️ | 2025")

