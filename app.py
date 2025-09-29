# app.py — Home Credit: Segmentation + Credit Score (Logistic Regression)
# Run: streamlit run app.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# --- Streamlit container compatibility helper (for older versions on the Cloud) ---
def box():
    """
    Streamlit >= 1.29 supports st.container(border=True).
    Older versions do not have the 'border' parameter and throw a TypeError.
    This wrapper works in both cases.
    """
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Credit Risk Simulator", layout="wide")

def sync_mpl_with_streamlit_theme():
    """Synchronizes Matplotlib chart colors with the active Streamlit theme."""
    bg  = st.get_option("theme.backgroundColor")
    sec = st.get_option("theme.secondaryBackgroundColor")
    txt = st.get_option("theme.textColor")
    mpl.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": sec,
        "axes.edgecolor": txt,
        "axes.labelcolor": txt,
        "text.color": txt,
        "xtick.color": txt,
        "ytick.color": txt,
        "grid.color": "#666666",
    })

sync_mpl_with_streamlit_theme()
st.title("🏦 Credit Risk Simulator")
st.caption("Home Credit • Calibrated Logistic Regression + K-Means Segment Interpretation")

ART = Path("artifacts")

# -----------------------------
# Session state & flags
# -----------------------------
# Initialize session state to manage app state across reruns.

if "valid_run" not in st.session_state:
    st.session_state["valid_run"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None

# If True, soft warnings will also block the final rendering.
BLOCK_ON_WARNINGS = True

# Load default widget values into session state once on first run.
DEFAULTS = {
    "AGE_YEARS": 40.0,
    "YEARS_EMPLOYED": 5.0,
    "AMT_INCOME_TOTAL": 180_000.0,
    "CNT_CHILDREN": 0,
    "AMT_CREDIT": 600_000.0,
    "AMT_ANNUITY": 30_000.0,
    "EXT_SOURCE_2": 0.55,
    "EXT_SOURCE_3": 0.50,
    "CODE_GENDER": "F",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_INCOME_TYPE": "Working",
    "NAME_CONTRACT_TYPE": "Cash loans",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---- Quick Presets (3 personas for easy demonstration) -------------------------
PRESETS = {
    "🟢 Low Risk (Stable)": {
        "AGE_YEARS": 46.0, "YEARS_EMPLOYED": 18.0,
        "AMT_INCOME_TOTAL": 1_200_000.0, "AMT_CREDIT": 800_000.0, "AMT_ANNUITY": 120_000.0,
        "EXT_SOURCE_2": 0.72, "EXT_SOURCE_3": 0.70, "CNT_CHILDREN": 1,
        "CODE_GENDER": "F", "NAME_FAMILY_STATUS": "Married",
        "NAME_EDUCATION_TYPE": "Higher education", "NAME_INCOME_TYPE": "Working",
        "NAME_CONTRACT_TYPE": "Cash loans",
    },
    "🟠 High Debt Burden": {
        "AGE_YEARS": 45.0, "YEARS_EMPLOYED": 4.8,
        "AMT_INCOME_TOTAL": 180_000.0,
        "AMT_CREDIT": 1_152_000.0,      # 180,000 * 6.4  → CTI ≈ 6.40
        "AMT_ANNUITY": 43_150.0,        # 1,152,000 / 26.7 ≈ 43,146
        "EXT_SOURCE_2": 0.55, "EXT_SOURCE_3": 0.55,
        "CNT_CHILDREN": 2,
        "CODE_GENDER": "M", "NAME_FAMILY_STATUS": "Married",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Working",
        "NAME_CONTRACT_TYPE": "Cash loans",
    },
     # 🔴 High Risk (Short Emp.): Very short employment, low external scores, moderate CTI (2–4)
    "🔴 High Risk (Fragile)": {
        "AGE_YEARS": 41.0, "YEARS_EMPLOYED": 1.0,     # Short employment duration
        "AMT_INCOME_TOTAL": 160_000.0,
        "AMT_CREDIT": 480_000.0,                      # CTI ≈ 3.0 (moderate)
        "AMT_ANNUITY": 26_000.0,                      # Term ≈ 18.5 years
        "EXT_SOURCE_2": 0.38, "EXT_SOURCE_3": 0.36,   # Low external scores
        "CNT_CHILDREN": 0,
        "CODE_GENDER": "M", "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_EDUCATION_TYPE": "Incomplete higher",
        "NAME_INCOME_TYPE": "Commercial associate",
        "NAME_CONTRACT_TYPE": "Cash loans",
    },
}


def apply_preset(values: dict):
    """Callback function to write preset values into session_state and rerun the app."""
    for k, v in values.items():
        st.session_state[k] = v
    # Safety check: ensure Years Employed does not exceed Age.
    st.session_state["YEARS_EMPLOYED"] = min(
        float(st.session_state.get("YEARS_EMPLOYED", 0.0)),
        float(st.session_state.get("AGE_YEARS", 40.0))
    )
    st.rerun()

# -----------------------------
# Helper Functions & Resources
# -----------------------------
@st.cache_resource
def load_artifacts():
    """
    Loads all required artifacts (models, preprocessors, configs) from the disk.
    This function is cached to prevent reloading heavy files on every app interaction.
    """
    # Load the main configuration file, which stores metadata and parameters.
    with (ART / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Load segmentation assets
    preproc_kmeans = load(ART / "preprocessor_kmeans.joblib")
    kmeans_k3      = load(ART / "kmeans_k3.joblib")

    # Load Logistic Regression assets
    cal_lr         = load(ART / "logreg_calibrated.joblib")     # Calibrated model for the final decision
    lr_raw         = load(ART / "logreg_raw.joblib")            # Raw pipeline, kept for coefficient analysis

    # Load the decision threshold
    with (ART / "threshold.json").open("r", encoding="utf-8") as f:
        thr = json.load(f)["youden_threshold"]

    # (Optional) Load pre-calculated performance metrics for the model card display
    metrics = None
    if (ART / "metrics.json").exists():
        with (ART / "metrics.json").open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    return cfg, preproc_kmeans, kmeans_k3, cal_lr, lr_raw, float(thr), metrics

# Load all necessary artifacts once and store them in global-like variables.
CFG, PRE_KM, KM3, CAL_LR, LR_RAW, THR, METRICS = load_artifacts()

# Unpack key settings from the config file for cleaner access throughout the app.
WIN_COLS   = CFG["winsor_cols"]
W_LOW      = CFG["winsor_low_bounds"]
W_HIGH     = CFG["winsor_high_bounds"]
INPUT_COLS = CFG["model_input_cols"]
OHE_CATS   = CFG.get("ohe_categories", {})          # Use .get for safety if key is missing

def apply_winsor(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Winsorization using the exact same boundaries as in the training notebook."""
    df = df.copy()
    for c in WIN_COLS:
        if c in df.columns:
            lo, hi = W_LOW.get(c), W_HIGH.get(c)
            if lo is not None and hi is not None:
                df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=lo, upper=hi)
    return df

def compute_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Generates derived features, ensuring consistency with the notebook's logic."""
    df = df.copy()
    def safe_div(a, b):         # A nested helper function to prevent division-by-zero errors
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        return pd.Series(np.where((b > 0) & np.isfinite(a) & np.isfinite(b), a / b, np.nan), index=a.index)

    # Calculate key financial ratios
    df["CREDIT_TO_INCOME_RATIO"]  = safe_div(df["AMT_CREDIT"],  df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["EMPLOYED_TO_AGE_RATIO"]   = safe_div(df["YEARS_EMPLOYED"], df["AGE_YEARS"])
    df["CREDIT_TERM"]             = safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    return df

def validate_row_inputs(row: dict):
    """
    Checks user inputs for logical inconsistencies before running the models.
    Returns a list of hard errors (which block execution) and soft warnings.
    """
    errs, warns = [], []
    # Safely convert inputs to the correct types for validation
    age = float(row["AGE_YEARS"])
    years = float(row["YEARS_EMPLOYED"])
    inc = float(row["AMT_INCOME_TOTAL"])
    cred = float(row["AMT_CREDIT"])
    ann = float(row["AMT_ANNUITY"])
    ext2 = float(row["EXT_SOURCE_2"])
    ext3 = float(row["EXT_SOURCE_3"])
    children = int(row["CNT_CHILDREN"])

    # Hard constraints: these are logically impossible and must be fixed by the user.
    if age < 18 or age > 85: errs.append("Age must be between 18 and 85.")
    if years < 0: errs.append("Years Employed cannot be negative.")
    if years > age: errs.append("Years Employed cannot exceed Age.")
    if inc <= 0: errs.append("Annual Income must be positive.")
    if cred <= 0: errs.append("Credit Amount must be positive.")
    if ann <= 0: errs.append("Annuity must be positive.")
    if children < 0: errs.append("Number of Children cannot be negative.")

    # External scores must be within the [0,1] range.
    if not (0 <= ext2 <= 1): errs.append("EXT_SOURCE_2 must be within [0, 1].")
    if not (0 <= ext3 <= 1): errs.append("EXT_SOURCE_3 must be within [0, 1].")
    if ann > cred: errs.append("Annuity cannot exceed Credit Amount (implies term < 1 year).")

   # Soft checks: these are not impossible, but highly unusual and worth noting.
    if inc > 0:
        cti = cred / inc
        if cti > 50:
            warns.append(f"Credit/Income ratio (CTI≈{cti:.1f}) is extremely high.")
    if age > 0:
        emp_age = years / age
        if emp_age > 1:
            errs.append("Employed/Age cannot exceed 1. Check Age / Years Employed.")
    return errs, warns

def local_lr_drivers_signed(cal_lr, df_w: pd.DataFrame, topn: int = 5, min_abs: float = 1e-12):
    """
    Calculates local feature contributions (x * beta) for a single prediction.
    This explains which features pushed the probability up (risk factors) or down (protective factors).
    """
    # Navigate through the CalibratedClassifierCV to get the underlying LR pipeline
    pipe = getattr(cal_lr, "base_estimator", None) or getattr(cal_lr, "estimator", None)
    if pipe is None or "preprocessor" not in pipe.named_steps or "model" not in pipe.named_steps:
        return [], []
    pre = pipe.named_steps["preprocessor"]
    lr  = pipe.named_steps["model"]

    # Preprocess the input row and get model coefficients
    X = pre.transform(df_w)
    if hasattr(X, "toarray"):
        X = X.toarray()
    x = X.ravel()
    coef = lr.coef_.ravel()

    # Element-wise product to find the contribution of each feature to the log-odds
    contrib = x * coef
    names = pre.get_feature_names_out()

    # Get top positive contributors (risk factors), sorted descending
    idx_desc = np.argsort(contrib)[::-1]
    pos = [(names[i], float(contrib[i])) for i in idx_desc if contrib[i] >  min_abs][:topn]

    # Get top negative contributors (protective factors), sorted ascending
    idx_asc = np.argsort(contrib)
    neg = [(names[i], float(contrib[i])) for i in idx_asc if contrib[i] < -min_abs][:topn]
    return pos, neg

# A dictionary to hold the interpretations for each segment. This is the content layer.
# SEGMENT_NOTES = {
#     1: ("🔴 Segment 1 – High Risk",
#         "Characterized by short employment history and/or low external scores. Applications should be routed for detailed review."),
#     0: ("🟠 Segment 0 – Medium-High Risk (High Debt Burden)",
#         "High credit-to-income ratio (CTI↑). Debt restructuring or cautious limits are recommended."),
#     2: ("🟢 Segment 2 – Low Risk (Stable Professionals)",
#         "Long employment history, good external scores. Suitable for expedited processes/premium products."),
# }

# A dictionary to hold the interpretations for each segment. This is the content layer.
SEGMENT_NOTES = {
    2: ("🟢 Segment 2 – Low Risk (Stable Professionals)",
        "Profile defined by long employment (avg. 18 years) and high external scores. A reliable customer base suitable for expedited approvals."),
    
    0: ("🟠 Segment 0 – High Debt Burden (CTI↑)",
        "This profile's primary risk driver is a high Credit-to-Income Ratio (avg. >6). Recommend cautious limits and close monitoring."),
        
    1: ("🔴 Segment 1 – High Risk (Short Tenure & Low External)",
        "Defined by a combination of short employment (avg. <4 years) and low external scores, making this the highest-risk group. Strict credit policies should apply."),
}

# --- Dictionaries and functions for creating user-friendly feature names ---
# These are used to translate raw feature names (e.g., 'AMT_INCOME_TOTAL') into readable labels (e.g., 'Annual Income').
PRETTY_NUM = {
    "AMT_INCOME_TOTAL": "Annual Income",
    "AMT_CREDIT": "Credit Amount",
    "AMT_ANNUITY": "Annuity",
    "EXT_SOURCE_2": "External Score 2",
    "EXT_SOURCE_3": "External Score 3",
    "AGE_YEARS": "Age",
    "YEARS_EMPLOYED": "Years Employed",
    "CREDIT_TO_INCOME_RATIO": "Credit/Income (CTI)",
    "ANNUITY_TO_INCOME_RATIO": "Annuity/Income",
    "EMPLOYED_TO_AGE_RATIO": "Employed/Age",
    "CREDIT_TERM": "Credit Term (≈ years)",
    "CNT_CHILDREN": "Number of Children",
}
PRETTY_CAT = {
    "CODE_GENDER": "Gender",
    "NAME_FAMILY_STATUS": "Marital Status",
    "NAME_EDUCATION_TYPE": "Education",
    "NAME_INCOME_TYPE": "Income Type",
    "NAME_CONTRACT_TYPE": "Contract Type",
}

def _split_cat_raw(raw: str, cat_cols):
    """Safely splits a one-hot encoded feature name like 'cat__COL_NAME_VALUE' into ('COL_NAME', 'VALUE')."""
    name = raw.replace("cat__", "")
    # Find the longest matching column name to handle cases where column names contain underscores
    match = None
    for c in sorted(cat_cols, key=len, reverse=True):
        if name == c or name.startswith(c + "_"):
            match = c
            break
    if match is None:
        return None, name
    level = name[len(match) + 1 :] if len(name) > len(match) else ""
    return match, level

def pretty_feature(raw_name: str, preprocessor) -> str:
    """Cleans prefixes like 'num__' and 'cat__' and returns a readable feature name for the UI."""
    if raw_name.startswith("num__"):
        col = raw_name.replace("num__", "")
        return PRETTY_NUM.get(col, col.replace("_", " "))
    if raw_name.startswith("cat__"):
        cat_cols = preprocessor.transformers_[1][2]
        col, level = _split_cat_raw(raw_name, cat_cols)
        col_lbl = PRETTY_CAT.get(col, (col or "").replace("_", " "))
        lvl_lbl = str(level).replace("_", " ")
        return f"{col_lbl} = {lvl_lbl}"
    return raw_name.replace("_", " ")

def prettify_driver_list(drivers, preprocessor):
    """Converts a list of raw driver names into a list of pretty, human-readable names."""
    return [(pretty_feature(n, preprocessor), v) for n, v in drivers]

def cat_opts(col, fallback):
    """Gets the category levels for a dropdown from the config file, with a fallback list for safety."""
    return OHE_CATS.get(col, fallback)

# ---- Decision engine (with rule trace) ----
def decide_with_trace(prob, thr, seg_id, cti):
    """
    Applies the policy layer to determine the final recommendation.
    Returns: (decision, reasons_list, rule_tag, trace_dict)
    - delta = prob - thr  (+ if above threshold, - if below)
    - near  = is the absolute difference < 2 percentage points?
    """
    delta = float(prob - thr)           # Positive => PD >= THR, Negative => PD < THR
    near  = abs(delta) < 0.02
    below = delta < 0                   # Is the probability below the threshold?
    risky_seg = seg_id in [0, 1]        # Is the customer in a risky segment?
    high_cti  = cti > 6                 # Is the debt burden excessive?

    # Default case: Approve if below threshold
    if below:
        decision = "🟢 Approve"
        reasons  = [f"PD {prob*100:.1f}% < threshold {thr*100:.1f}%"]
        rule_tag = "Approve — PD below threshold"

        # Escalate to Manual Review if score is very close to threshold AND in a risky segment, OR if debt is too high
        if (near and risky_seg) or high_cti:
            decision = "🟡 Manual Review"
            tag = "Segment 0/1 (near thr)" if (near and risky_seg) else f"CTI≈{cti:.1f}"
            reasons = [f"PD below threshold (margin {abs(delta)*100:.1f} p.p.)", tag]
            rule_tag = "Manual Review — Near threshold in risky segment or CTI > 6"
    else:
        # Above threshold: Decline
        decision = "🔴 Decline"
        reasons  = [f"PD {prob*100:.1f}% ≥ threshold {thr*100:.1f}%"]
        rule_tag = "Decline — PD above threshold"

    # Create a trace dictionary to explain the decision logic in the UI
    trace = dict(
        prob=float(prob), thr=float(thr),
        delta=float(delta),    # signed difference: prob - thr
        near=bool(near),
        below=bool(below),
        risky_seg=bool(risky_seg),
        high_cti=bool(high_cti),
        seg_id=int(seg_id),
        cti=float(cti),
    )
    return decision, reasons, rule_tag, trace

# -----------------------------
# Top: User Input Form + Presets
# -----------------------------

st.subheader("📋 Application Information")

# Proactively clamp YEARS_EMPLOYED to AGE before drawing the widgets.
# This prevents the initial default value from being invalid if the user changes the age.
st.session_state["YEARS_EMPLOYED"] = min(
    float(st.session_state.get("YEARS_EMPLOYED", 0.0)),
    float(st.session_state.get("AGE_YEARS", 40.0))
)

# ⚡ Quick Presets (buttons at the top for easy demonstration)
st.markdown("### Quick Presets")
cols = st.columns(len(PRESETS))
for i, (name, vals) in enumerate(PRESETS.items()):
    if cols[i].button(name, use_container_width=True, key=f"preset_{i}"):
        apply_preset(vals)

# The form will only trigger a rerun when the submit button is clicked.
with st.form("input_form"):
    c1, c2, c3 = st.columns(3)  # Organize the form into three columns for a cleaner UI

    with c1:
        # Using session_state keys ensures that values persist across reruns (e.g., after preset clicks).
        st.number_input("Age (years)", 18.0, 85.0, step=1.0, key="AGE_YEARS")
        st.number_input("Years Employed", 0.0, float(st.session_state.get("AGE_YEARS", 40.0)),
                        step=0.5, key="YEARS_EMPLOYED")
        st.number_input("Total Annual Income", min_value=1000.0, step=1000.0, key="AMT_INCOME_TOTAL")
        st.number_input("Number of Children", 0, 10, step=1, key="CNT_CHILDREN")

    with c2:
        st.number_input("Credit Amount", 1000.0, step=1000.0, key="AMT_CREDIT")
        st.number_input("Annuity (Yearly Payment)", 1.0, step=500.0, key="AMT_ANNUITY")
        st.number_input("External Source 2 (0–1)", min_value=0.0, max_value=1.0, step=0.01, key="EXT_SOURCE_2")
        st.number_input("External Source 3 (0–1)", min_value=0.0, max_value=1.0, step=0.01, key="EXT_SOURCE_3")

    with c3:
        st.selectbox("Gender", cat_opts("CODE_GENDER", ["F","M","XNA"]), key="CODE_GENDER")
        st.selectbox("Marital Status", cat_opts("NAME_FAMILY_STATUS",
                    ["Married","Single / not married","Civil marriage","Separated","Widow"]), key="NAME_FAMILY_STATUS")
        st.selectbox("Education", cat_opts("NAME_EDUCATION_TYPE",
                    ["Secondary / secondary special","Higher education","Incomplete higher","Lower secondary","Academic degree"]),
                    key="NAME_EDUCATION_TYPE")
        st.selectbox("Income Type", cat_opts("NAME_INCOME_TYPE",
                    ["Working","State servant","Commercial associate","Pensioner","Unemployed","Student","Businessman","Maternity leave"]),
                    key="NAME_INCOME_TYPE")
        st.selectbox("Contract Type", cat_opts("NAME_CONTRACT_TYPE", ["Cash loans","Revolving loans"]),
                    key="NAME_CONTRACT_TYPE")

    # This button triggers the form submission and the logic below.
    submitted = st.form_submit_button("Get Score")

# -----------------------------
# When submitted: validate -> compute -> store results
# -----------------------------
if submitted:
    # 1. Collect all inputs from session state into a dictionary.
    row = {
        "CODE_GENDER": st.session_state["CODE_GENDER"],
        "CNT_CHILDREN": int(st.session_state["CNT_CHILDREN"]),
        "NAME_FAMILY_STATUS": st.session_state["NAME_FAMILY_STATUS"],
        "NAME_EDUCATION_TYPE": st.session_state["NAME_EDUCATION_TYPE"],
        "AMT_INCOME_TOTAL": float(st.session_state["AMT_INCOME_TOTAL"]),
        "NAME_INCOME_TYPE": st.session_state["NAME_INCOME_TYPE"],
        "NAME_CONTRACT_TYPE": st.session_state["NAME_CONTRACT_TYPE"],
        "AMT_CREDIT": float(st.session_state["AMT_CREDIT"]),
        "AMT_ANNUITY": float(st.session_state["AMT_ANNUITY"]),
        "EXT_SOURCE_2": float(st.session_state["EXT_SOURCE_2"]),
        "EXT_SOURCE_3": float(st.session_state["EXT_SOURCE_3"]),
        "AGE_YEARS": float(st.session_state["AGE_YEARS"]),
        "YEARS_EMPLOYED": float(st.session_state["YEARS_EMPLOYED"]),
    }

    # 2. Validate the inputs FIRST.
    errs, warns = validate_row_inputs(row)
    if errs or (BLOCK_ON_WARNINGS and warns):
        # If invalid, clear any previous results and mark the run as invalid.
        st.session_state["valid_run"] = False
        st.session_state["results"] = None

        if errs:
            st.error("Please fix the following input issues to continue:")
            for e in errs: st.write(f"- {e}")
        if warns and BLOCK_ON_WARNINGS:
            st.warning("Fix these warnings to continue:")
            for w in warns: st.write(f"- {w}")
    else:
        # If inputs are valid, proceed with calculations.
        for w in warns:
            st.warning(w)

        # 3. Apply the full data preparation pipeline.
        df = pd.DataFrame([row], dtype=object)
        df = compute_engineered(df) # Calculate derived features
        df = df.reindex(columns=INPUT_COLS, fill_value=np.nan)  # Ensure schema consistency
        df_w = apply_winsor(df) # Apply Winsorization

        # --- Run Models ---
        # Get the final calibrated probability.
        prob = float(CAL_LR.predict_proba(df_w)[0, 1])

        # Get the top factors influencing the score for interpretability.
        drivers_pos, drivers_neg = local_lr_drivers_signed(CAL_LR, df_w, topn=5)
        try:
            pipe = getattr(CAL_LR, "base_estimator", None) or getattr(CAL_LR, "estimator", None)
            pre = pipe.named_steps["preprocessor"]
            drivers_pos = prettify_driver_list(drivers_pos, pre)
            drivers_neg = prettify_driver_list(drivers_neg, pre)
        except Exception:
            pass    # Fail silently if drivers can't be computed
        
        # Get the segment assignment for contextual information.
        seg_id = int(KM3.predict(PRE_KM.transform(df_w))[0])
        seg_title, seg_note = SEGMENT_NOTES.get(seg_id, (f"Segment {seg_id}", ""))

        # Apply the policy layer to get the final recommendation and reasons.
        cti = float(df["CREDIT_TO_INCOME_RATIO"].iloc[0])
        decision, reasons, rule_tag, trace = decide_with_trace(prob, THR, seg_id, cti)

        # 4. Store all results in session state for the rendering section.
        st.session_state["results"] = {
            "prob": prob,
            "decision": decision,
            "reasons": reasons,
            "drivers_pos": drivers_pos,
            "drivers_neg": drivers_neg,
            "seg_title": seg_title,
            "seg_note": seg_note,
            "df": df,
            "trace": trace,
            "rule_tag": rule_tag,
        }
        st.session_state["valid_run"] = True

# -----------------------------
# RENDER: This entire section runs only if we have a valid and successful run.
# -----------------------------
# Retrieve the results stored in the session state from the computation step.

res = st.session_state.get("results")
if st.session_state.get("valid_run") and res:

    # --------- TOP SECTION: Final decision and LR summary ---------
    st.subheader("🎯 Final Decision (Calibrated Logistic Regression)")
    col_prob, col_decision = st.columns([1,1])
    with col_prob:
        st.metric("Probability of Default (PD)", f"{res['prob']*100:.1f}%")
    with col_decision:
        st.metric("Recommendation", res["decision"])

    # Extract the trace dictionary for detailed explanation
    t = res["trace"]

    with box(): # Use the compatibility helper for bordered containers
        st.markdown("#### 🧭 Why this decision?")

        # Safely get values from the trace dictionary
        prob = float(t.get("prob", 0.0))
        thr  = float(t.get("thr", 0.0))
        seg_id = int(t.get("seg_id", -1))
        cti = float(t.get("cti", float("nan")))
        delta = prob - thr                 # if Positive, PD > THR
        delta_pp = delta * 100
        below = prob < thr
        dir_word = "over by" if delta_pp >= 0 else "below by"

        # Display common information for all cases
        st.markdown(
            f"- **Decision Logic:** `{res.get('rule_tag','')}`  \n"
            f"- **PD / Threshold:** **{prob*100:.1f}%** / **{thr*100:.1f}%** "
            f"({dir_word} {abs(delta_pp):.1f} p.p.)  \n"
            f"- **Segment:** **{res['seg_title']}** (id={seg_id})  \n"
            f"- **CTI:** **{cti:.2f}**"
        )

        # Show the secondary checks only if the PD is below the threshold
        if below:
            st.markdown(f"**Primary reason:** PD below threshold by **{abs(delta_pp):.1f} p.p.**")
            near      = bool(t.get("near", False))
            risky_seg = bool(t.get("risky_seg", False))
            high_cti  = bool(t.get("high_cti", False))
            st.markdown(
                f"- Near threshold (<2 p.p.)? {'✅' if near else '❌'}  \n"
                f"- Risky segment (0/1)? {'✅' if risky_seg else '❌'}  \n"
                f"- High CTI (>6)? {'✅' if high_cti else '❌'}"
            )
        else:
            st.markdown(f"**Primary reason:** PD exceeds threshold by **{abs(delta_pp):.1f} p.p.**")

        with st.expander("Show decision logic (pseudo)"):
            st.code(
                "delta = PD - THR\n"
                "if delta >= 0: Decline\n"
                "elif (abs(delta) < 0.02 and segment in {0,1}) or (CTI > 6): Manual Review\n"
                "else: Approve",
                language="text",
            )
    # Display the concise reasons and the threshold value.
    st.caption(" • ".join(res["reasons"]) + f" • Threshold (Youden): **{THR:.4f}**")

    # Display the top factors that increased or decreased the risk score.
    if res.get("drivers_pos"):
        st.caption("**Top factors increasing PD:** " + ", ".join(name for name, _ in res["drivers_pos"]))
    if res.get("drivers_neg"):
        st.caption("**Top factors decreasing PD:** " + ", ".join(name for name, _ in res["drivers_neg"]))

    # Display the overall model performance metrics.
    if METRICS:
        st.markdown("**Model Card (Test Set Summary)**")
        mk = METRICS
        st.write(
            f"AUC: **{mk['auc_cal']:.3f}**  •  PR-AUC: **{mk['prauc_cal']:.3f}**  •  "
            f"Gini: **{mk['gini']:.3f}**  •  KS: **{mk['ks']:.3f}**"
        )
        st.write(
            f"TPR@threshold: **{mk['tpr_at_threshold']:.3f}**  •  FPR@threshold: **{mk['fpr_at_threshold']:.3f}**"
        )

    st.divider()

    # --------- BOTTOM SECTION: Segment interpretation ---------
    st.subheader("📊 Customer Segment & Interpretation (K-Means)")
    st.markdown(f"**Estimated Segment:** {res['seg_title']}")
    st.caption(res["seg_note"])

    # Display the main engineered ratios for the user's input.
    show_cols = ["CREDIT_TO_INCOME_RATIO","ANNUITY_TO_INCOME_RATIO","EMPLOYED_TO_AGE_RATIO","CREDIT_TERM"]
    pretty = (res["df"][show_cols]
              .rename(columns={
                  "CREDIT_TO_INCOME_RATIO":"Credit/Income (CTI)",
                  "ANNUITY_TO_INCOME_RATIO":"Annuity/Income",
                  "EMPLOYED_TO_AGE_RATIO":"Employed/Age",
                  "CREDIT_TERM":"Credit Term (≈ years)"
              }).round(3))
    st.dataframe(pretty, use_container_width=True)

    st.divider()
    with st.expander("🔧 Input Summary (as per model schema)"):
        st.dataframe(res["df"], use_container_width=True)

else:
    # If we are here, it's either the first run or the last run was invalid.
    st.info("Fill out the form above and press **Get Score**. Results render only when inputs pass validation.")

# -----------------------------
# Scatter plot sample loading (SHOW ONLY on valid runs)
# -----------------------------
@st.cache_resource
def load_scatter_sample():
    """Load the pre-generated sample data for the background of the scatter plot."""
    p_parq = ART / "scatter_sample.parquet"
    p_csv  = ART / "scatter_sample.csv"

    # Try to read Parquet first, but fall back to CSV safely if pyarrow is missing or file is corrupt.
    if p_parq.exists():
        try:
            return pd.read_parquet(p_parq)  # No engine needed; will raise ImportError if pyarrow is missing
        except Exception as e:
            # This might happen on the Cloud if pyarrow is not installed or the file is problematic
            pass

    if p_csv.exists():
        try:
            return pd.read_csv(p_csv)
        except Exception:
            pass

    return None

# Render the map only if the last run was valid and successful.
res = st.session_state.get("results")
if st.session_state.get("valid_run") and res:
    # Load the sample data only when it's needed, inside the conditional block.
    sample_df = load_scatter_sample()
    st.markdown("### 📍 Your Position (Segment Map)")

    if sample_df is None:
        st.info("Background sample data not found. Please generate the sample file in the notebook "
                "and save it as `artifacts/scatter_sample.parquet`.")
    else:
        # Get theme colors to make the plot's background match the app's theme.
        bg  = st.get_option("theme.backgroundColor") or "#FFFFFF"
        sec = st.get_option("theme.secondaryBackgroundColor") or "#F5F7FB"

        # Create a new figure and axes for the plot.
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg)
        ax.set_facecolor(sec)

        # Draw the background scatter plot with the sample data.
        sns.scatterplot(
            data=sample_df,
            x="CREDIT_TO_INCOME_RATIO",
            y="YEARS_EMPLOYED",
            hue="Segment",
            palette="viridis",
            alpha=0.35, # Use transparency to show density
            s=18,
            ax=ax,
            legend=True
        )

        # Get the user's coordinates from the results.
        user_cti   = float(res["df"]["CREDIT_TO_INCOME_RATIO"].iloc[0])
        user_years = float(res["df"]["YEARS_EMPLOYED"].iloc[0])

        # Overlay the user's application as a large 'X' on the plot.
        ax.scatter([user_cti], [user_years],
                   marker="X", s=240, linewidths=1.5, edgecolor="black",
                   color="red", zorder=10, label="You Are Here")
        ax.annotate("You Are Here",
                    xy=(user_cti, user_years),
                    xytext=(user_cti * 1.05 if user_cti != 0 else 0.5, user_years + 1.5),
                    arrowprops=dict(arrowstyle="->", lw=1.2),
                    fontsize=11)

        # Dynamically adjust axis limits for better visualization.
        x_max = max(user_cti, sample_df["CREDIT_TO_INCOME_RATIO"].quantile(0.995))
        y_max = max(user_years, sample_df["YEARS_EMPLOYED"].quantile(0.995))
        ax.set_xlim(left=0, right=x_max * 1.05)
        ax.set_ylim(bottom=0, top=y_max * 1.05)

        # Ensure the "You Are Here" label appears correctly in the legend.
        handles, labels = ax.get_legend_handles_labels()
        if "You Are Here" not in labels:
            handles.append(plt.Line2D([0], [0], marker='X', color='w',
                                      markerfacecolor='red', markeredgecolor='black',
                                      markersize=10, linewidth=0, label='You Are Here'))
            labels.append("You Are Here")
        ax.legend(handles, labels, title="Segment", loc="upper right")

        # Set plot titles and labels.
        ax.set_title("Segment Distribution by the Two Most Differentiating Features", fontsize=16)
        ax.set_xlabel("Credit-to-Income Ratio (CTI)", fontsize=12)
        ax.set_ylabel("Years Employed", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Display the plot in the Streamlit app.
        st.pyplot(fig)