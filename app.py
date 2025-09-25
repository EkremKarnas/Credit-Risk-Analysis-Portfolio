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

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Credit Risk Simulator", layout="wide")
def sync_mpl_with_streamlit_theme():
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
st.caption("Home Credit (application_train.csv) • Calibrated Logistic Regression (final decision) + K-Means Segment Interpretation")

ART = Path("artifacts")

# -----------------------------
# Session state (to hide stale results after input changes)
# -----------------------------
# Initialize session state variables if they don't exist.
# This prevents errors on the first run.
if "valid_run" not in st.session_state:
    st.session_state["valid_run"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None

# A flag to control behavior: if True, soft warnings will also block the final rendering.
BLOCK_ON_WARNINGS = True

# -----------------------------
# Helper Functions & Resources
# -----------------------------
@st.cache_resource       # Crucial for performance: loads heavy models and configs only once.
def load_artifacts():
    """Loads all required artifacts (models, preprocessors, configs) from the disk."""
    # Load the main configuration file, which stores metadata and parameters.
    with (ART / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Load segmentation assets
    preproc_kmeans = load(ART / "preprocessor_kmeans.joblib")
    kmeans_k3      = load(ART / "kmeans_k3.joblib")

    # Load Logistic Regression assets
    cal_lr         = load(ART / "logreg_calibrated.joblib")  # Calibrated model for the final decision
    lr_raw         = load(ART / "logreg_raw.joblib")         # Raw pipeline, kept for reference and analysis

    # Load the decision threshold from its own file
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
OHE_CATS   = CFG.get("ohe_categories", {})  # Use .get for safety if key is missing

def apply_winsor(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Winsorization using the exact same boundaries as in the notebook."""
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
    def safe_div(a, b): # A helper function to prevent division-by-zero errors
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        return pd.Series(np.where((b > 0) & np.isfinite(a) & np.isfinite(b), a / b, np.nan), index=a.index)

    # Calculate key financial ratios
    df["CREDIT_TO_INCOME_RATIO"]  = safe_div(df["AMT_CREDIT"],  df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["EMPLOYED_TO_AGE_RATIO"]   = safe_div(df["YEARS_EMPLOYED"], df["AGE_YEARS"])
    df["CREDIT_TERM"]             = safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    return df

# -------- Input validation (hard block on errors; optionally block on warnings) --------
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
    if age < 18 or age > 85: errs.append("Age must be between 18 and 85.") # Assuming one can start working at 18
    if years < 0: errs.append("Years Employed cannot be negative.")
    if years > age: errs.append("Years Employed cannot exceed Age.")
    if inc <= 0: errs.append("Annual Income must be positive.")
    if cred <= 0: errs.append("Credit Amount must be positive.")
    if ann <= 0: errs.append("Annuity must be positive.")
    if children < 0: errs.append("Number of Children cannot be negative.")

    # External scores must be within the [0,1] range.
    if not (0 <= ext2 <= 1): errs.append("EXT_SOURCE_2 must be within [0, 1].")
    if not (0 <= ext3 <= 1): errs.append("EXT_SOURCE_3 must be within [0, 1].")

    # Relationships between fields
    if ann > cred:
        errs.append("Annuity cannot exceed Credit Amount (implies term < 1 year).")

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

# --- Local LR drivers (x * beta): return both positive and negative contributors ---
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

# A dictionary to hold the interpretations for each segment.
SEGMENT_NOTES = {
    1: ("🔴 Segment 1 – High Risk",
        "Characterized by short employment history and/or low external scores. Applications should be routed for detailed review."),
    0: ("🟠 Segment 0 – Medium-High Risk (High Debt Burden)",
        "High credit-to-income ratio (CTI↑). Debt restructuring or cautious limits are recommended."),
    2: ("🟢 Segment 2 – Low Risk (Stable Professionals)",
        "Long employment history, good external scores. Suitable for expedited processes/premium products."),
}

# ---------- Dictionaries and functions for creating user-friendly feature names ----------
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
    """Cleans prefixes like 'num__' and 'cat__' and returns a readable feature name."""
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
    """Converts a list of raw driver names into a list of pretty names."""
    return [(pretty_feature(n, preprocessor), v) for n, v in drivers]
def cat_opts(col, fallback):
    """Gets the category levels for a dropdown from the config file, with a fallback list."""
    return OHE_CATS.get(col, fallback)

# -----------------------------
# Top: User Input Form
# -----------------------------
st.subheader("📋 Application Information")

# Using st.form ensures that the app reruns only when the submit button is clicked.
with st.form("input_form"):
    c1, c2, c3 = st.columns(3)  # Organize the form into three columns for a cleaner UI
    with c1:
        age = st.number_input("Age (years)", min_value=18.0, max_value=85.0, value=40.0, step=1.0)

        # Dynamic max value for Years Employed based on Age (in-field guardrail)
        years_emp = st.number_input(
            "Years Employed",
            min_value=0.0,
            max_value=float(age),                 # Dynamically set the upper limit
            value=min(5.0, float(age)),           # Also pull the default value into a safe range
            step=0.5
        )

        income   = st.number_input("Total Annual Income", min_value=1000.0, value=180_000.0, step=1000.0)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)

    with c2:
        amt_credit  = st.number_input("Credit Amount", 1000.0, value=600_000.0, step=1000.0)
        amt_annuity = st.number_input("Annuity (Yearly Payment)", 1.0, value=30_000.0, step=500.0)
        ext2        = st.number_input("EXT_SOURCE_2 (0–1)", 0.0, 1.0, 0.55, 0.01)
        ext3        = st.number_input("EXT_SOURCE_3 (0–1)", 0.0, 1.0, 0.50, 0.01)
    with c3:
        code_gender = st.selectbox("Gender (CODE_GENDER)", cat_opts("CODE_GENDER", ["F","M","XNA"]), index=0)
        family      = st.selectbox("Marital Status", cat_opts("NAME_FAMILY_STATUS",
                                ["Married","Single / not married","Civil marriage","Separated","Widow"]), index=0)
        edu         = st.selectbox("Education", cat_opts("NAME_EDUCATION_TYPE",
                                ["Secondary / secondary special","Higher education","Incomplete higher",
                                 "Lower secondary","Academic degree"]), index=0)
        income_type = st.selectbox("Income Type", cat_opts("NAME_INCOME_TYPE",
                                ["Working","State servant","Commercial associate","Pensioner",
                                 "Unemployed","Student","Businessman","Maternity leave"]), index=0)
        contract    = st.selectbox("Contract Type", cat_opts("NAME_CONTRACT_TYPE",
                                ["Cash loans","Revolving loans"]), index=0)

    # This button triggers the form submission and the logic below.
    submitted = st.form_submit_button("Get Score")

# -----------------------------
# When submitted: validate -> compute -> store results or hide
# -----------------------------
if submitted:
    # 1. Build a dictionary from the raw form inputs
    row = {
        "CODE_GENDER": code_gender,
        "CNT_CHILDREN": int(children),
        "NAME_FAMILY_STATUS": family,
        "NAME_EDUCATION_TYPE": edu,
        "AMT_INCOME_TOTAL": float(income),
        "NAME_INCOME_TYPE": income_type,
        "NAME_CONTRACT_TYPE": contract,
        "AMT_CREDIT": float(amt_credit),
        "AMT_ANNUITY": float(amt_annuity),
        "EXT_SOURCE_2": float(ext2),
        "EXT_SOURCE_3": float(ext3),
        "AGE_YEARS": float(age),
        "YEARS_EMPLOYED": float(years_emp),
    }

    # 2. Validate the inputs FIRST
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

        # 3. Apply the full data preparation pipeline
        df = pd.DataFrame([row], dtype=object)
        df = compute_engineered(df) # Calculate derived features
        df = df.reindex(columns=INPUT_COLS, fill_value=np.nan)  # Ensure schema consistency
        df_w = apply_winsor(df) # Apply Winsorization

        # --- Run Models ---
        # Get the final calibrated probability from the Logistic Regression model.
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

        # 4. Apply the simple policy layer for the final recommendation
        decision = "🟢 Approve"
        reasons = [f"PD {prob*100:.1f}% < threshold {THR*100:.1f}%"]
        if prob >= THR:
            decision = "🔴 Decline"
            reasons = [f"PD {prob*100:.1f}% ≥ threshold {THR*100:.1f}%"]
        else:   # Below threshold, but check for edge cases
            margin = THR - prob
            cti = float(df["CREDIT_TO_INCOME_RATIO"].iloc[0])
            # If the score is very close to the threshold and the customer is in a risky segment, or if CTI is too high...
            if (margin < 0.02 and seg_id in [0,1]) or (cti > 6):
                decision = "🟡 Manual Review"
                tag = "Segment 0/1" if seg_id in [0,1] else f"CTI≈{cti:.1f}"
                reasons = [f"PD below threshold (margin {margin*100:.1f} p.p.)", tag]

        # 5. Store all results in session state for rendering.
        st.session_state["results"] = {
            "prob": prob,
            "decision": decision,
            "reasons": reasons,
            "drivers_pos": drivers_pos,
            "drivers_neg": drivers_neg,
            "seg_title": seg_title,
            "seg_note": seg_note,
            "df": df,
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
    # Display the reasons for the recommendation.
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
    st.subheader("📊 Customer Segment & Interpretation (K-Means, k=3)")
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
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    return None

sample_df = load_scatter_sample()

# Render the map only if the last run was valid and successful.
res = st.session_state.get("results")
if st.session_state.get("valid_run") and res:
    st.markdown("### 📍 Your Position (Segment Map)")

    if sample_df is None:
        st.info("Background sample data not found. Please generate the sample file in the notebook "
                "and save it as `artifacts/scatter_sample.parquet`.")
    else:
        # --- add these 3 lines right before creating the figure ---
        bg  = st.get_option("theme.backgroundColor") or "#FFFFFF"
        sec = st.get_option("theme.secondaryBackgroundColor") or "#F5F7FB"
        txt = st.get_option("theme.textColor") or "#111827"

        fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg)  # <-- facecolor ver
        ax.set_facecolor(sec)       
        fig, ax = plt.subplots(figsize=(12, 8))
        # Draw the background scatter plot with the sample data.
        sns.scatterplot(
            data=sample_df,
            x="CREDIT_TO_INCOME_RATIO",
            y="YEARS_EMPLOYED",
            hue="Segment",
            palette="viridis",
            alpha=0.35,
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

        # Dynamically adjust axis limits for better visualization, preventing outliers from squeezing the plot.
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

        ax.set_title("Segment Distribution by the Two Most Differentiating Features", fontsize=16)
        ax.set_xlabel("Credit-to-Income Ratio (CTI)", fontsize=12)
        ax.set_ylabel("Years Employed", fontsize=12)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
# If not a valid run, we draw nothing; the info/error messages above will be displayed instead.


