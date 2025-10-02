# Home Credit — Credit Risk Analysis & Segmentation

An end-to-end credit risk project featuring customer **segmentation (K-Means)** and a **credit scoring model (calibrated Logistic Regression)**, deployed as an interactive **Streamlit** application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-analysis-portfolio.streamlit.app/)

---

## 📊 Project Overview

This repository demonstrates a complete ML lifecycle on the Home Credit Default Risk dataset, covering:
- **Customer Segmentation (K-Means, k=3):** Unsupervised clustering to identify 3 distinct applicant personas, providing strategic portfolio insight and context for individual decisions.
- **Credit Scoring (Logistic Regression + Calibration):** A supervised model that predicts the Probability of Default (PD). The model is calibrated to produce trustworthy probabilities aligned with real-world frequencies.
- **Interactive Simulation (Streamlit):** A web application where users can input applicant data to receive a real-time PD score, a final recommendation (Approve / Manual Review / Decline), top risk drivers, and their position on a "Segment Map".
- **Reproducibility:** Models, preprocessors, thresholds, and configurations are exported as versioned artifacts to keep the notebooks and the final application perfectly in sync.

> **Quick Performance Snapshot (Test Set)**  
> AUC ≈ **0.732** • PR-AUC ≈ **0.212** • *Calibrated probabilities via Isotonic Regression (Brier Score significantly improved)*

---

## 🧠 Decision Policy in the App

The Streamlit application applies a lightweight, rule-based decision layer on top of the predicted PD score to generate a final recommendation:
- **Primary Rule:** If `PD ≥ Threshold` → **Decline**; otherwise **Approve**.
- **Escalation Rule:** Even if below the threshold, if the score is very close to it (*and* the applicant is in a risky segment) **or** if the Credit-to-Income Ratio is excessively high (`CTI > 6`), the recommendation is escalated to **Manual Review**.
- The `Threshold` itself is the Youden-optimal point computed in the notebook and exported as an artifact.

---

## 📂 Repository Structure

```
repo-root/
├── app.py                               # Streamlit application
├── artifacts/                           # Generated models, preprocessors, configs (output of notebooks)
│   ├── config.json
│   ├── logreg_calibrated.joblib
│   ├── logreg_raw.joblib
│   ├── preprocessor_kmeans.joblib
│   ├── kmeans_k3.joblib
│   ├── threshold.json
│   └── metrics.json
├── 1_Customer_Segmentation_(K-Means).ipynb                     
├── 2_Credit_Scoring_Model_(Logistic_Regression).ipynb
│  
├── assets/                              # Static assets for README/app (screenshots, icons, etc.)
│   └── app_preview.png
├── requirements.txt                     # Python dependencies
├── runtime.txt                          # (Optional) Python version for Streamlit Cloud (e.g., “3.10”)
├── .gitignore
├── LICENSE                              # (Recommended) Project license
└── README.md
```
---

## 📓 Notebooks

1.  **`1_Customer_Segmentation_(K-Means).ipynb`**  
    Builds and validates the K-Means (k=3) segmentation model, then exports the segmentation preprocessor, model, and shared project configuration.

2.  **`2_Credit_Scoring_Model_(Logistic_Regression).ipynb`**  
    Trains and **calibrates** the Logistic Regression PD model, computes the Youden-optimal threshold, and exports the final calibrated model and all performance metrics.

> ✅ **Important:** The `artifacts/` folder is included in this repository, so you can run the app directly. If you wish to regenerate the artifacts, you must run the notebooks in order.

---

## 🚀 How to Run the Simulation Locally

**Prerequisites**
- Python **3.10+**
- `pip` package installer

**1. Clone the repository**
```bash
git clone https://github.com/EkremKarnas/Credit-Risk-Analysis-Portfolio.git
cd Credit-Risk-Analysis-Portfolio
```
**2. (Recommended) Create and activate a virtual environment**
```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the Streamlit application**
```bash
streamlit run app.py
```
The application will then open in your web browser.

## 🔎 Notes on Modeling

*   **k=3 vs k=4:** While `k=4` showed a small gain in the Elbow Method, **`k=3`** was selected for the final segmentation model. This decision prioritized creating crisper, more statistically stable, and more easily interpretable clusters for the interactive "Segment Map".

*   **Calibration:** The Logistic Regression model was probability-calibrated using Isotonic Regression (`CalibratedClassifierCV`). This process preserves the model's ranking power (AUC/PR-AUC remain the same) while making the Probability of Default (PD) outputs more realistic and trustworthy (Brier Score significantly improves).

*   **Explainability:** The Streamlit app surfaces local LR drivers (the top positive and negative contributors) for each prediction to provide transparency into the model's decision-making process.

## 🗂️ Data

The dataset used is the **Home Credit Default Risk** competition from Kaggle. Please refer to the dataset's original source for license and usage terms. The raw data files are not stored in this repository due to their size and should be downloaded separately.

## 📜 License & Disclaimer

This project is released under the **MIT License**. See the `LICENSE` file for details.
This is an educational and portfolio project and is not affiliated with Home Credit in any way.
