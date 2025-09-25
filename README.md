# Home Credit — Credit Risk Analysis & Segmentation denemeee

An end-to-end credit risk project featuring customer **segmentation (K-Means)** and a **credit scoring model (calibrated Logistic Regression)**, deployed as an interactive **Streamlit** application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<streamlit-uygulamanizin-linki-buraya>.streamlit.app/)

---

## 📊 Project Overview

This repository demonstrates a complete machine learning lifecycle on the “Home Credit Default Risk” dataset, covering:
- **Customer Segmentation (K-Means, k=3):** Unsupervised clustering to identify 3 distinct customer personas, providing a strategic overview of the applicant base.
- **Credit Scoring (Logistic Regression + Calibration):** A supervised model that predicts the Probability of Default (PD). The model is calibrated to ensure its probability outputs are reliable and can be tied to real-world outcomes.
- **Interactive Simulation (Streamlit):** A web application where users can input applicant data to receive a real-time risk assessment, including a final recommendation (Approve / Decline / Manual Review), key risk drivers, and their position on a "Segment Map".
- **Reproducibility:** The entire pipeline, including models, preprocessors, and configuration, is exported as versioned artifacts to ensure consistency between the notebooks and the deployed application.

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
├── notebooks/                           # Development notebooks
│   ├── 1_Customer_Segmentation_(K-Means).ipynb
│   └── 2_Credit_Scoring_Model_(Logistic_Regression).ipynb
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

The analysis in this repository follows a two-step process:

1.  **`1_Customer_Segmentation_(K-Means).ipynb`**  
    This notebook reads the raw data, performs feature engineering, builds and validates the K-Means segments (k=3), and exports the necessary artifacts (preprocessor, model, config files).

2.  **`2_Credit_Scoring_Model_(Logistic_Regression).ipynb`**  
    This notebook loads the artifacts from the first notebook to ensure consistency. It then trains, evaluates, and calibrates the Logistic Regression model, exporting the final calibrated model and its performance metrics.

> ✅ **Important:** Run both notebooks in order to generate the `artifacts/` folder, which is required before launching the Streamlit app.

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

## 🔎 Notes on Modeling

*   **k=3 vs k=4:** Although k=4 showed some gain in the Elbow Method, **k=3** was selected for the final segmentation model. This decision prioritized creating crisper, more statistically robust, and more easily interpretable clusters for the interactive "Segment Map" in the Streamlit app.
*   **Calibration:** The Logistic Regression model was probability-calibrated using Isotonic Regression. This significantly improved the reliability of the Probability of Default (PD) outputs, making them more aligned with real-world frequencies.
*   **Explainability:** The Streamlit app displays a list of the top local drivers for each prediction, showing which features contributed most to increasing or decreasing the calculated PD score.

## 🗂️ Data

The dataset used is the **Home Credit Default Risk** competition from Kaggle. Please refer to the dataset's original source for license and usage terms. The raw data files are not stored in this repository due to their size and should be downloaded separately.

## 📜 License

This project is released under the **MIT License**. See the `LICENSE` file for details.
