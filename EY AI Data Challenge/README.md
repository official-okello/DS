# 🌊 Water Quality Prediction System (XGBoost + Explainability)

## 📌 Project Overview

This project implements a **production-ready machine learning system** for **water quality prediction** using **XGBoost**, with **full explainability via SHAP**.
The solution is designed for **real-world deployment**, prioritizing **accuracy, transparency, and reproducibility**, and is suitable for **data science challenges, policy decision support, and operational monitoring**.

The system:

* Predicts a **Water Quality Index (WQI)** from environmental and physicochemical variables
* Uses **XGBoost** for strong tabular performance
* Provides **global and local explanations** using **SHAP**
* Runs entirely via **Python scripts (no notebooks)**

---

## 🧠 Why XGBoost?

XGBoost was selected because:

* It consistently outperforms neural networks on **structured environmental data**
* Handles **non-linear interactions and missing values**
* Trains efficiently on limited or noisy datasets
* Supports **native SHAP explainability**

---

## 🗂 Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── csv's
│   └── processed/
│       └── csv's
│
├── src/
    ├── evaluate.py
    ├── features.py
    ├── preprocessing.py
│   ├── train_xgboost.py
│   ├── tune_xgboost.py
│   └── shap.py
│
├── outputs/
│   ├── models/
│   │   └── xgb_model.pkl
│   ├── predictions/
│   └── shap/
│       ├── shap_summary.png
│       ├── shap_bar.png
│       └── shap_force_0.png
│
├── requirements.txt
└── README.md
```

---

## 📊 Data Description

**Input features** typically include though not limited to:

* pH
* Turbidity
* Dissolved oxygen
* Electrical conductivity
* Temperature
* Nitrate / phosphate levels
* Time and location indicators

**Target variables**:

* `Total Alkalinity` (continuous regression target)
* `Electrical Conductance` (continuous regression target)
* `Dissolved Reactive Phosphorus` (continuous regression target)

---

## ⚙️ Environment Setup

### 1️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> Quick smoke test: after creating and activating your virtual environment and installing the requirements, run:
>
> ```bash
> python3 scripts/run_smoke_test.py
> ```
> This runs a small end-to-end check (preprocessing -> features) and prints shapes to confirm everything is wired up.

---

## 🏗 Model Training

### Train baseline XGBoost model

```bash
python src/train_xgboost.py
```

This:

* Loads processed features
* Trains an XGBoost regressor
* Saves the model to:

```
outputs/models/xgb_model.pkl
```

---

## 🎯 Hyperparameter Tuning

```bash
python src/tune_xgboost.py
```

Tuning strategy:

* Randomized or grid search
* Cross-validated RMSE
* Early stopping for efficiency

Best parameters are automatically applied and the final model is saved.

---

## 🔍 Explainability with SHAP

```bash
python src/explain_shap.py
```

### Outputs generated

| File               | Purpose                               |
| ------------------ | ------------------------------------- |
| `shap_summary.png` | Global feature impact                 |
| `shap_bar.png`     | Feature importance ranking            |
| `shap_force_0.png` | Local explanation (single prediction) |

---

## 📈 Model Explainability Strategy

We use **SHAP (SHapley Additive Explanations)** to:

* Explain **why** predictions are made
* Identify **key environmental drivers** of water quality
* Support **regulatory trust and transparency**

### Global Explanation

Shows which variables most influence predictions across the dataset.

### Local Explanation

Explains individual predictions — critical for alerts and audits.

---

## 🧪 Evaluation Metrics

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

These metrics ensure:

* Accuracy
* Stability
* Interpretability trade-off

---

## 🚀 Deployment-Ready Design

This system is designed for:

* Batch inference
* API integration
* Dashboarding (Superset / Power BI)
* Monitoring and retraining

Recommended extensions:

* Data drift detection
* SHAP-based alerting
* Time-series validation

---

## 📜 Reproducibility & Ethics

✔ Fully deterministic training
✔ Explainable predictions
✔ Transparent decision logic
✔ Suitable for environmental governance

---

## 👤 Author

**Julius Okello**
Data Scientist | ML Engineer
Focus: AI for Environmental & Social Impact

---

## 📎 License

This project is provided for educational, research, and challenge participation purposes. It was used for submission towards the EY AI Data Challenge 2026 for Water Quality Prediction

---