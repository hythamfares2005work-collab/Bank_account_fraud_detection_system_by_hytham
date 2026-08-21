# Bank Account Fraud Detection

An end-to-end exploratory data analysis and machine learning project built on the **Bank Account Fraud (BAF) dataset** (NeurIPS 2022), covering data quality checks, SQL-driven business analysis, statistical hypothesis testing, visualization, preprocessing, and XGBoost modeling for detecting fraudulent bank account applications.

> **Status:** Pre-MLOps phase complete. MLOps phase (experiment tracking, API serving, containerization, CI/CD) in progress.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Modeling Results](#modeling-results)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Roadmap](#roadmap)

---

## Dataset

This project uses the **Base** variant of the [Bank Account Fraud (BAF) dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), a synthetic, privacy-preserving suite of tabular datasets introduced at NeurIPS 2022, designed to reflect realistic fraud-detection challenges (severe class imbalance, temporal drift, and group fairness considerations).

| | |
|---|---|
| Rows | ~1,000,000 |
| Columns | 32 |
| Target | `fraud_bool` |
| Fraud rate | ~1.1% (severe class imbalance) |
| Missing values | None |
| Duplicate rows | None |

Features include applicant demographics and income, address history, device and session metadata, credit risk score, payment type, and banking relationship signals.

> **Note:** `BAF.csv` is not included in this repository due to its size (~200MB). Download it from Kaggle and place it in the project's data directory before running the notebooks.

---

## Project Structure

```
.
├── notebooks/
│   ├── 01_health_checks.ipynb        # Data quality checks
│   ├── 02_business_questions.ipynb   # SQL analysis + hypothesis testing
│   ├── 03_visualization.ipynb        # EDA visualizations + outlier detection
│   ├── 04_data_preprocessing.ipynb   # Encoding & preprocessing
│   └── 05_modeling.ipynb             # XGBoost modeling & experiment tracking
├── data/
│   ├── BAF.csv                       # Raw dataset (not tracked, download separately)
│   └── preprocessed_data.csv         # Output of preprocessing notebook
└── README.md
```

### 1. `01_health_checks.ipynb`
Baseline data quality checks: null values, duplicate rows, disguised nulls (e.g. `"n/a"`, `"unknown"` strings), datatype validation, and a first look at the severity of class imbalance in the target variable.

### 2. `02_business_questions.ipynb`
Loads the dataset into a local **PostgreSQL** database and answers ten business questions entirely in SQL — window functions, CTEs, and aggregate filtering — covering fraud rate by employment status, credit risk score, payment type, device OS, session length, and more. Includes a **two-proportion z-test** (via `statsmodels`) to confirm whether the difference in fraud rate between foreign and domestic requests is statistically significant.

### 3. `03_visualization.ipynb`
Visualizes the class imbalance, then performs:
- **Univariate outlier detection** via per-feature histograms
- **Multivariate outlier detection** via PCA (2-component projection)

### 4. `04_data_preprocessing.ipynb`
Encodes categorical columns with `LabelEncoder` and exports the model-ready dataset as `preprocessed_data.csv`.

### 5. `05_modeling.ipynb`
Trains an **XGBoost classifier** and compares two strategies for handling class imbalance — **SMOTE** oversampling vs. XGBoost's built-in **`scale_pos_weight`** — evaluated on ROC-AUC and PR-AUC. Includes an in-progress **MLflow** experiment tracking loop.

---

## Key Findings

- **Employment status:** customers with employment status `CC` have the highest fraud rate (~2.47%), notably higher than all other categories.
- **Credit risk score:** fraud rate increases with credit risk score, with a clear spike starting once the score crosses into the higher bins.
- **Payment type:** `AC` has the highest fraud rate, with `AB` and `AD` close behind.
- **Foreign vs. domestic requests:** foreign requests have a fraud rate roughly double that of domestic requests (2.2% vs 1.1%), and a two-proportion z-test confirms this difference is statistically significant (z ≈ -16.88).
- **Device OS:** sessions on "other" and Linux operating systems are most frequently associated with fraud.
- **Email type:** customers using free email providers have a fraud rate of ~1.38%, versus ~0.80% for paid/corporate emails.
- **Seasonality:** month 8 has the highest fraud rate; month 3 has the lowest.
- **Outliers:** no extreme multivariate outliers were found via PCA — fraud cases are broadly scattered through the data, and the 2-component projection only explains ~30% of total variance, underscoring that this is a genuinely hard classification problem rather than one driven by a few anomalous points.

---

## Modeling Results

Two approaches were compared for handling the ~1.1% positive class rate:

| Approach | Train ROC-AUC | Test ROC-AUC | Train PR-AUC | Test PR-AUC |
|---|---|---|---|---|
| SMOTE oversampling | 0.985 | 0.828 | 0.986 | 0.066 |
| `scale_pos_weight` | 0.903 | 0.884 | 0.166 | 0.135 |

**`scale_pos_weight` was the clear winner.** SMOTE caused severe overfitting — the model memorized the synthetic minority samples (Train PR-AUC of 0.99 collapsing to a Test PR-AUC of 0.07) — while `scale_pos_weight` generalized much better, with only mild overfitting and a higher score on both metrics on held-out data.

---

## Tech Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Database:** PostgreSQL (via `psycopg2`, `SQLAlchemy`)
- **Statistics:** `statsmodels` (proportions z-test)
- **ML:** Scikit-learn, XGBoost, `imbalanced-learn` (SMOTE)
- **Experiment tracking:** MLflow *(in progress)*

---

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn mlflow statsmodels psycopg2-binary sqlalchemy
   ```

3. **Download the dataset**
   Get `Base.csv` from the [BAF dataset on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) and place it in `data/BAF.csv`.

4. **Set up PostgreSQL** (required for `02_business_questions.ipynb`)
   Create a local database and update the connection parameters (`host`, `dbname`, `user`, `password`, `port`) in the notebook to match your setup.

5. **Run the notebooks in order**
   ```
   01_health_checks.ipynb → 02_business_questions.ipynb → 03_visualization.ipynb → 04_data_preprocessing.ipynb → 05_modeling.ipynb
   ```

---

## Roadmap

The next phase of this project introduces a full MLOps stack:

- [ ] **MLflow** — experiment tracking and model registry *(in progress)*
- [ ] **FastAPI** — serve the trained model as a REST API
- [ ] **Docker** — containerize the training and serving pipeline
- [ ] **GitHub Actions** — CI/CD for automated testing and deployment

---

## License

Add a license of your choice (e.g. MIT) here.
