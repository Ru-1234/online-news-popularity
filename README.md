# 📰 Predicting Online News Popularity

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Course](https://img.shields.io/badge/Course-Probability%20%26%20Statistics-orange)
![University](https://img.shields.io/badge/FAST-NUCES%20Islamabad-red)

> *Can we predict whether a news article will go viral — before it does?*
> This project attempts to answer that question using statistical modeling,
> hypothesis testing, and machine learning on real-world messy data.

---

## 🎓 About

This project was developed as a **semester project** for:

| | |
|---|---|
| **Course** | Probability & Statistics (MT-2005) |
| **University** | FAST NUCES Islamabad |
| **Semester** | Spring 2026 |

### 👩‍💻 Developed By

Maham Anjum & Romaisa

---

## 📦 The Dataset

**UCI Online News Popularity Dataset**
- ~39,000 news articles scraped from **Mashable**
- 58 numerical input features
- Target variable: `shares` (how many times the article was shared on social media)

### 😤 Why This Dataset Was a Nightmare

This dataset looked clean on the surface but was full of statistical landmines:

#### 🔴 Extreme Skewness
The `shares` column had skewness exceeding 10. A tiny fraction of articles went
viral (100,000+ shares) while the vast majority got under 2,000. This meant raw
`shares` was completely unusable as a regression target — a plain linear model
would be pulled entirely by outliers. We applied a **log(1+x) transformation**
to bring the distribution closer to normal before modeling.

#### 🔴 Highly Correlated Feature Pairs
Several feature pairs turned out to carry nearly identical information:

| Feature 1 | Feature 2 | Correlation |
|---|---|---|
| `n_unique_tokens` | `n_non_stop_unique_tokens` | 0.938 |
| `n_non_stop_words` | `average_token_length` | 0.944 |
| `kw_max_min` | `kw_avg_min` | 0.941 |

All three exceeded our 0.90 threshold and one from each pair was dropped to
reduce redundancy and multicollinearity.

#### 🔴 Out-of-Range Token Ratios
Features like `n_unique_tokens` and `n_non_stop_unique_tokens` are defined as
ratios bounded between 0 and 1. A small number of rows had values exceeding 1.0,
indicating data collection errors — these rows were removed during preprocessing.

#### 🔴 Non-Informative Columns
`url` (just an article identifier) and `timedelta` (days since data collection,
not a property of the article itself) were dropped immediately as they carry
zero predictive value.

#### 🔴 Weak Individual Correlations
No single feature correlated with `shares` above **0.11**. This is a known
property of this dataset — news virality is driven by complex social and
contextual factors no single numerical feature can capture alone.

#### 🔴 Infinite VIF on LDA Features
LDA topic scores (LDA_00 to LDA_04) must sum to 1 by definition, creating an
exact linear dependency that produces infinite VIF values. Day-of-week binary
flags have a similar near-perfect constraint (an article is published on exactly
one day), causing mild multicollinearity across all seven indicators.

#### 🔴 Massive Outlier Presence
IQR-based outlier detection flagged 11.46% of articles as outliers in raw
`shares`. Outliers were **retained** — viral articles represent genuine
real-world events and carry authentic signal about extreme popularity. Removing
them would bias the model toward ordinary content.

---

## 🔬 What We Did

### 🧹 1. Preprocessing
- Dropped `url` and `timedelta`
- Removed rows with shares ≤ 0 (physically impossible — 0 rows affected)
- Removed 1 row where token ratio features exceeded valid range (> 1.0)
- Checked for zero-variance features — none found
- Dropped one feature from each highly correlated pair (threshold > 0.90)
- Final dataset: **39,643 articles, 56 features**

### 📊 2. Exploratory Data Analysis
- Descriptive statistics with skewness, kurtosis, median, IQR
- Distribution histograms for 6 key variables with mean/median overlaid
- Pearson correlation heatmap across 18 selected features
- **Data-driven** scatter plots: top 7 features by absolute correlation with log(shares)
- Median shares broken down by content channel and day of week

### 📐 3. Feature Engineering & Normalization
- `log_shares` = log(1 + shares) as regression target (reduces skewness)
- Binary `share_class`: High (1) if shares ≥ 1,400 (median), Low (0) otherwise
- Median threshold chosen over mean — mean (≈3,395) is inflated by outliers and
  would create severe class imbalance; median produces a balanced 53/47 split
- 80/20 train-test split with `random_state=42`
- Z-score standardization fitted on training data only (prevents data leakage)

### 🤖 4. Modeling

| Model | Task | Metric |
|---|---|---|
| Mean Baseline | Regression | RMSE = 0.9261, R² ≈ 0.00 |
| **Linear Regression** | Predict log(shares) | **RMSE = 0.8648, R² = 0.1278** |
| Majority Baseline | Classification | Accuracy = 52.60% |
| **Logistic Regression** | Classify High vs Low | **Accuracy = 64.69%** |

### 🧪 5. Hypothesis Testing

Five tests conducted at significance level **α = 0.05**:

| # | Question | Test Used | p-value | Decision |
|---|---|---|---|---|
| 1 | Do weekend articles get different shares? | Welch's t-test (two-tailed) | 0.000321 | Reject H₀ |
| 2 | Is sentiment polarity correlated with shares? | Pearson correlation | 0.406433 | Fail to Reject H₀ |
| 3 | Do articles with images get more shares? | Welch's t-test (one-tailed) | ≈ 0.000000 | Reject H₀ |
| 4 | Do Tech articles get more shares than World? | Welch's t-test (one-tailed) | ≈ 0.000000 | Reject H₀ |
| 5 | Is article word count correlated with shares? | Pearson correlation | 0.000001 | Reject H₀ |

4 out of 5 null hypotheses rejected. **Sentiment polarity (T2) was not significant**
(p = 0.406) — article tone alone does not drive virality.

### 🔍 6. Model Diagnostics & Validation
- **VIF analysis** — LDA features show infinite VIF (compositional constraint);
  day-of-week features show mild multicollinearity
- **IQR + z-score outlier detection** — 11.46% of articles flagged; retained intentionally
- **Regression diagnostics** — residuals vs fitted (mild heteroscedasticity), Q-Q plot
  (heavier tails than Gaussian), residual histogram
- **Statsmodels OLS** on 5,000-sample subset — 12 out of 55 features statistically
  significant (p < 0.05)
- **Actual vs Predicted plot** — wide scatter confirms modest but real predictive power

---

## 📈 Key Findings

- **Linear regression beats the mean baseline** across all metrics — RMSE drops 6.62%,
  R² rises from ~0 to 0.1278. The model explains ~13% of variance in log(shares),
  consistent with published research on this dataset.

- **`kw_avg_avg` is the strongest predictor** (coef = 0.405, p ≈ 6.27e-20) — articles
  covering topics that historically attract shares are themselves more likely to be shared.
  Past keyword popularity is the best available signal.

- **Sentiment polarity does not predict shares** (p = 0.406) — whether an article is
  positive or negative in tone has no statistically significant effect on virality.

- **Weekend publishing significantly increases shares** (p = 0.000321, Weekday median:
  1,400 vs Weekend median: 1,900) — audience browsing peaks on weekends.

- **Articles with images get significantly more shares** (p ≈ 0) — though both groups
  share the same median (1,400), suggesting extreme viral values drive the mean difference.

- **Tech channel outperforms World channel** (p ≈ 0, median 1,700 vs 1,100).

- **Logistic Regression achieves 64.69% accuracy** — +12.09 percentage points over the
  52.60% majority-class baseline. Higher recall on High class (0.69) than Low (0.60).

- **Virality is inherently noisy** — ~87% of variance in shares remains unexplained.
  External factors dominate but are absent from this dataset.

---

## 🗂️ Repository Structure
online-news-popularity/

├── OnlineNewsPopularity.csv       # Dataset ([UCI ML Repository](https://archive.ics.uci.edu/dataset/332/online+news+popularity))  
├── OnlineNewsPopularity.names     # Feature descriptions                  
├── README.md                                    
├── SemesterProject.pdf            # Original project brief provided by course instructor             
├── project.ipynb                  # Full annotated notebook with markdown explanations             
├── report.pdf                     # Our written project report               

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Ru-1234/online-news-popularity.git
cd online-news-popularity  

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy jupyter

# Launch the notebook
jupyter notebook project.ipynb
```

---

## 📚 Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` / `seaborn` | Visualizations |
| `scikit-learn` | Linear & Logistic Regression, preprocessing, metrics |
| `statsmodels` | OLS regression with p-values, VIF |
| `scipy` | Hypothesis testing (Pearson correlation, Welch's t-test) |

---

## 📖 References

- Fernandes, K., Vinagre, P., & Cortez, P. (2015). *A Proactive Intelligent Decision
  Support System for Predicting the Popularity of Online News.*
  [UCI ML Repository](https://archive.ics.uci.edu/dataset/332/online+news+popularity)
