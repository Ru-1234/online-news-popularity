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
ratios and should be bounded between 0 and 1. However, a small number of rows
had values exceeding 1.0 — indicating data collection errors. These rows were
removed during preprocessing.

#### 🔴 Non-Informative Columns
`url` (just an article identifier) and `timedelta` (days since data collection,
not a property of the article itself) were dropped immediately as they carry
zero predictive value for share count.

#### 🔴 Weak Individual Correlations
No single feature correlated with `shares` above **0.15**. The strongest
predictor (`kw_avg_avg`) only reached 0.11. This is a known property of this
dataset — news virality is driven by complex social and contextual factors that
no single numerical feature can capture alone.

#### 🔴 Massive Outlier Presence
IQR-based outlier detection flagged a large percentage of articles as outliers
in raw `shares`. The log transformation significantly reduced this, but the
boxplot of raw shares remains nearly unreadable — a testament to how extreme
the distribution is.

---

## 🔬 What We Did

### 🧹 1. Preprocessing
- Dropped `url` and `timedelta`
- Removed rows with shares ≤ 0 (physically impossible)
- Removed rows where token ratio features exceeded valid range (> 1.0)
- Identified and dropped zero-variance features (none found)
- Dropped one feature from each highly correlated pair (threshold > 0.90)

### 📊 2. Exploratory Data Analysis
- Descriptive statistics table with mean, std, skewness, kurtosis, median, IQR
- Distribution histograms for 6 key variables with mean/median overlaid
- Full correlation heatmap across 18 selected features
- **Data-driven** scatter plots: top 7 features by correlation with log(shares)
- Median shares broken down by content channel and day of week

### 📐 3. Feature Engineering & Normalization
- Created `log_shares` = log(1 + shares) as regression target
- Created binary `share_class` (1 = high, 0 = low) using median threshold
- 80/20 train-test split with fixed random seed
- Z-score standardization fitted on training set only (no data leakage)

### 🤖 4. Modeling

| Model | Task | Metric |
|---|---|---|
| Mean Baseline | Regression | RMSE = 0.9261, R² ≈ 0.00 |
| **Linear Regression** | Predict log(shares) | **RMSE = 0.8648, R² = 0.1278** |
| Majority Baseline | Classification | Accuracy ≈ 0.50 |
| **Logistic Regression** | Classify High vs Low | **Accuracy ≈ 0.63** |

### 🧪 5. Hypothesis Testing

We formulated and tested 5 statistical hypotheses at α = 0.05:

| # | Question | Test Used | Decision |
|---|---|---|---|
| 1 | Do weekend articles get different shares than weekday? | Mann-Whitney U | Reject H₀ |
| 2 | Is sentiment polarity correlated with shares? | Spearman correlation | Reject H₀ |
| 3 | Do articles with images get more shares? | Mann-Whitney U | Reject H₀ |
| 4 | Do Tech articles get more shares than World articles? | Mann-Whitney U | Reject H₀ |
| 5 | Is article word count correlated with shares? | Pearson correlation | Reject H₀ |

All 5 null hypotheses were rejected — each feature tested has a statistically
significant relationship with article popularity.

### 🔍 6. Model Diagnostics & Validation
- **VIF analysis** — checked all features for multicollinearity after dropping correlated pairs
- **IQR outlier detection** — quantified outlier percentage in raw shares
- **Regression diagnostics** — residuals vs fitted, Q-Q plot, residual histogram
- **Statsmodels OLS** — obtained p-values for all coefficients to identify statistically significant predictors
- **Actual vs Predicted plot** — visual assessment of regression fit

---

## 📈 Key Findings

- **Linear regression meaningfully beats the mean baseline** (RMSE drops 6.6%,
  R² rises from 0 to 0.1278) but the modest R² confirms that ~87% of variance
  in shares is unexplained — consistent with published research on this dataset.

- **`kw_avg_avg`** (average shares of articles covering similar keywords) is the
  strongest predictor — the topic's past popularity is the best signal for future virality.

- **Weekend publishing** leads to significantly lower shares than weekday publishing,
  suggesting audience behavior plays a major role.

- **Articles with images** get significantly more shares than those without.

- **Tech channel** articles consistently outperform World channel articles in shares.

- **Virality is inherently non-linear** — a linear model is a useful starting point
  but has a hard ceiling on this dataset. Tree-based models would likely perform significantly better.

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
cd viral-or-void

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
| `scipy` | Hypothesis testing (Pearson, Spearman, Mann-Whitney) |

---

## 📖 References

- Fernandes, K., Vinagre, P., & Cortez, P. (2015). *A Proactive Intelligent Decision
  Support System for Predicting the Popularity of Online News.*
  [UCI ML Repository](https://archive.ics.uci.edu/dataset/332/online+news+popularity)
