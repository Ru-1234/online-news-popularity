# Predicting Online News Popularity

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

Machine learning and statistical analysis project using the UCI Online News Popularity dataset to predict article virality through regression and classification models.

Developed for **Probability & Statistics (MT-2005)** at FAST NUCES Islamabad, Spring 2026.

---

## Project Overview

This project analyzes approximately 39,000 online news articles published by Mashable and investigates whether article popularity can be predicted before publication.

The project combines:

- Statistical preprocessing
- Exploratory Data Analysis (EDA)
- Hypothesis testing
- Regression modeling
- Classification modeling
- Diagnostic analysis

The dataset presents several real-world machine learning challenges including extreme skewness, multicollinearity, weak feature correlations, and large outlier presence.

---

## Dataset

Dataset: Online News Popularity Dataset from the UCI Machine Learning Repository

- ~39,000 news articles
- 58 numerical input features
- Target variable: `shares`
- Source: Mashable

### Features Include

- Article structure metrics
- Keyword statistics
- Topic distributions
- Sentiment scores
- Content channel indicators
- Day-of-week publication indicators

---

## Data Challenges

| Issue | Handling |
|---|---|
| Extreme skewness in `shares` | Applied `log(1 + shares)` transformation |
| Highly correlated feature pairs | Removed features with correlation > 0.90 |
| Invalid token ratio values (>1.0) | Removed erroneous rows |
| Non-informative columns (`url`, `timedelta`) | Dropped during preprocessing |
| Multicollinearity in LDA features | Diagnosed using VIF |
| Large outlier presence | Retained due to real-world significance |

---

## Preprocessing

- Dropped `url` and `timedelta`
- Removed rows with invalid ratio features
- Checked for zero-variance features
- Removed highly correlated predictors
- Standardized features using Z-score normalization
- Performed train-test split (`80/20`, `random_state=42`)

### Final Dataset

- **39,643 articles**
- **56 features**

---

## Exploratory Data Analysis

EDA included:

- Distribution analysis
- Skewness and kurtosis analysis
- Correlation heatmaps
- Scatter plots
- Channel-wise popularity comparisons
- Day-of-week popularity comparisons

### Key Observation

No individual feature had strong predictive power (`max |r| ≈ 0.11`), indicating that online virality is driven by complex and noisy interactions rather than isolated article characteristics.

---

## Feature Engineering

### Regression Target

```python
log_shares = log(1 + shares)
````

Used to reduce heavy right-skewness in the target distribution.

### Classification Target

```python
share_class = 1 if shares >= median(shares) else 0
```

* High Popularity → 1
* Low Popularity → 0

Median threshold was chosen over mean to avoid severe class imbalance caused by viral outliers.

---

## Models

| Model               | Task           | Performance                |
| ------------------- | -------------- | -------------------------- |
| Mean Baseline       | Regression     | RMSE = 0.9261              |
| Linear Regression   | Regression     | RMSE = 0.8648, R² = 0.1278 |
| Majority Baseline   | Classification | Accuracy = 52.60%          |
| Logistic Regression | Classification | Accuracy = 64.69%          |

---

## Hypothesis Testing

All tests used significance level:

```text
α = 0.05
```

| Question                                      | Test                | p-value    | Decision          |
| --------------------------------------------- | ------------------- | ---------- | ----------------- |
| Do weekend articles receive different shares? | Welch’s t-test      | 0.000321   | Reject H₀         |
| Is sentiment polarity correlated with shares? | Pearson correlation | 0.406433   | Fail to Reject H₀ |
| Do articles with images receive more shares?  | Welch’s t-test      | ≈ 0.000000 | Reject H₀         |
| Do Tech articles outperform World articles?   | Welch’s t-test      | ≈ 0.000000 | Reject H₀         |
| Is article word count correlated with shares? | Pearson correlation | 0.000001   | Reject H₀         |

---

## Key Findings

* `kw_avg_avg` was the strongest predictor of article popularity.
* Weekend articles received significantly higher engagement.
* Articles containing images achieved higher average shares.
* Tech articles significantly outperformed World news articles.
* Sentiment polarity showed no statistically significant relationship with virality.
* Logistic Regression outperformed the majority baseline by +12.09%.
* Online news popularity remains highly noisy and difficult to predict.

---

## Model Diagnostics

Performed diagnostic analysis including:

* Variance Inflation Factor (VIF)
* Residual vs fitted analysis
* Q-Q plots
* Residual histograms
* Actual vs predicted visualization
* Outlier detection using IQR and Z-score methods

### Important Observation

LDA topic features produced infinite VIF values because topic probabilities sum to 1, creating an exact linear dependency.

---

## Reproducibility

* Random seed: `42`
* Train/test split: `80/20`
* Scaling fitted only on training data to prevent leakage

---

## Repository Structure

```text
online-news-popularity/

├── OnlineNewsPopularity.csv
├── OnlineNewsPopularity.names
├── README.md
├── project.ipynb
├── report.pdf
├── SemesterProject.pdf
├── requirements.txt
```

---

## Getting Started

### Clone Repository

```bash
git clone https://github.com/Ru-1234/online-news-popularity.git
cd online-news-popularity
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Notebook

```bash
jupyter notebook project.ipynb
```

---

## Tech Stack

| Library      | Purpose                   |
| ------------ | ------------------------- |
| pandas       | Data manipulation         |
| numpy        | Numerical computation     |
| matplotlib   | Visualization             |
| seaborn      | Statistical visualization |
| scikit-learn | Machine learning models   |
| scipy        | Hypothesis testing        |
| statsmodels  | Statistical diagnostics   |

---

## References

Fernandes, K., Vinagre, P., & Cortez, P. (2015).

*A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News.*

UCI Machine Learning Repository.

---

## Authors

* Maham Anjum
* Romaisa

FAST NUCES Islamabad — Spring 2026

---

## License

This project is licensed under the MIT License.
