# Lasso and Selective Inference: Credit Card Default Analysis

Seminar project for *Statistics with R* (Saarland University): comparing a statistically validated lasso model (with post-selection inference) against a Random Forest for predicting credit card default.

## Project Overview

This project analyzes the Default of Credit Card Clients dataset from Taiwan (Yeh & Lien, 2009) to answer one main question: **which variables does the lasso pick as important, which of them stay significant after adjusting for selective inference, and how does this list compare to a Random Forest's ranking?**

It builds 105 engineered predictors from six months of repayment history, bill statements and payment amounts, fits a lasso logistic regression, applies exact polyhedral post-selection inference (Lee et al., 2016), and compares the result with Random Forest variable importance under class imbalance.

**Dataset Source:** [Kaggle - Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) / UCI Machine Learning Repository

**Full write-up:** [`NB_SeminarPaper.pdf`](NB_SeminarPaper.pdf)

## Dataset Description

- 30,000 credit card clients, 23 original predictors
- Binary target: `default payment next month` (about 22% defaulted)
- Predictors: demographics (credit limit, sex, education, marital status, age), six months of repayment status (`PAY_0` to `PAY_6`), bill statement amounts (`BILL_AMT1` to `BILL_AMT6`) and previous payment amounts (`PAY_AMT1` to `PAY_AMT6`), April to September 2005

## Research Questions

1. Do the lasso and the Random Forest agree on which variables matter most for predicting default?
2. The monthly variables are highly correlated. Does the lasso pick just one month while the Random Forest spreads importance across all months, and does selective inference reflect this with wider confidence intervals?
3. Does adjusting for selective inference change which variables count as "significant" compared to a naive analysis?

## Work Description

### 1. Data Preparation

- Verified unique IDs and zero missing values
- Recoded undocumented codes in `EDUCATION` (0, 5, 6) and `MARRIAGE` (0) into the "other" category instead of deleting rows
- Kept negative bill amounts, since they represent valid credit balances
- Stratified 70/30 train/test split (21,000 / 9,000) done **before** feature engineering to prevent data leakage; all cutoffs and scaling parameters come from the training set only

### 2. Feature Engineering

Built with a single function across four areas, keeping the original monthly variables next to the new summaries:

- **Demographics:** age groups and credit-limit tiers
- **Repayment status:** average/maximum delay, delay trend and momentum, flags for serious delinquency
- **Bill amounts:** monthly credit utilization ratios, months with negative balance, "ever maxed out" flag
- **Payment amounts:** payment-to-bill ratios, paid-in-full flags, payment summaries

Result: **105 candidate predictors**.

### 3. Lasso and Selective Inference

- Lasso logistic regression with 10-fold cross-validated lambda (58 active features)
- Post-selection inference with `fixedLassoInf()` from the `selectiveInference` package (`type = "partial"`)
- Comparison against a naive unpenalized `glm` on the same 58 variables

### 4. Random Forest

- Two forests with 500 trees each: unadjusted (natural class distribution) and class-balanced (downsampled bootstrap)
- Variable importance measured by Mean Decrease Accuracy

## Key Results

| Finding | Result |
|---|---|
| Lasso-selected variables | 58 of 105 |
| Still significant after selective inference (p < 0.05) | 24 of 58 |
| Rank correlation, lasso vs. balanced RF | ρ = 0.642 |
| Top-10 overlap, lasso vs. balanced RF | 1 of 10 |
| Mean CI width, monthly variables vs. other variables | 1.05 vs. 0.78 |

**Out-of-sample performance (test set, n = 9,000):**

| Model | Test AUC | 95% CI |
|---|---|---|
| Lasso logistic regression | 0.775 | [0.763, 0.787] |
| Random Forest (unadjusted) | 0.761 | [0.748, 0.773] |
| Random Forest (class-balanced) | 0.773 | [0.761, 0.785] |

**Takeaways:**
- Past payment behavior (`avg_delay`, `ever_seriously_delinquent`, `PAY_*`) is the strongest driver of default.
- Naive inference is overly optimistic: 8 variables lose significance after correction, and the corrected confidence intervals are wider for every variable.
- The lasso picks representatives from correlated monthly blocks, while the Random Forest spreads importance across all months.
- The lasso is slightly more accurate than both forests and is far easier to interpret.

## Repository Structure

```
├── README.md
├── NB_SeminarPaper.pdf        # Full seminar paper
├── NB_Script.R                # Complete R analysis script
├── UCI_Credit_Card.csv        # Dataset
└── figures/                   # Plots produced by the script
```

## How to Run

1. Clone the repository and open `NonaBrutyan_Script.R` in RStudio.
2. Install the required packages:
   ```r
   install.packages(c("dplyr", "caret", "tidyr", "purrr", "glmnet",
                      "selectiveInference", "randomForest", "pROC"))
   ```
3. Run the script from top to bottom. A seed (`set.seed(42)`) is set, so results are reproducible.

## Technologies Used

- **R** (dplyr, tidyr, purrr, caret)
- **Modeling:** glmnet (lasso), randomForest
- **Inference:** selectiveInference
- **Evaluation:** pROC

## References

- Lee, J. D., Sun, D. L., Sun, Y., & Taylor, J. E. (2016). Exact post-selection inference, with application to the lasso. *Annals of Statistics*, 44(3), 907-927.
- Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
- Stelzer, A. (2019). Predicting credit default probabilities using machine learning techniques in the face of unequal class distributions. arXiv:1907.12996.
- Kammer, M., Dunkler, D., Michiels, S., & Heinze, G. (2022). Evaluating methods for Lasso selective inference in biomedical research. *BMC Medical Research Methodology*, 22, 206.
