########################################################################
# STEP 1: Installing necessary packages and downloading data
########################################################################

install.packages("dplyr")
install.packages("caret")
install.packages("tidyr")
install.packages("purrr")
install.packages("glmnet")
install.packages("selectiveInference")
install.packages("randomForest")
install.packages("pROC")



library(dplyr)
library(caret)
library(tidyr)
library(purrr)
library(glmnet)
library(selectiveInference)
library(randomForest)
library(pROC)


raw <- read.csv("~/Downloads/UCI_Credit_Card.csv", stringsAsFactors = FALSE)

########################################################################
# STEP 2: Data Preparation
########################################################################

# 2.1 Data checks and cleaning

# unique clients check
sum(duplicated(raw$ID)) 

# missing values check
sum(is.na(raw)) 

# undocumented codes 0, 5, 6
table(raw$EDUCATION) 

# undocumented code 0
table(raw$MARRIAGE) 

# class balance
table(raw$default.payment.next.month) / nrow(raw)   

# negative BILL_AMT values 
sapply(raw[, c("BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6")],
       function(x) sum(x < 0))

# 2.2 Train/test split

set.seed(42)
train_index <- createDataPartition(raw$default.payment.next.month, p = 0.7, list = FALSE)
train_raw <- raw[train_index, ]
test_raw  <- raw[-train_index, ]


########################################################################
# STEP 3: Feature engineering (105 features)
########################################################################


pay_cols  <- c("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
bill_cols <- c("BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
               "BILL_AMT4", "BILL_AMT5", "BILL_AMT6")
paid_cols <- c("PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
               "PAY_AMT4", "PAY_AMT5", "PAY_AMT6")

month_index_for_trend <- 1:6

#linear models slope function, to find the direction of moving
slope <- function(row) coef(lm(row ~ month_index_for_trend))[2] 

engineer_features <- function(raw_split, limit_breaks) {
  
  
  # 3.0 Cleanup of undocumented category codes
  
  
  df <- raw_split %>%
    rename(default = `default.payment.next.month`) %>%
    mutate(
      EDUCATION = case_when(EDUCATION %in% c(0, 5, 6) ~ 4L, TRUE ~ EDUCATION),
      MARRIAGE  = case_when(MARRIAGE == 0 ~ 3L, TRUE ~ MARRIAGE),
      SEX = factor(SEX, levels = c(1, 2), labels = c("male", "female")),
      EDUCATION = factor(EDUCATION, levels = c(1, 2, 3, 4),
                         labels = c("grad_school", "university",
                                    "high_school", "other")),
      MARRIAGE = factor(MARRIAGE, levels = c(1, 2, 3),
                        labels = c("married", "single", "other"))
    )
  
  pay_mat  <- as.matrix(df[, pay_cols])
  bill_mat <- as.matrix(df[, bill_cols])
  paid_mat <- as.matrix(df[, paid_cols])
  limit    <- pmax(df$LIMIT_BAL, 1)
  
  
  # 3.1 Demographics
 
  
  df <- df %>%
    mutate(
      age_group = factor(case_when(
        AGE < 25 ~ "young_<25",
        AGE >= 25 & AGE < 40 ~ "adult_25_39",
        AGE >= 40 & AGE < 60 ~ "midlife_40_59",
        AGE >= 60 ~ "senior_60plus"
      ), levels = c("young_<25", "adult_25_39", "midlife_40_59", "senior_60plus")),
      
      # fit on train, applied to test
      limit_tier = factor(
        cut(LIMIT_BAL, breaks = limit_breaks,
            labels = c("low_limit", "mid_limit", "high_limit"),
            include.lowest = TRUE)
      )
    )
  
 
  # 3.2 Repayment status features

  
  df <- df %>%
    mutate(
      avg_delay                = rowMeans(pmax(pay_mat, 0)),
      max_delay                = apply(pay_mat, 1, max),
      min_delay                = apply(pay_mat, 1, min),
      delay_range               = max_delay - min_delay,
      n_months_delayed          = rowSums(pay_mat >= 1),
      n_months_no_consumption   = rowSums(pay_mat == -2),
      recent_delay_status       = PAY_0,
      delay_trend               = apply(pay_mat, 1, slope),
      always_paid_on_time       = as.integer(max_delay <= 0),
      ever_seriously_delinquent = as.integer(max_delay >= 2),
      
      recent_avg_delay = rowMeans(pmax(pay_mat[, 1:3], 0)),
      older_avg_delay   = rowMeans(pmax(pay_mat[, 4:6], 0)),
      delay_momentum    = recent_avg_delay - older_avg_delay,
      
      delta_delay_m1_m2 = pay_mat[, 1] - pay_mat[, 2],
      delta_delay_m2_m3 = pay_mat[, 2] - pay_mat[, 3],
      delta_delay_m3_m4 = pay_mat[, 3] - pay_mat[, 4],
      delta_delay_m4_m5 = pay_mat[, 4] - pay_mat[, 5],
      delta_delay_m5_m6 = pay_mat[, 5] - pay_mat[, 6]
    )
  
  
  # 3.3 Bill amount features
 
  
  util_mat <- bill_mat / limit
  colnames(util_mat) <- paste0("util_m", 1:6)
  
  df <- df %>%
    mutate(
      avg_bill_amt   = rowMeans(bill_mat),
      max_bill_amt   = apply(bill_mat, 1, max),
      min_bill_amt   = apply(bill_mat, 1, min),
      bill_range     = max_bill_amt - min_bill_amt,
      bill_amt_sd    = apply(bill_mat, 1, sd),
      bill_trend     = apply(bill_mat, 1, slope),
      n_months_negative_bill = rowSums(bill_mat < 0),
      
      util_avg   = rowMeans(util_mat),
      util_sd    = apply(util_mat, 1, sd),
      util_trend = apply(util_mat, 1, slope),
      ever_maxed_out = as.integer(apply(util_mat, 1, max) > 0.9),
      
      recent_avg_bill = rowMeans(bill_mat[, 1:3]),
      older_avg_bill  = rowMeans(bill_mat[, 4:6]),
      bill_momentum   = recent_avg_bill - older_avg_bill,
      
      recent_avg_util = rowMeans(util_mat[, 1:3]),
      older_avg_util  = rowMeans(util_mat[, 4:6]),
      util_momentum   = recent_avg_util - older_avg_util,
      
      delta_bill_m1_m2 = bill_mat[, 1] - bill_mat[, 2],
      delta_bill_m2_m3 = bill_mat[, 2] - bill_mat[, 3],
      delta_bill_m3_m4 = bill_mat[, 3] - bill_mat[, 4],
      delta_bill_m4_m5 = bill_mat[, 4] - bill_mat[, 5],
      delta_bill_m5_m6 = bill_mat[, 5] - bill_mat[, 6],
      
      delta_util_m1_m2 = util_mat[, 1] - util_mat[, 2],
      delta_util_m2_m3 = util_mat[, 2] - util_mat[, 3],
      delta_util_m3_m4 = util_mat[, 3] - util_mat[, 4],
      delta_util_m4_m5 = util_mat[, 4] - util_mat[, 5],
      delta_util_m5_m6 = util_mat[, 5] - util_mat[, 6]
    ) %>%
    bind_cols(as.data.frame(util_mat))
  
  
  # 3.4 Payment amount features
  
  
  pay_to_bill_ratio <- sapply(1:5, function(i) {
    paid_mat[, i] / pmax(bill_mat[, i + 1], 1)
  })
  colnames(pay_to_bill_ratio) <- paste0("pay_to_bill_ratio_m", 1:5)
  
  paid_in_full <- sapply(1:5, function(i) {
    as.integer(paid_mat[, i] >= bill_mat[, i + 1])
  })
  colnames(paid_in_full) <- paste0("paid_in_full_m", 1:5)
  
  df <- df %>%
    mutate(
      avg_paid_amt   = rowMeans(paid_mat),
      max_paid_amt   = apply(paid_mat, 1, max),
      min_paid_amt   = apply(paid_mat, 1, min),
      paid_amt_sd    = apply(paid_mat, 1, sd),
      total_paid     = rowSums(paid_mat),
      total_billed   = rowSums(bill_mat),
      overall_repayment_ratio = total_paid / pmax(abs(total_billed), 1),
      n_months_zero_payment   = rowSums(paid_mat == 0),
      
      recent_avg_payment = rowMeans(paid_mat[, 1:3]),
      older_avg_payment  = rowMeans(paid_mat[, 4:6]),
      payment_momentum   = recent_avg_payment - older_avg_payment,
      
      delta_payment_m1_m2 = paid_mat[, 1] - paid_mat[, 2],
      delta_payment_m2_m3 = paid_mat[, 2] - paid_mat[, 3],
      delta_payment_m3_m4 = paid_mat[, 3] - paid_mat[, 4],
      delta_payment_m4_m5 = paid_mat[, 4] - paid_mat[, 5],
      delta_payment_m5_m6 = paid_mat[, 5] - paid_mat[, 6]
    ) %>%
    bind_cols(as.data.frame(pay_to_bill_ratio)) %>%
    bind_cols(as.data.frame(paid_in_full)) %>%
    mutate(
      avg_pay_to_bill_ratio = rowMeans(pay_to_bill_ratio, na.rm = TRUE),
      pay_to_bill_ratio_sd  = apply(pay_to_bill_ratio, 1, sd),
      n_months_paid_in_full = rowSums(paid_in_full)
    )
  
 
  # 3.5 Final feature set

  
  feature_cols <- c(
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "age_group", "limit_tier",
    pay_cols,
    "avg_delay", "max_delay", "min_delay", "delay_range",
    "n_months_delayed", "n_months_no_consumption", "recent_delay_status",
    "delay_trend", "always_paid_on_time", "ever_seriously_delinquent",
    "recent_avg_delay", "older_avg_delay", "delay_momentum",
    "delta_delay_m1_m2", "delta_delay_m2_m3", "delta_delay_m3_m4",
    "delta_delay_m4_m5", "delta_delay_m5_m6",
    bill_cols,
    "avg_bill_amt", "max_bill_amt", "min_bill_amt", "bill_range",
    "bill_amt_sd", "bill_trend", "n_months_negative_bill",
    "util_avg", "util_sd", "util_trend", "ever_maxed_out",
    paste0("util_m", 1:6),
    "recent_avg_bill", "older_avg_bill", "bill_momentum",
    "recent_avg_util", "older_avg_util", "util_momentum",
    "delta_bill_m1_m2", "delta_bill_m2_m3", "delta_bill_m3_m4",
    "delta_bill_m4_m5", "delta_bill_m5_m6",
    "delta_util_m1_m2", "delta_util_m2_m3", "delta_util_m3_m4",
    "delta_util_m4_m5", "delta_util_m5_m6",
    paid_cols,
    "avg_paid_amt", "max_paid_amt", "min_paid_amt", "paid_amt_sd",
    "total_paid", "total_billed", "overall_repayment_ratio",
    "n_months_zero_payment",
    "recent_avg_payment", "older_avg_payment", "payment_momentum",
    "delta_payment_m1_m2", "delta_payment_m2_m3", "delta_payment_m3_m4",
    "delta_payment_m4_m5", "delta_payment_m5_m6",
    paste0("pay_to_bill_ratio_m", 1:5),
    "avg_pay_to_bill_ratio", "pay_to_bill_ratio_sd",
    paste0("paid_in_full_m", 1:5),
    "n_months_paid_in_full",
    "default"
  )
  
  df %>% dplyr::select(ID, all_of(feature_cols))
}

# fit limit_tier breakpoints on train only, then apply to both splits
limit_breaks <- quantile(train_raw$LIMIT_BAL, probs = c(0, 1/3, 2/3, 1))

train_final <- engineer_features(train_raw, limit_breaks)
test_final  <- engineer_features(test_raw,  limit_breaks)


n_predictors <- ncol(train_final) - 2  # minus ID and default
cat("Final feature set:", n_predictors, "predictors (+ ID, + outcome)\n")
cat("Train default rate:", round(mean(train_final$default), 4), "\n")
cat("Test default rate:", round(mean(test_final$default), 4), "\n")


########################################################################
# STEP 4: Preparing for lasso / selective inference
########################################################################


# 4.1 Categorical encoding (SEX, EDUCATION, MARRIAGE, age_group, limit_tier)
predictors_train <- train_final %>% dplyr::select(-ID, -default)
predictors_test  <- test_final  %>% dplyr::select(-ID, -default)

X_train_full <- model.matrix(~ . - 1, data = predictors_train)
X_test_full  <- model.matrix(~ . - 1, data = predictors_test)
X_test_full  <- X_test_full[, colnames(X_train_full)]  # same column order

y_train <- train_final$default
y_test  <- test_final$default

# 4.2 Check for linear dependence among columns 
find_linear_combos <- function(X, tol = 1e-7) {
  qr_x <- qr(X, tol = tol)
  rank_x <- qr_x$rank
  if (rank_x == ncol(X)) return(integer(0))
  sort(qr_x$pivot[(rank_x + 1):ncol(X)])
}

dependent_idx <- find_linear_combos(X_train_full)
dependent_cols <- colnames(X_train_full)[dependent_idx]
cat("Linearly dependent columns dropped:", length(dependent_cols), "\n")
print(dependent_cols)

if (length(dependent_cols) > 0) {
  X_train_full <- X_train_full[, -dependent_idx]
  X_test_full  <- X_test_full[, colnames(X_train_full)]
}

# 4.3 Correlation heatmap of the monthly blocks (train only) 
monthly_cols <- c(pay_cols, bill_cols, paid_cols, paste0("util_m", 1:6))
monthly_cols <- intersect(monthly_cols, colnames(X_train_full))
cor_monthly <- cor(X_train_full[, monthly_cols])


heatmap(cor_monthly, symm = TRUE, Rowv = NA, Colv = NA,
        main = "Correlation: monthly blocks")

# 4.4 Feature standardization 
train_center <- colMeans(X_train_full)
train_scale  <- apply(X_train_full, 2, sd)
train_scale[train_scale == 0] <- 1  # preventing division by 0

X_train_scaled <- scale(X_train_full, center = train_center, scale = train_scale)
X_test_scaled  <- scale(X_test_full,  center = train_center, scale = train_scale)

########################################################################
# STEP 5: LASSO and Selective Inference 
########################################################################

# 5.1 LASSO on fixed lambda
set.seed(42)
cvfit <- cv.glmnet(X_train_scaled, y_train, family = "binomial",
                   alpha = 1, standardize = FALSE)

lambda_fixed <- cvfit$lambda.min
cat("Fixed lambda (lambda.min):", lambda_fixed, "\n")

fit <- glmnet(X_train_scaled, y_train, family = "binomial", alpha = 1,
              lambda = lambda_fixed, standardize = FALSE, thresh = 1e-12)

lasso_coefs <- coef(fit)
nonzero_idx <- which(lasso_coefs[-1, 1] != 0)  # drop intercept 
selected_vars <- rownames(lasso_coefs)[-1][nonzero_idx]
selected_coefs <- lasso_coefs[-1, 1][nonzero_idx]

cat("Number of nonzero coefficients:", length(selected_vars), "\n")
print(sort(selected_coefs, decreasing = TRUE))


# 5.2 Selective inference on lasso-selected variables

n <- nrow(X_train_scaled)
beta_full <- as.numeric(coef(fit))  # includes intercept 


si_result <- fixedLassoInf(x = X_train_scaled, y = y_train, beta = beta_full,
                           lambda = lambda_fixed * n,
                           family = "binomial", type = "partial", alpha = 0.05)

print(si_result)

si_table <- data.frame(
  variable = colnames(X_train_scaled)[si_result$vars],
  estimate = si_result$coef0,
  ci_lower = si_result$ci[, 1],
  ci_upper = si_result$ci[, 2],
  p_value  = si_result$pv
)
si_table$ci_width <- si_table$ci_upper - si_table$ci_lower
si_table <- si_table[order(si_table$p_value), ]
print(si_table)


sig_vars <- si_table[si_table$p_value < 0.05, ]


cat("Number of significant variables (p < 0.05):", nrow(sig_vars), "out of", nrow(si_table), "\n")
print(sig_vars[, c("variable", "estimate", "p_value", "ci_width")])

########################################################################
# STEP 6: Selective Inference vs Naive Inference
########################################################################

# 6.1 Naive logistic regression on Lasso-selected variables
naive_data <- as.data.frame(X_train_scaled[, selected_vars])
naive_data$default <- y_train

naive_fit <- glm(default ~ ., data = naive_data, family = "binomial")
naive_summary <- summary(naive_fit)$coefficients

naive_table <- data.frame(
  variable  = rownames(naive_summary)[-1],           
  naive_est = naive_summary[-1, "Estimate"],
  naive_se  = naive_summary[-1, "Std. Error"],
  naive_p   = naive_summary[-1, "Pr(>|z|)"]
)
naive_table$naive_ci_lower <- naive_table$naive_est - 1.96 * naive_table$naive_se
naive_table$naive_ci_upper <- naive_table$naive_est + 1.96 * naive_table$naive_se


# 6.2 Merge naive results with the corrected (post-selection) results
comparison <- merge(naive_table, si_table, by = "variable")
comparison$naive_significant     <- comparison$naive_p < 0.05
comparison$corrected_significant <- comparison$p_value < 0.05
comparison$naive_ci_width <- comparison$naive_ci_upper - comparison$naive_ci_lower
comparison <- comparison[order(comparison$p_value), ]

cat("Variables significant naively but NOT after correction:\n")
print(comparison$variable[comparison$naive_significant & !comparison$corrected_significant])

cat("\nVariables significant under BOTH naive and corrected inference:\n")
print(comparison$variable[comparison$naive_significant & comparison$corrected_significant])


# Illustration 1 - Bar chart of naive vs corrected significance
# (higher bar = stronger evidence)
cap_val <- 300
plot_data <- comparison[order(comparison$p_value), ]
n_show <- min(20, nrow(plot_data))          # top 20 by corrected p-value
plot_data <- plot_data[1:n_show, ]

naive_neglog10  <- pmin(-log10(plot_data$naive_p), cap_val)
corr_neglog10   <- pmin(-log10(plot_data$p_value), cap_val)
naive_neglog10[!is.finite(naive_neglog10)] <- cap_val
corr_neglog10[!is.finite(corr_neglog10)]   <- cap_val

bar_matrix <- rbind(naive_neglog10, corr_neglog10)
colnames(bar_matrix) <- plot_data$variable

png("~/Downloads/naive_vs_corrected_pvalues.png", width = 1200, height = 1000)
par(mar = c(14, 5, 2, 2))
barplot(bar_matrix, beside = TRUE, col = c("steelblue", "firebrick"),
        las = 2, ylab = "-log10(p)",
        legend.text = c("Naive", "Corrected"), args.legend = list(x = "topright"))
abline(h = -log10(0.05), lty = 2, col = "gray40")
dev.off()

# Illustration 2 - Forest plot comparing confidence intervals
# (wider - greater statistical uncertainty)
ci_vals <- c(plot_data$naive_ci_lower, plot_data$naive_ci_upper,
             plot_data$ci_lower, plot_data$ci_upper)
ci_vals <- ci_vals[is.finite(ci_vals)]
xlim_range <- range(ci_vals)

naive_lo <- pmax(pmin(plot_data$naive_ci_lower, xlim_range[2]), xlim_range[1])
naive_hi <- pmax(pmin(plot_data$naive_ci_upper, xlim_range[2]), xlim_range[1])

png("~/Downloads/naive_vs_corrected_CIs.png", width = 1000, height = 800)
par(mar = c(5, 15, 10, 2))
y_pos <- seq_len(n_show)
plot(NULL, xlim = xlim_range, ylim = c(0.5, n_show + 0.5), yaxt = "n",
     xlab = "Coefficient", ylab = "")
axis(2, at = y_pos, labels = plot_data$variable, las = 2)
abline(v = 0, lty = 2, col = "gray60")
segments(naive_lo, y_pos - 0.15, naive_hi, y_pos - 0.15, col = "steelblue", lwd = 2)
points(plot_data$naive_est, y_pos - 0.15, pch = 19, col = "steelblue")
segments(plot_data$ci_lower, y_pos + 0.15, plot_data$ci_upper, y_pos + 0.15, col = "firebrick", lwd = 2)
points(plot_data$estimate, y_pos + 0.15, pch = 19, col = "firebrick")
legend("topright", legend = c("Naive", "Corrected"), col = c("steelblue", "firebrick"), lwd = 2, pch = 19)
dev.off()

# Illustration 3 - CI width comparison 
comparison$is_monthly <- comparison$variable %in% c(pay_cols, bill_cols, paid_cols)


png("~/Downloads/ci_width_comparison.png", width = 900, height = 700)
plot(comparison$naive_ci_width, comparison$ci_width,
     col = ifelse(comparison$is_monthly, "firebrick", "steelblue"),
     pch = 19, xlab = "Naive CI width", ylab = "Corrected CI width")
abline(0, 1, lty = 2, col = "gray40")
legend("topleft", legend = c("Monthly", "Other"), col = c("firebrick", "steelblue"), pch = 19)
dev.off()


########################################################################
# STEP 7: Random forests:raw and balanced 
########################################################################


# 7.0 Prepare data 
rf_train <- train_final %>%
  dplyr::select(-ID) %>%
  mutate(default = factor(default, levels = c(0, 1), labels = c("no", "yes")))

rf_test <- test_final %>%
  dplyr::select(-ID) %>%
  mutate(default = factor(default, levels = c(0, 1), labels = c("no", "yes")))

# 7.1 Unadjusted random forest (trained on the natural ~22%/78% split)
set.seed(42)
rf_raw <- randomForest(default ~ ., data = rf_train,
                       ntree = 500, importance = TRUE)

# 7.2 Balanced random forest - equal numbers of defaulters and non-defaulters 
n_min <- min(table(rf_train$default))
set.seed(42)
rf_balanced <- randomForest(default ~ ., data = rf_train,
                            ntree = 500, importance = TRUE,
                            sampsize = c(no = n_min, yes = n_min),
                            strata = rf_train$default)

# 7.3 Variable importance from both fits
# MeanDecreaseAccuracy: drop in accuracy when a variable is permuted
# MeanDecreaseGini: total improvement in group purity every time trees split on that variable

imp_raw <- importance(rf_raw)
imp_balanced <- importance(rf_balanced)

importance_table <- data.frame(
  variable = rownames(imp_raw),
  raw_accuracy      = imp_raw[, "MeanDecreaseAccuracy"],
  raw_gini          = imp_raw[, "MeanDecreaseGini"],
  balanced_accuracy = imp_balanced[rownames(imp_raw), "MeanDecreaseAccuracy"],
  balanced_gini     = imp_balanced[rownames(imp_raw), "MeanDecreaseGini"]
)
importance_table <- importance_table[order(-importance_table$raw_accuracy), ]



cat("Raw RF OOB error rate:", rf_raw$err.rate[nrow(rf_raw$err.rate), "OOB"], "\n")
cat("Balanced RF OOB error rate:", rf_balanced$err.rate[nrow(rf_balanced$err.rate), "OOB"], "\n")


# 7.4 Rank comparing 
rank_cor <- cor(rank(-importance_table$raw_accuracy),
                rank(-importance_table$balanced_accuracy),
                method = "spearman")
cat("Spearman rank correlation (raw vs. balanced importance):", round(rank_cor, 3), "\n")

top10_raw <- importance_table$variable[order(-importance_table$raw_accuracy)][1:10]
top10_bal <- importance_table$variable[order(-importance_table$balanced_accuracy)][1:10]
cat("Top 10 overlap (raw vs. balanced):", length(intersect(top10_raw, top10_bal)), "of 10\n")


matching_vars <- intersect(top10_raw, top10_bal)
cat("The overlapping top-10 variables are:\n")
print(matching_vars)


#######################################################################
# STEP 8: Lasso vs. RF  
########################################################################

# 8.0 Combined ranking table
lasso_all <- data.frame(variable = rownames(lasso_coefs)[-1],
                        lasso_coef = as.numeric(lasso_coefs)[-1])

comparable_vars <- intersect(lasso_all$variable, importance_table$variable)

rank_table <- merge(lasso_all[lasso_all$variable %in% comparable_vars, ],
                    importance_table[importance_table$variable %in% comparable_vars, ],
                    by = "variable")
rank_table$lasso_rank <- rank(-abs(rank_table$lasso_coef))
rank_table$rf_rank    <- rank(-rank_table$raw_accuracy)
rank_table$rank_diff  <- abs(rank_table$lasso_rank - rank_table$rf_rank)
rank_table <- rank_table[order(rank_table$rank_diff), ]


# 8.1 Rank correlation and top overlaps (RQ1)
rq1_cor <- cor(rank_table$lasso_rank, rank_table$rf_rank, method = "spearman")
cat("\nSpearman rank correlation (lasso vs. RF importance):", round(rq1_cor, 3), "\n")

top10_lasso <- rank_table$variable[order(rank_table$lasso_rank)][1:10]
top10_rf    <- rank_table$variable[order(rank_table$rf_rank)][1:10]
cat("Top 10 overlap (lasso vs. RF):", length(intersect(top10_lasso, top10_rf)), "of 10\n")
print(intersect(top10_lasso, top10_rf))

for (k in c(5, 10, 20, 30)) {
  top_k_lasso <- rank_table$variable[order(rank_table$lasso_rank)][1:k]
  top_k_rf    <- rank_table$variable[order(rank_table$rf_rank)][1:k]
  
  common_features <- intersect(top_k_lasso, top_k_rf)
  
  cat("\n--- Top", k, "Overlap:", length(common_features), "of", k, "---\n")
  if (length(common_features) > 0) {
    print(common_features)
  } else {
    cat("None\n")
  }
}

# View across all comparable variables 
cat("\nVariables where lasso and RF agree most closely (smallest rank_diff):\n")
print(head(rank_table[, c("variable", "lasso_rank", "rf_rank", "rank_diff")], 15))

cat("\nVariables where lasso and RF disagree most (largest rank_diff):\n")
print(tail(rank_table[, c("variable", "lasso_rank", "rf_rank", "rank_diff")], 15))

n_comparable <- nrow(rank_table)
cat("\nMedian rank_diff across all comparable variables:",
    median(rank_table$rank_diff), "\n")

# 8.2 Correlated monthly blocks (RQ2)
monthly_blocks <- list(PAY = pay_cols, BILL_AMT = bill_cols, PAY_AMT = paid_cols)

for (block_name in names(monthly_blocks)) {
  cols <- monthly_blocks[[block_name]]
  cat("\n--", block_name, "--\n")
  cat("Lasso selected:", intersect(cols, selected_vars), "\n")
  block_imp <- importance_table[importance_table$variable %in% cols, c("variable", "raw_accuracy")]
  block_imp <- block_imp[order(block_imp$variable), ]
  print(block_imp)
  cat("RF importance spread (SD across the 6 months):", round(sd(block_imp$raw_accuracy), 4), "\n")
}

# 8.3 CI widths 
monthly_raw <- c(pay_cols, bill_cols, paid_cols)
si_table$is_monthly <- si_table$variable %in% monthly_raw

cat("\nMean post-selection CI width, monthly raw variables:",
    round(mean(si_table$ci_width[si_table$is_monthly]), 4), "\n")
cat("Mean post-selection CI width, other selected variables:",
    round(mean(si_table$ci_width[!si_table$is_monthly]), 4), "\n")

# 8.4 Illustration 4 - RF importance across the 6 months per block, lasso-selected months (red)

png("~/Downloads/monthly_block_comparison.png", width = 1600, height = 700, res = 100)

par(mfrow = c(1, 3), 
    mar = c(8, 6, 4, 2), 
    cex.lab = 1.6,   
    cex.axis = 1.4,  
    cex.main = 2.0)  

for (block_name in names(monthly_blocks)) {
  cols <- monthly_blocks[[block_name]]
  block_imp <- importance_table[match(cols, importance_table$variable), c("variable", "raw_accuracy")]
  bar_col <- ifelse(block_imp$variable %in% selected_vars, "firebrick", "steelblue")
  
  barplot(block_imp$raw_accuracy, 
          names.arg = block_imp$variable, 
          col = bar_col,
          las = 2, 
          ylab = "RF importance", 
          main = block_name,
          cex.names = 1.5)  
}

dev.off()

########################################################################
# STEP 9: Predictive performance on test set
########################################################################


# 9.1 Predicted probabilities on test data, one per model
lasso_pred       <- as.numeric(predict(fit, newx = X_test_scaled, type = "response"))
rf_raw_pred      <- predict(rf_raw, newdata = rf_test, type = "prob")[, "yes"]
rf_balanced_pred <- predict(rf_balanced, newdata = rf_test, type = "prob")[, "yes"]

# 9.2 ROC curves and AUC, computed only on the held-out test set 
roc_lasso  <- roc(y_test, lasso_pred, quiet = TRUE)
roc_rf_raw <- roc(y_test, rf_raw_pred, quiet = TRUE)
roc_rf_bal <- roc(y_test, rf_balanced_pred, quiet = TRUE)

auc_table <- data.frame(
  model = c("Lasso", "RF (raw)", "RF (balanced)"),
  auc   = c(auc(roc_lasso), auc(roc_rf_raw), auc(roc_rf_bal))
)
cat("\nTest-set AUC:\n")
print(auc_table)


ci_lasso  <- ci.auc(roc_lasso)
ci_rf_raw <- ci.auc(roc_rf_raw)
ci_rf_bal <- ci.auc(roc_rf_bal)

auc_table <- data.frame(
  model = c("Lasso", "RF (raw)", "RF (balanced)"),
  auc   = c(auc(roc_lasso), auc(roc_rf_raw), auc(roc_rf_bal)),
  ci_lower = c(ci_lasso[1], ci_rf_raw[1], ci_rf_bal[1]),
  ci_upper = c(ci_lasso[3], ci_rf_raw[3], ci_rf_bal[3])
)

auc_table$auc_ci <- sprintf(
  "%.3f (%.3f - %.3f)", 
  auc_table$auc, 
  auc_table$ci_lower, 
  auc_table$ci_upper
)


cat("\nTest-set AUC with 95% Confidence Intervals:\n")
print(auc_table[, c("model", "auc_ci")])

# 9.3 Illustration - ROC curves 

png("~/Downloads/roc_comparison.png", width = 800, height = 800)
plot(roc_lasso, col = "steelblue")
lines(roc_rf_raw, col = "firebrick")
lines(roc_rf_bal, col = "darkgreen")
legend("bottomright",
       legend = c(paste0("Lasso (", round(auc(roc_lasso), 3), ")"),
                  paste0("RF raw (", round(auc(roc_rf_raw), 3), ")"),
                  paste0("RF balanced (", round(auc(roc_rf_bal), 3), ")")),
       col = c("steelblue", "firebrick", "darkgreen"), lwd = 2)
dev.off()
