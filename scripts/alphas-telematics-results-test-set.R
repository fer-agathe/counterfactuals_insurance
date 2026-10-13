# In this file, we combine the results of applying DiCE and the three other CE 
# techniques on the entire test set from the telematics data.
# Import packages
library(dplyr)
library(tidyr)
library(iml)
library(purrr)
source("scripts/utils.R")

# WhatIf, NICE and MOC results varying alpha: 0.05, 0.2
# alpha = 0.05
load("out/telematics-ce-test-alpha-5.Rda")
res_test_5 <- res_all_test
# alpha = 0.2
load("out/telematics-ce-test-alpha-2.Rda")
res_test_2 <- res_all_test
rm(res_all_test)
# DiCE results
ces_dice <- read.csv("out/cf_dice_test.csv")
load("out/evals_dice.Rda") # evals

#### Pre-processing for DiCE ####
ces_dice <- ces_dice |>
  mutate(Duration = 1) |>
  mutate(
    Insured_sex = as.factor(ifelse(Insured_sexMale == 1, "Male", "Female")),
    Region = as.factor(ifelse(RegionUrban == 1, "Urban", "Rural")),
    Pct_drive_rush = factor(case_when(
      Pct_drive_rushmostly_am == 1 ~ "mostly_am",
      Pct_drive_rushmostly_pm == 1 ~ "mostly_pm",
      Pct_drive_rushboth == 1 ~ "both",
      TRUE ~ "none"
    ), levels = c("none", "mostly_am", "mostly_pm", "both")),
    Accel_14miles = as.factor(Accel_14miles1),
    Brake_14miles = as.factor(Brake_14miles1),
    Insured_age = as.integer(Insured_age),
    Car_age = as.integer(Car_age),
    Credit_score = as.integer(Credit_score)
  ) |>
  select(-c("Insured_sexMale", "RegionUrban", "Pct_drive_rushmostly_am",
            "Pct_drive_rushmostly_pm", "Pct_drive_rushboth", "Accel_14miles1",
            "Brake_14miles1"))

# Gamma GLM
df_train <- read.csv("out/df_train.csv")
df_train <- df_train |>
  mutate(
    Insured_sex = as.factor(Insured_sex),
    Region = as.factor(Region),
    Pct_drive_rush = factor(Pct_drive_rush, levels = c("none", "mostly_am", "mostly_pm", "both")),
    Accel_14miles = as.factor(Accel_14miles),
    Brake_14miles = as.factor(Brake_14miles)
  )

glm_model <- glm(
  Y ~ . - Duration, family = poisson(), offset = log(Duration), data = df_train
)

df_calib <- read.csv("out/df_calib.csv")
df_test <- read.csv("out/df_test.csv")

df_calib <- df_calib |>
  mutate(
    Insured_sex = as.factor(Insured_sex),
    Region = as.factor(Region),
    Pct_drive_rush = factor(Pct_drive_rush, levels = c("none", "mostly_am", "mostly_pm", "both")),
    Accel_14miles = as.factor(Accel_14miles),
    Brake_14miles = as.factor(Brake_14miles)
  )

df_test <- df_test |>
  mutate(
    Insured_sex = as.factor(Insured_sex),
    Region = as.factor(Region),
    Pct_drive_rush = factor(Pct_drive_rush, levels = c("none", "mostly_am", "mostly_pm", "both")),
    Accel_14miles = as.factor(Accel_14miles),
    Brake_14miles = as.factor(Brake_14miles)
  )

df_test_ <- df_test |>
  mutate(Duration = 1)

# GLM Predictions
glm_pred_train <- predict(glm_model, newdata = df_train, type = "response") / df_train$Duration
glm_pred_calib <- predict(glm_model, newdata = df_calib, type = "response") / df_calib$Duration
glm_pred_test <- predict(glm_model, newdata = df_test, type = "response") / df_test$Duration

# Predictor functions
predict_fun_glm <- function(model, newdata) {
  expo = newdata$Duration
  predict(model, newdata = newdata, type = "response") / expo
}

df_train_ <- df_train |>
  mutate(Duration = 1)

predictor_glm <- Predictor$new(
  model = glm_model,
  data = df_train_ |> select(-Y),
  y = df_train_$Y,
  predict.function = predict_fun_glm
)

indices <- unique(ces_dice$idx_query)
df_y_u <- data.frame(
  idx_query = indices,
  y_u = .9 * glm_pred_test[indices]
)

# Predictions of query instances
pred_queries <- 
  tibble(
    idx_query = indices,
    pred_query = glm_pred_test[indices]
  )
pred_queries <- pred_queries |>
  mutate(risk_decile = cut(
    pred_query,
    breaks = quantile(
      pred_query, probs = seq(0, 1, length.out = 4), na.rm = TRUE
    ),
    include.lowest = TRUE,
    labels = 1:3
  ))

data_dice <- ces_dice |>
  drop_na() |>
  select(c(
    idx_query, Duration, Insured_age, Insured_sex, Car_age, Credit_score, 
    Region, Pct_drive_rush, Accel_14miles, Brake_14miles
  ))

#### Tables varying alpha ####
generate_tables <- function(res_all, res_dice, alpha) {
  lacp_dice <- split_cp_lacp(
    glm_model, 
    predict_fun_glm, 
    df_calib,
    res_dice |> select(-idx_query),
    alpha = alpha
  )
  df_dice <- lacp_dice |>
    mutate(idx_query = res_dice$idx_query) |>
    left_join(df_y_u, by = "idx_query") |>
    select(idx_query, upper, y_u, pred) |>
    mutate(reliable = if_else(upper <= y_u, 1, 0)) |>
    select(idx_query, reliable, pred) |>
    group_by(idx_query) |>
    mutate(idx_cf = row_number()) |>
    ungroup()
  
  reliability_dice <- tibble(
    method = "dice",
    reliability = mean(df_dice$reliable)
  ) 
  
  eval_dice <- cbind(df_dice, evals |> select(-idx_query)) |>
    select(-dist_train)
  # We only keep queries with corresponding reliable CEs
  eval_dice <- eval_dice |>
    mutate(
      query_reliable = as.integer(any(reliable == 1)),
      .by = idx_query
    ) |>
    mutate(
      prop_reliable = mean(query_reliable[!duplicated(idx_query)])
    ) |>
    filter(query_reliable == 1) |>
    select(-query_reliable) |>
    mutate(alpha = alpha) |>
    left_join(pred_queries, by = "idx_query") |>
    mutate(reduction = (pred_query - pred)/pred_query) |>
    select(-pred_query)
  
  # We group by idx_query
  df_dice_queries <- eval_dice |>
    group_by(alpha, prop_reliable, idx_query, reliable) |> 
    summarise(
      proximity = mean(dist_x_interest), 
      sparsity = mean(no_changed),
      reduction = mean(reduction)
    ) |>
    mutate(method = "dice") |>
    select(
      c(
        "method", "alpha", "idx_query", "reliable", "proximity",
        "sparsity", "reduction" #,"prop_reliable"
      )
    )
  
  # We do the same with WhatIf, NICE and MOC
  df_all <- res_all |>
    select(idx_query, method, pred, reliable, dist_x_interest, no_changed) |>
    group_by(idx_query) |>
    mutate(idx_cf = row_number()) |>
    ungroup() |>
    mutate(
      query_reliable = as.integer(any(reliable == 1)),
      .by = c(method, idx_query)
    ) |>
    mutate(
      prop_reliable = mean(query_reliable[!duplicated(idx_query)]),
      .by = method
    ) 
  
  reliability_all <- df_all |>
    group_by(method) |>
    summarise(reliability = mean(reliable))
  
  df_all <- df_all |>
    filter(query_reliable == 1) |>
    select(-query_reliable) |>
    mutate(alpha = alpha) |>
    left_join(pred_queries, by = "idx_query") |>
    mutate(reduction = (pred_query - pred)/pred_query) |>
    select(-pred_query)
  
  df_queries <- df_all |>
    group_by(method, alpha, prop_reliable, idx_query, reliable) |> 
    summarise(
      proximity = mean(dist_x_interest), 
      sparsity = mean(no_changed),
      reduction = mean(reduction)
    )
  
  # We combine results of the four CE methods
  df_alpha <- rbind(df_queries, df_dice_queries)
  df_diff <- df_alpha |>
    group_by(method, alpha, prop_reliable, idx_query) |>
    summarise(
      across(
        c(proximity, sparsity, reduction),
        ~ .x[reliable == 1][1] - .x[reliable == 0][1],
        .names = "diff_{.col}"
      ),
      .groups = "drop"
    )
  
  average_diff <- df_diff |>
    group_by(method, alpha, prop_reliable) |>
    summarise(
      diff_proximity = mean(diff_proximity),
      diff_sparsity = mean(diff_sparsity),
      diff_reduction = mean(diff_reduction)
    )
  
  # Average differences by risk-decile of prediction
  df_all_risk <- rbind(
    df_all,
    eval_dice |> 
      mutate(method = "dice") |>
      select(c(
        "idx_query", "method", "pred", "reliable", "dist_x_interest", 
        "no_changed", "idx_cf", "alpha", "reduction",
        "risk_decile", "prop_reliable"
      ))
    )
  
  df_queries_risk <- df_all_risk |>
    group_by(method, alpha, prop_reliable, idx_query, reliable, risk_decile) |>
    summarise(
      proximity = mean(dist_x_interest), 
      sparsity = mean(no_changed),
      reduction = mean(reduction)
    )

  df_diff_risk <- df_queries_risk |>
    group_by(method, alpha, prop_reliable, risk_decile, idx_query) |>
    summarise(
      across(
        c(proximity, sparsity, reduction),
        ~ .x[reliable == 1][1] - .x[reliable == 0][1],
        .names = "diff_{.col}"
      ),
      .groups = "drop"
    )
  
  average_diff_risk <- df_diff_risk |>
    group_by(method, alpha, prop_reliable, risk_decile) |>
    summarise(
      diff_proximity = mean(diff_proximity),
      diff_sparsity = mean(diff_sparsity),
      diff_reduction = mean(diff_reduction)
    ) |>
    mutate(risk = case_when(
      risk_decile == 1 ~ "Low",
      risk_decile == 2 ~ "Medium",
      risk_decile == 3 ~ "High",
      TRUE ~ NA
      )) |>
    select(-risk_decile)
  
  list(
    global_average = average_diff,
    risk_average = average_diff_risk,
    reliability_average = rbind(reliability_all, reliability_dice)
  )
}

# alpha = 0.2
tables_alpha_2 <- generate_tables(
  res_all = res_test_2 |> drop_na(),
  res_dice = data_dice,
  alpha = .2
)
table_alpha_2_global <- tables_alpha_2$global_average
table_alpha_2_risk <- tables_alpha_2$risk_average
table_alpha_2_rel <- tables_alpha_2$reliability_average

# alpha = 0.05
tables_alpha_5 <- generate_tables(
  res_all = res_test_5 |> drop_na(),
  res_dice = data_dice,
  alpha = .05
)
table_alpha_5_global <- tables_alpha_5$global_average
table_alpha_5_risk <- tables_alpha_5$risk_average
table_alpha_5_rel <- tables_alpha_5$reliability_average

# All tables together
table_global <- rbind(
  table_alpha_5_global, table_alpha_2_global
  )
#table_risk <- rbind(
#  table_alpha_5_risk, table_alpha_2_risk
#)

