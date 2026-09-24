# In this file, we combine the results of applying DiCE and the three other CE 
# techniques on the entire test set from the telematics data.

# Import packages
library(dplyr)
library(tidyr)
library(iml)
library(purrr)
source("scripts/utils.R")

# WhatIf, NICE and MOC results
load("out/telematics-ce-test.Rda")
names(res_all_test)
# DiCE results
ces_dice <- read.csv("out/cf_dice_test.csv")
names(ces_dice)

#### LACP on DiCE ####
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

# Uncertainty in the test set
data_dice <- ces_dice |>
  drop_na() |>
  select(c(
    idx_query, Duration, Insured_age, Insured_sex, Car_age, Credit_score, 
    Region, Pct_drive_rush, Accel_14miles, Brake_14miles
  ))

lacp_dice <- split_cp_lacp(
  glm_model, 
  predict_fun_glm, 
  df_calib,
  data_dice |> select(-idx_query),
  alpha = .2
)

indices <- unique(ces_dice$idx_query)
df_y_u <- data.frame(
  idx_query = indices,
  y_u = .9 * glm_pred_test[indices]
)

data_dice_reliable <- lacp_dice |>
  mutate(idx_query = data_dice$idx_query) |>
  left_join(df_y_u, by = "idx_query") |>
  select(idx_query, upper, y_u) |>
  mutate(reliable = if_else(upper <= y_u, 1, 0)) 

# Evaluate criteria
indices_wo_na <- unique(data_dice$idx_query)
#handlers(global = TRUE)
#with_progress({
#  idx <- indices_wo_na
#  p <- progressor(along = idx)
#  evals <- map_dfr(
#    idx,
#    function(idx_q) {
#      p()
#        res <- evaluate(
#          data = data_dice |> filter(idx_query == idx_q) |> select(-idx_query), 
#          idx_query = idx_q, 
#          distance_function = distance_function, 
#          x_interest = df_test_[idx_q,] |> select(-Y), 
#          X_train = df_train |> select(-Y)
#          )
#      res
#    }
#  )
#})
#save(evals, file = "out/evals_dice.Rda")

load("out/evals_dice.Rda")
data_dice_reliable_eval <- cbind(data_dice_reliable, evals |> select(-idx_query))
data_dice_reliable_eval <- data_dice_reliable_eval |>
  group_by(idx_query) |>
  mutate(idx_cf = row_number()) |>
  ungroup()

ces_dice <- ces_dice |> 
  group_by(idx_query) |>
  mutate(idx_cf = row_number()) |>
  ungroup()

res_all_test_dice <- ces_dice |>
  left_join(data_dice_reliable_eval, by = c("idx_query", "idx_cf")) |>
  select(-idx_cf) |>
  mutate(
    valid = ifelse(is.na(pred), NA, 1),
    method = "dice"
    ) |>
  select(c(
    "pred", "upper", "dist_train", "dist_x_interest", "no_changed", "valid",
    "time", "Pct_drive_rush", "Accel_14miles", "Brake_14miles", "method", 
    "reliable", "idx_query", "n_duplicates"  
  )) |>
  rename(duplicates = n_duplicates)

#### Combining results ####
res_all_test <- rbind(res_all_test |> mutate(duplicates = 1), res_all_test_dice)
nb_obs <- length(unique(res_all_test$idx_query))
res_all_test <- res_all_test |>
  mutate(
    cf = ifelse(is.na(pred), 0, 1)
  ) |>
  group_by(method, idx_query) |>
  mutate(n_cf = sum(cf)) |>
  ungroup()

res_all_test_wo_na <- res_all_test |>
  drop_na()

res_shared <- res_all_test_wo_na |>
  group_by(across(-c(
    method, time, cf, n_cf, upper, dist_train, dist_x_interest, no_changed, 
    valid, reliable  
    ))) |>
  mutate(
    n_methods = n_distinct(method),
    shared = if_else(n_methods > 1, 1, 0)
  ) |>
  ungroup() |>
  select(-n_methods)

#### Table 6 ####

res_all_test_wo_na |>
  group_by(method, reliable) |> 
  summarise(no_changed = mean(no_changed))

res_all_test_wo_na |>
  group_by(reliable) |> 
  summarise(no_changed = mean(no_changed))

# Calculate severity reduction
pred_queries <- 
  tibble(
    idx_query = unique(res_all_test_wo_na$idx_query),
    pred_query = glm_pred_test[unique(res_all_test_wo_na$idx_query)]
  )
res_all_test_wo_na_sev <- res_all_test_wo_na |>
  left_join(pred_queries, by = "idx_query") |>
  mutate(reduction = (pred_query - pred)/pred_query) |>
  filter(valid != 0)
res_all_test_wo_na_sev |> 
  group_by(method, reliable) |>
  summarise(reduction = mean(reduction))
res_all_test_wo_na_sev |> 
  group_by(reliable) |>
  summarise(reduction = mean(reduction))

#### Table 7 ####

res_ce_test <- res_shared |>
  group_by(method) |>
  summarise(
    pred = mean(pred),
    upper = mean(upper),
    plausibility = mean(dist_train),
    proximity = mean(dist_x_interest),
    sparsity = mean(no_changed),
    validity = mean(valid),
    reliability = mean(reliable),
    exec_time = mean(time),
    shared = mean(shared),
    duplicates = mean(duplicates),
    nb = n()
  )

res_distinct <- res_all_test |>
  select(method, idx_query, n_cf, cf) |>
  distinct() |>
  group_by(method) |>
  mutate(imp = sum(cf == 0)) |>
  ungroup() |>
  filter(n_cf != 0) |>
  group_by(method) |>
  summarise(
    n_cf = mean(n_cf),
    prop_unfound = mean(imp)/nb_obs
  )

res_ce_test <- res_ce_test |> 
  left_join(res_distinct, by = "method")
View(res_ce_test |> select(method, plausibility, proximity, sparsity, validity, 
                           reliability, exec_time, shared, duplicates,
                           n_cf))
