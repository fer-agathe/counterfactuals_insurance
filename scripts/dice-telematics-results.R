# In this file, we calculate the Gower distance over CEs from DiCE in the 
# telematics data for post-hoc plausibility filtering.

# Import packages
library(dplyr)
library(tidyr)
library(iml)
source("scripts/utils.R")

# DiCE results
ces_dice <- read.csv("out/cf_dice.csv")
nrow(ces_dice)

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

df_train <- read.csv("out/df_train.csv")
df_train <- df_train |>
  mutate(
    Insured_sex = as.factor(Insured_sex),
    Region = as.factor(Region),
    Pct_drive_rush = factor(Pct_drive_rush, levels = c("none", "mostly_am", "mostly_pm", "both")),
    Accel_14miles = as.factor(Accel_14miles),
    Brake_14miles = as.factor(Brake_14miles)
  )

ces_dice <- ces_dice |>
  mutate(across(
    where(is.factor),
    ~ factor(.x, levels = levels(df_train[[cur_column()]]))
  ))

# Gamma GLM
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
df_test_ <- df_test |>
  mutate(Duration = 1)

predictor_glm <- Predictor$new(
  model = glm_model,
  data = df_train_ |> select(-Y),
  y = df_train_$Y,
  predict.function = predict_fun_glm
)

# Uncertainty on the CEs
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

# Calculate plausibility criterion
evals <- evaluate(
  data = data_dice |> select(-idx_query), 
  idx_query = indices, 
  distance_function = distance_function, 
  x_interest = df_test_[indices,] |> select(-Y), 
  X_train = df_train |> select(-Y)
  )

data_dice_reliable_eval <- cbind(data_dice_reliable, evals |> select(-idx_query))
data_dice_reliable_eval <- data_dice_reliable_eval |>
  group_by(idx_query) |>
  mutate(idx_cf = row_number()) |>
  ungroup()

ces_dice <- ces_dice |> 
  group_by(idx_query) |>
  mutate(idx_cf = row_number()) |>
  ungroup()

res_dice <- ces_dice |>
  left_join(data_dice_reliable_eval, by = c("idx_query", "idx_cf")) |>
  select(-idx_cf) |>
  select(c(
    "pred", "upper", "dist_train", "dist_x_interest", "no_changed",
    "Pct_drive_rush", "Accel_14miles", "Brake_14miles",
    "reliable", "idx_query"
  ))
res_dice

ordered_res_dice <- res_dice |> arrange(dist_train) |> slice(1:5)
max_dist <- max(ordered_res_dice$dist_train)

res_dice |> filter(dist_train <= max_dist)