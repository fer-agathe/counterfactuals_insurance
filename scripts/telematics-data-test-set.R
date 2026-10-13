# In this R script, we apply the three CE techniques from the R package
# counterfactuals (WhatIf, NICE, MOC) to the entire test set of the telematics
# dataset. We then save the results to compare these techniques with the DiCE
# package in Python.

# Import packages
library(tidyr)
library(dplyr)
library(xgboost)
library(purrr)
library(progressr)
library(iml)
library(lubridate)
library(rfema)
library(caret)
library(scales)
library(ranger)
library(tibble)
library(ggplot2)
library(CASdatasets)
# We take our extension of the R package counterfactuals 
#devtools::install_github("fer-agathe/counterfactuals")
library(counterfactuals)

source("scripts/utils.R")
set.seed(2026)

# We don't want the logs from the genetic algo. to be displayed.
library(lgr)
lgr::get_logger("bbotk")$set_threshold("off")
lgr::get_logger("mlr3")$set_threshold("off")
lgr::get_logger("mlr3misc")$set_threshold("off")
lgr::get_logger("miesmuschel")$set_threshold("off")
# Warnings are not displayed.
options(warn = -1)

library(wesanderson)
colors_ <- wes_palette('Rushmore1')

#### Data Pre-processing ####
data("catelematic13")
data <- catelematic13
# There are no missing values.
sum(is.na(data)) 
names(data) <- gsub("[\\. ]", "_", names(data))
#names(data) # feature names
data <- data |> filter(Car_age >= 0)

# Exposure
data_expo <- data |> select(Duration) |> mutate(Duration = Duration / 366)

# Traditional variables: we keep only six of them
trad_variables <- c("Insured_age", "Insured_sex", "Car_age", "Credit_score", "Region")
data_trad <- data |> select(all_of(trad_variables))
# There are two categ. variables at the character format: Insured_sex and Region 

# Transform Pct_drive_rush_am/pm
data <- data |>
  mutate(
    Pct_drive_rush = case_when(
      Pct_drive_rush_am < 0.10 & Pct_drive_rush_pm < 0.10 ~ "none",
      Pct_drive_rush_am >= 0.10 & Pct_drive_rush_pm >= 0.10 ~ "both",
      Pct_drive_rush_am >= 0.10 & Pct_drive_rush_am > Pct_drive_rush_pm ~ "mostly_am",
      Pct_drive_rush_pm >= 0.10 & Pct_drive_rush_pm > Pct_drive_rush_am ~ "mostly_pm"
    )
  ) |> select(-Pct_drive_rush_am, -Pct_drive_rush_pm)

data$Pct_drive_rush <- factor(
  data$Pct_drive_rush,
  levels = c("none", "mostly_am", "mostly_pm", "both")
)

# Transform Accel/Brake_14miles
data <- data |> mutate(
  Accel_14miles = as.integer(Accel_14miles>0),
  Brake_14miles = as.integer(Brake_14miles>0)
)

telematics_variables <- c("Pct_drive_rush", "Accel_14miles", "Brake_14miles")
data_telematics <- data |> select(all_of(telematics_variables))

df <- tibble(
  data_expo,
  data_trad,
  data_telematics
)
df <- df |> mutate(Y = data$NB_Claim)

# We transform categ. variables into factors
df <- df |>
  mutate(
    Insured_sex = as.factor(Insured_sex),
    Region = as.factor(Region),
    Accel_14miles = as.factor(Accel_14miles),
    Brake_14miles = as.factor(Brake_14miles)
  )

# Splitting the dataset into:
# a training set, to fit the severity models (GLM Poisson),
# a calibration set, to fit the Conformal Prediction (CP) quantities,
# a test set, to calculate counterfactual explanations and associated uncertainties.
n <- nrow(df)
# Random permutation of row indices
idx <- sample(seq_len(n))
# Sizes
n_train <- floor(0.60 * n)
n_calib <- floor(0.30 * n)
n_test  <- n - n_train - n_calib
# Split indices
train_idx <- idx[1:n_train]
calib_idx <- idx[(n_train + 1):(n_train + n_calib)]
test_idx  <- idx[(n_train + n_calib + 1):n]
# Create datasets
df_train <- df[train_idx, ]
df_calib <- df[calib_idx, ]
df_test  <- df[test_idx, ]

# Save for DiCE
write.csv(df_train, "out/df_train.csv", row.names = FALSE)
write.csv(df_calib, "out/df_calib.csv", row.names = FALSE)
write.csv(df_test,  "out/df_test.csv",  row.names = FALSE)

#### GLM fitting ####

# Poisson GLM
glm_model <- glm(
  Y ~ . - Duration, family = poisson(), offset = log(Duration), data = df_train
)

# Save model for DiCE
save(glm_model, file = "out/glm_poisson.RData")
write.csv(model.matrix(glm_model), "out/glm_design_matrix_train.csv", row.names = FALSE)
coef_df <- data.frame(
  name = names(coef(glm_model)),
  value = as.numeric(coef(glm_model))
)
write.csv(coef_df, "out/glm_coefficients.csv", row.names = FALSE)

# GLM Predictions
glm_pred_train <- predict(glm_model, newdata = df_train, type = "response") / df_train$Duration
glm_pred_calib <- predict(glm_model, newdata = df_calib, type = "response") / df_calib$Duration
glm_pred_test <- predict(glm_model, newdata = df_test, type = "response") / df_test$Duration

# We force Exposure = 1 for all obs. bc we don't need it in the CEs
df_train_ <- df_train |> mutate(Duration = 1) |> distinct()
df_test_ <- df_test |> mutate(Duration = 1)

# Predictor functions
predict_fun_glm <- function(model, newdata) {
  expo = newdata$Duration
  predict(model, newdata = newdata, type = "response") / expo
}

predictor_glm <- Predictor$new(
  model = glm_model,
  data = df_train_ |> select(-Y),
  y = df_train_$Y,
  predict.function = predict_fun_glm
)

#### Conformal Prediction ####

# Uncertainty in the test set
lacp_glm <- split_cp_lacp(
  model = glm_model,
  predict_fun = predict_fun_glm,
  data_calib = df_calib,
  data_test = df_test_,
  alpha = .2
)

#### Counterfactual Explanations ####

# CE techniques: WhatIf, NICE and MOC
whatif_glm <- WhatIfRegr$new(
  predictor_glm,
  n_counterfactuals = 100L, # Max. value
  distance_function = "gower_c",
  fixed_features = c(trad_variables, "Duration"),
  epsilon = .3
)

nice_glm <- NICERegr$new(
  predictor_glm,
  optimization = "proximity",
  x_nn_correct = TRUE,
  return_multiple = TRUE,
  finish_early = FALSE,
  distance_function = "gower_c",
  fixed_features = c(trad_variables, "Duration")
)

moc_glm <- MOCRegr$new(
  predictor_glm,
  fixed_features = c(trad_variables, "Duration"),
  distance_function = "gower_c",
  mu = 5,
  n_generations = 10
)

ce_one_query <- function(idx_q) {
  
  # Query information
  query <- df_test_[idx_q,]
  pred_query <- glm_pred_test[idx_q]
  
  # Calculate CEs
  y_u <- .9*pred_query
  
  # WhatIf
  res_ce_whatif <- calculate_ce(
    query, 
    pred_query, 
    y_u, 
    predictor = predictor_glm, 
    predict_fun = predict_fun_glm,
    df_calib = df_calib,
    model = glm_model,
    ce_object = whatif_glm,
    alpha = .2
  )
  
  # NICE
  res_ce_nice <- 
    calculate_ce(
    query, 
    pred_query, 
    y_u, 
    predictor = predictor_glm, 
    predict_fun = predict_fun_glm,
    df_calib = df_calib,
    model = glm_model,
    ce_object = nice_glm,
    alpha = .2
  )
  
  # MOC
  res_ce_moc <- suppressMessages(calculate_ce(
      query, 
      pred_query, 
      y_u, 
      predictor = predictor_glm, 
      predict_fun = predict_fun_glm,
      df_calib = df_calib,
      model = glm_model,
      ce_object = moc_glm,
      alpha = .2
      ))
  
  # Combine results
  res_all_ce <- res_ce_whatif |>
    mutate(method = "whatif") |> 
    bind_rows(
      res_ce_nice |>
        mutate(method = "nice")
    ) |>
    bind_rows(
      res_ce_moc |>
        mutate(method = "moc")
    ) |>
    mutate(
      reliable = if_else(upper <= y_u, 1, 0)
    ) |>
    mutate(idx_query = idx_q)
  
  res_all_ce
}

idx_q <- 1
res_all_1 <- ce_one_query(idx_q)

# We sample 1000 rows from df_test
#indices <- sample(seq_len(nrow(df_test_)), 1000, replace = FALSE)
#write.csv(indices, "out/sampled_indices.csv", row.names = FALSE)
indices <- read.csv("out/sampled_indices.csv")
indices <- indices$x

handlers(global = TRUE)
with_progress({
  idx <- indices
  p <- progressor(along = idx)
  res_all_test <- map_dfr(
    idx,
    function(idx_q) {
      p()
      invisible(capture.output(
        res <- ce_one_query(idx_q)
      ))
      res
    }
  )
})
save(res_all_test, file = "out/telematics-ce-test-alpha-2.Rda")
