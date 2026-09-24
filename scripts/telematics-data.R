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

library(wesanderson)
colors_ <- wes_palette('Rushmore1')

#### Data pre-processing ####
data("catelematic13")
data <- catelematic13
# There are no missing values.
sum(is.na(data)) 
names(data) <- gsub("[\\. ]", "_", names(data))
#names(data) # feature names
cat("Number of features:", ncol(data), "\nNumber of policyholders:", nrow(data), "\n")
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

# Actionable features
telematics_variables <- c("Pct_drive_rush", "Accel_14miles", "Brake_14miles")
data_telematics <- data |> select(all_of(telematics_variables))

# Final dataframe
df <- tibble(
  data_expo,
  data_trad,
  data_telematics
)
cat("Number of features:", ncol(df), "\nNumber of policyholders:", nrow(df), "\n")

# Response variable: number of claims
df <- df |> mutate(Y = data$NB_Claim)

# We transform categ. variables into factors
df <- df |>
  mutate(
    Insured_sex = as.factor(Insured_sex),
    Region = as.factor(Region),
    Accel_14miles = as.factor(Accel_14miles),
    Brake_14miles = as.factor(Brake_14miles)
  )

head(df)
cat("Number of features:", ncol(df), "\nNumber of policyholders:", nrow(df), "\n")

# Show outcome variable proportion
prop <- round(prop.table(table(df$Y)) * 100, 2)
prop

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

# Proportion of Y in subsets
prop_train <- round(prop.table(table(df_train$Y)) * 100, 2)
prop_train
prop_calib <- round(prop.table(table(df_calib$Y)) * 100, 2)
prop_calib
prop_test <- round(prop.table(table(df_test$Y)) * 100, 2)
prop_test

#### GLM Poisson model fitting ####

# Gamma GLM
glm_model <- glm(
  Y ~ . - Duration, family = poisson(), offset = log(Duration), data = df_train
)
summary(glm_model)

# GLM Predictions: annual rate (i.e., divided by the exposure)
glm_pred_train <- predict(glm_model, newdata = df_train, type = "response") / df_train$Duration
glm_pred_calib <- predict(glm_model, newdata = df_calib, type = "response") / df_calib$Duration
glm_pred_test <- predict(glm_model, newdata = df_test, type = "response") / df_test$Duration

#### Result Evaluation ####

# Metrics

res_train <- evaluate_poisson_model(df_train$Y, glm_pred_train) |>
  mutate(subset = "overall", set = "train", model = "glm-poisson")

res_calib <- evaluate_poisson_model(df_calib$Y, glm_pred_calib) |>
  mutate(subset = "overall", set = "calib", model = "glm-poisson") 

res_test <- evaluate_poisson_model(df_test$Y, glm_pred_test) |>
  mutate(subset = "overall", set = "test", model = "glm-poisson")

results <- res_train |>
  bind_rows(res_calib) |>
  bind_rows(res_test) |>
  select(set, model, subset, deviance, mae, rmse)
results

#### Conformal Prediction ####

# Query index in the test set
idx_query <- which(glm_pred_test >= .2 & df_test$Y>=1)[2]
query <- df_test[idx_query,]
query |> select(all_of(telematics_variables))
df_test$Y[idx_query]
expo_query <- df_test$Duration[idx_query]
glm_pred_query <- glm_pred_test[idx_query]
glm_pred_query

# Exposure treatment: the same as the query instance so that it is considered 
# as an immutable feature
df_train_ <- df_train |> mutate(Duration = expo_query) |> distinct()

# Predictor object for applying methods of the R package counterfactuals
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

# Uncertainty in the test set
lacp_glm <- split_cp_lacp(
  model = glm_model,
  predict_fun = predict_fun_glm,
  data_calib = df_calib,
  data_test = df_test,
  alpha = .2
)
summary(lacp_glm$width)

# Check prediction on query instance
c(glm_pred_query, predictor_glm$predict(query))

# Uncertainty of the query instance
upper_query <- lacp_glm$upper[idx_query]
width_query <- lacp_glm$width[idx_query]

#### Counterfactual Explanations ####

##### WhatIf #####

whatif_glm <- WhatIfRegr$new(
  predictor_glm,
  n_counterfactuals = 100L,
  distance_function = "gower_c",
  fixed_features = c(trad_variables, "Duration"),
  epsilon = .3,
  exposure = NULL # or "Duration"
)

ces_whatif_glm <- whatif_glm$find_counterfactuals(
  query,
  desired_outcome = c(0, .2)
)

eval_df_whatif_glm <- ces_whatif_glm$evaluate() 

all_df_whatif_glm <- eval_df_whatif_glm |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_glm$predict(eval_df_whatif_glm)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = glm_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_whatif_glm <- split_cp_lacp(
  model = glm_model,
  predict_fun = predict_fun_glm,
  data_cal = df_calib,
  data_test = all_df_whatif_glm |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )),
  alpha = .2
)

all_df_whatif_glm <- all_df_whatif_glm |> 
  left_join(
    lacp_cf_whatif_glm |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
     "Duration", trad_variables, telematics_variables
    )
  )
all_df_whatif_glm$width[which(all_df_whatif_glm$label == "query")] <- width_query
all_df_whatif_glm$upper[which(all_df_whatif_glm$label == "query")] <- upper_query

# All CFs
cf_whatif_glm <- all_df_whatif_glm |>
  filter(label == "cf") |>
  mutate(sum_dist = dist_x_interest + dist_train + dist_target) |>
  arrange(sum_dist) |>
  bind_rows(
    all_df_whatif_glm |>
      filter(label == "query")
  ) |>
  select(all_of(c("label", telematics_variables, "pred", "upper", "width")))

# Without the CP constraint
unaware_cf_whatif_glm <- cf_whatif_glm |>
  slice(1:5)
unaware_cf_whatif_glm

# With the CP constraint
aware_cf_whatif_glm <- cf_whatif_glm |>
  filter(upper <= .21)
aware_cf_whatif_glm

##### NICE #####

nice_glm <- NICERegr$new(
  predictor_glm,
  optimization = "proximity",
  x_nn_correct = TRUE,
  return_multiple = TRUE,
  finish_early = FALSE,
  distance_function = "gower_c",
  fixed_features = c(trad_variables, "Duration")
)

ces_nice_glm <- nice_glm$find_counterfactuals(
  query,
  desired_outcome = c(0, .2)
)

eval_df_nice_glm <- ces_nice_glm$evaluate() 
nrow(eval_df_nice_glm)

all_df_nice_glm <- eval_df_nice_glm |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_glm$predict(eval_df_nice_glm)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = glm_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_nice_glm <- split_cp_lacp(
  model = glm_model,
  predict_fun = predict_fun_glm,
  data_cal = df_calib,
  data_test = all_df_nice_glm |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )) |>
    distinct(),
  alpha = .2
)

all_df_nice_glm <- all_df_nice_glm |> 
  left_join(
    lacp_cf_nice_glm |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
      "Duration", trad_variables, telematics_variables
    )
  )
all_df_nice_glm$width[which(all_df_nice_glm$label == "query")] <- width_query
all_df_nice_glm$upper[which(all_df_nice_glm$label == "query")] <- upper_query

# All CFs
cf_nice_glm <- all_df_nice_glm |>
  filter(label == "cf") |>
  mutate(sum_dist = dist_x_interest + dist_train + dist_target) |>
  arrange(sum_dist) |>
  bind_rows(
    all_df_nice_glm |>
      filter(label == "query")
  ) |>
  select(all_of(c("label", telematics_variables, "pred", "upper", "width")))

# Without the CP constraint
unaware_cf_nice_glm <- cf_nice_glm |>
  slice(1:5)
unaware_cf_nice_glm

# With the CP constraint
aware_cf_nice_glm <- cf_nice_glm |>
  filter(upper < .21)
aware_cf_nice_glm

##### MOC #####
moc_glm <- MOCRegr$new(
  predictor_glm,
  fixed_features = c(trad_variables, "Duration"),
  distance_function = "gower_c",
  mu = 5,
  n_generations = 10
)

ces_moc_glm <- moc_glm$find_counterfactuals(
  query,
  desired_outcome = c(0, .2)
)

ces_moc_glm$subset_to_valid()
eval_df_moc_glm <- ces_moc_glm$evaluate() 
nrow(eval_df_moc_glm)

all_df_moc_glm <- eval_df_moc_glm |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_glm$predict(eval_df_moc_glm)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = glm_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_moc_glm <- split_cp_lacp(
  model = glm_model,
  predict_fun = predict_fun_glm,
  data_cal = df_calib,
  data_test = all_df_moc_glm |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )) |>
    distinct(),
  alpha = .2
)

all_df_moc_glm <- all_df_moc_glm |> 
  left_join(
    lacp_cf_moc_glm |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
      "Duration", trad_variables, telematics_variables
    )
  )
all_df_moc_glm$width[which(all_df_moc_glm$label == "query")] <- width_query
all_df_moc_glm$upper[which(all_df_moc_glm$label == "query")] <- upper_query

# All CFs
cf_moc_glm <- all_df_moc_glm |>
  filter(label == "cf") |>
  mutate(sum_dist = dist_x_interest + dist_train + dist_target) |>
  arrange(sum_dist) |>
  bind_rows(
    all_df_moc_glm |>
      filter(label == "query")
  ) |>
  select(all_of(c("label", telematics_variables, "pred", "upper", "width")))


# Without the CP constraint
unaware_cf_moc_glm <- cf_moc_glm |>
  slice(1:min(5, nrow(cf_moc_glm)-1))
unaware_cf_moc_glm

# With the CP constraint
aware_cf_moc_glm <- cf_moc_glm |>
  filter(upper < .21)
aware_cf_moc_glm

# We put all the CFs in common
cf_glm <- cf_whatif_glm |> 
  mutate(method = "whatif") |>
  bind_rows(
    cf_nice_glm |>
      mutate(method = "nice")
  ) |>
  bind_rows(
    cf_moc_glm |>
      mutate(method = "moc")
  ) |> 
  # keep whatif first bc first rows are kept
  distinct(Pct_drive_rush, Accel_14miles, Brake_14miles, .keep_all = TRUE)

#### Plots ####

# Preparing data to plot

data <- cf_whatif_glm

features_to_plot <- c(telematics_variables, "pred")

df <- data |>
  select(all_of(c(features_to_plot, "label"))) |>
  rename(pred. = pred)

df <- df[c(4,5,7,12,13),]

claim_query <- df |>
  filter(label == "query") |>
  pull(pred.)

# Fixed vertical range
range <- .15

# Number of modalities
n_rush <- n_distinct(df_test$Pct_drive_rush)
n_accel     <- n_distinct(df_test$Accel_14miles)
n_brake     <- n_distinct(df_test$Brake_14miles)

rush_scale <- range / (n_rush - 1)
accel_scale     <- range / (n_accel - 1)
brake_scale      <- range / (n_brake - 1)

df_to_plot <- df |>
  mutate(
    pct_rush = recode(
      as.character(Pct_drive_rush),
      "mostly_pm" = glm_pred_query,
      "both" = glm_pred_query - rush_scale,
      "mostly_am" = glm_pred_query - 2 * rush_scale,
      "none" = glm_pred_query - 3 * rush_scale
    ),
    
    accel_14 = case_when(
      Accel_14miles == 1 ~ glm_pred_query,
      Accel_14miles == 0  ~ glm_pred_query - accel_scale
    ),
    
    brake_14 = case_when(
      Brake_14miles == 1 ~ glm_pred_query,
      Brake_14miles == 0  ~ glm_pred_query - brake_scale
    )
  ) 

mat <- df_to_plot |>
  select(-c(label,Pct_drive_rush,Accel_14miles,Brake_14miles)) |>
  select(c(pct_rush, accel_14, brake_14, pred.)) |>
  as.matrix()

line_colors <- c(
  "query" = "#1F1F1F",
  "CE 1"   = "#D55E00",
  "CE 2"   = "#0072B2",
  "CE 3"   = "#009E73",
  "CE 4"   = "#CC79A7"
)

x <- seq_len(ncol(mat))

pdf("figs/telematics-decision-plot.pdf", width = 7, height = 4)

par(mar = c(5, 1, 2, 5))

plot(NA, xlim = c(0.75, ncol(mat) + 0.25), ylim = c(0.05, .25), xaxt = "n", 
     yaxt = "n", xlab = "", ylab = "", bty = "n")

abline(h = seq(0.05, .25, by = .05), col = "grey90", lty = 1)

df_to_plot$label[which(df_to_plot$label=="cf")] <- "CE"

cf_sum <- 1
for (i in seq_len(nrow(mat))) {
  label <- df_to_plot$label[i]
  if (label == "CE") {
    label <- paste(label, cf_sum)
    cf_sum = cf_sum + 1
  }
  lines(x, mat[i, ], type = "o", pch = 16, lty = 1, 
        lwd = ifelse(label == "query", 3, 2), col = line_colors[label])
}

axis(1, at = x, labels = colnames(mat), lwd = 0, lwd.ticks = 1, cex.axis = 0.9)

axis(4, at = seq(0.05, .25, by = .05), 
     labels = comma(seq(0.05, .25, by = .05)), las = 1, lwd = 0, lwd.ticks = 1,
     cex.axis = 0.85)

text(x = 1, y = mat[c(1,4), 1], 
     labels = c("mostly_pm", "mostly_am"), pos = 3, cex = 0.8)
text(x = 1, y = mat[3, 1], 
     labels = "none", pos = 1, cex = 0.8)

text(x = 2, y = mat[1, 2], labels = "no", pos = 1, cex = 0.8)
text(x = 2, y = mat[3, 2], labels = "yes", pos = 3, cex = 0.8)

text(x = 3, y = mat[2, 3], labels = "no", pos = 1, cex = 0.8)
text(x = 3, y = mat[1, 3], labels = "yes", pos = 3, cex = 0.8)


library(latex2exp)

legend_labels <- c(
  TeX("$x$"),
  TeX("$x'_{1}$"),
  TeX("$x'_{2}$"),
  TeX("$x'_{3}$"),
  TeX("$x'_{4}$")
)


legend("bottom", inset = c(0, -0.35), #legend = names(line_colors),
       legend = legend_labels,
       col = line_colors, lty = 1,lwd = c(3, 2, 2, 2, 2), 
       pch = 16, bty = "n", cex = 1, 
       horiz = TRUE, xpd = NA)

dev.off()
