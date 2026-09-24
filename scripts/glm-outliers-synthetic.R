# Import packages
library(tidyr)
library(dplyr)
library(ggplot2)
library(latex2exp)
library(tibble)
library(predictset)
library(xgboost)
library(mgcv)
library(iml)
library(wesanderson)
#devtools::install_github("fer-agathe/counterfactuals")
library(counterfactuals)

set.seed(2026)

# Colours
colors_ <- wes_palette('Rushmore1')

#### Data Simulation ####
n_in <- 500 # inliers
n_out <- 10 # outliers

# Continuous predictors
x_in <- matrix(rnorm(n_in * 2), ncol = 2)
colnames(x_in) <- c("x1", "x2")

# outliers (far in feature space)
x_out <- matrix(rnorm(n_out * 2, mean = 4, sd = 1.2), ncol = 2)
colnames(x_out) <- c("x1", "x2")

# mixture
x <- rbind(x_in, x_out)
outlier <- c(rep(0, n_in), rep(1, n_out))

# True Poisson GLM mean
eta <- -1 + .3 * x[,1] - .4 * x[,2]
mu <- exp(eta)

# Response
y <- rpois(length(mu), lambda = mu)
outlier_idx <- which(outlier == 1)
y[outlier_idx] <- sample(
  (max(y)-2):6, size = n_out, replace = TRUE
)

# Final synthetic dataset
data <- tibble(
  x1 = x[,1], 
  x2 = x[,2], 
  y = y, 
  outlier = outlier
)

data$outlier[460] = 1

# Split train/calibration/test sets:
out_rows <- which(data$outlier == 1)
in_rows <- which(data$outlier == 0)

# outlier proportions
p_out_train <- .2
p_out_cal <- .3
p_out_test <- .5

# inlier proportions
p_in_train <- .6
p_in_cal   <- .2
p_in_test  <- .2

# outlier splitting
train_out <- sample(
  out_rows,
  size = floor(p_out_train * n_out)
)
remaining_out <- setdiff(out_rows, train_out)
cal_out <- sample(
  remaining_out,
  size = floor(
    p_out_cal/(p_out_cal + p_out_test) *
      length(remaining_out)
  )
)
test_out <- setdiff(remaining_out, cal_out)

# inlier splitting
train_in <- sample(
  in_rows,
  size = floor(p_in_train * n_in)
)
remaining_in <- setdiff(in_rows, train_in)
cal_in <- sample(
  remaining_in,
  size = floor(
    p_in_cal/(p_in_cal + p_in_test) *
      length(remaining_in)
  )
)
test_in <- setdiff(remaining_in, cal_in)

# Final indices
train_idx <- c(train_out, train_in)
cal_idx   <- c(cal_out, cal_in)
test_idx  <- c(test_out, test_in)

# Datasets
data_train <- data[train_idx, ]
data_cal   <- data[cal_idx, ]
data_test  <- data[test_idx, ]

# Observed outlier proportion
table(data_train$outlier)
table(data_cal$outlier)
table(data_test$outlier)

#### GLM Poisson ####

glm_model <- glm(
  y ~ x1 + x2,
  family = poisson(),
  data = data_train
)

summary(glm_model)

glm_predict <- function(model, newdata) {
  predict(model, newdata, type = "response")
}

glm_pred_train <- glm_predict(glm_model, newdata = data_train)
glm_pred_cal <- glm_predict(glm_model, newdata = data_cal)
glm_pred_test <- glm_predict(glm_model, newdata = data_test)

#### Conformal Prediction ####

##### LACP #####

split_cp_lacp <- function(
    model,
    predict_fun,
    data_cal,
    data_test,
    alpha = .05 # coverage level
) {
  
  # Calculating conformity score values on the calib. set
  cal_pred <- predict_fun(model, data_cal)
  cal_residuals <- abs(data_cal$y - cal_pred)
  x_cal <- as.matrix(data_cal[, c("x1", "x2")])
  
  # Variance model with an XGBoost
  sigma_model <- xgboost(
    x = x_cal,
    y = cal_residuals,
    objective = "reg:squarederror",
    nrounds = 100,
    max_depth = 3,
    learning_rate = .05
  )
  
  sigma_cal <- predict(sigma_model, x_cal)
  # To prevent predicted values of residuals <= 0
  sigma_cal <- pmax(sigma_cal, 1e-6)
  
  # Conformity scores on the calib. set
  cal_scores <- cal_residuals / sigma_cal
  n_cal <- length(cal_scores)
  
  # Empirical quantile at the level 1-alpha
  qhat <- as.numeric(quantile(
    cal_scores,
    probs = ceiling((n_cal + 1) * (1 - alpha)) / n_cal
  )
  )
  
  # Calculation of the prediction sets for the test set
  test_pred <- predict_fun(model, data_test)
  # Conditional mean predictions
  x_test <- as.matrix(data_test[, c("x1", "x2")])
  # Conditional variance predictions
  sigma_test_pred <- predict(sigma_model, x_test)
  # Taking only strictly positive values for predicted residuals
  sigma_test_pred <- pmax(sigma_test_pred, 1e-6)
  # Lower bound of the prediction set (depending on x values)
  lower <- test_pred - qhat * sigma_test_pred
  upper <- test_pred + qhat * sigma_test_pred
  
  if (is.null(data_test$y)) {
    data_test$y <- rep(NA, nrow(data_test))
  }
  tibble(
    pred = test_pred,
    sigma = sigma_test_pred,
    lower = lower,
    upper = upper,
    width = 2 * qhat * sigma_test_pred
  ) |> 
    bind_cols(data_test[, c("x1", "x2","y")])
}

lacp_glm <- split_cp_lacp(
  model = glm_model,
  predict_fun = glm_predict,
  data_cal = data_cal,
  data_test = data_test
)

res_lacp_glm <- lacp_glm |> 
  mutate(outlier = data_test$outlier)

#### Uncertainty-unaware CEs  ####

##### WhatIf #####

out_query_idx <- 2
out_query <- data_test[out_query_idx, ]
out_query |> 
  mutate(
    glm_pred = glm_pred_test[out_query_idx])

res_lacp_glm[out_query_idx,] |> 
  mutate(model = "glm") |>
  select(-c(x1, x2, outlier)) |>
  select(model, pred, lower, upper, y)

# Predictor object for the R package "iml"
glm_predictor <- Predictor$new(
  model = glm_model,
  data = data_train |> select(-c(y, outlier)),
  y = data_train$y,
  predict.function = glm_predict
)

whatif_glm <- WhatIfRegr$new(
  glm_predictor,
  n_counterfactuals = 3L,
  distance_function = "gower_c"
)

ces_whatif_glm <- whatif_glm$find_counterfactuals(
  out_query |> select(-c(y, outlier)),
  desired_outcome = c(0, 1.5)
)

ces_whatif_glm$data

whatif_data_glm <- ces_whatif_glm$data
pred_whatif_glm <- ces_whatif_glm$predict()$..prediction
cf_whatif_glm <- whatif_data_glm |>
  mutate(pred = pred_whatif_glm, label = "cf") |>
  bind_rows(
    out_query |> select(-c(y, outlier)) |>
      mutate(pred = glm_pred_test[out_query_idx], label = "query")
  )
cf_whatif_glm

data_train |> 
  filter(
    x1 %in% cf_whatif_glm$x1 & x2 %in% cf_whatif_glm$x2
  )

##### Plot #####

# Smooth LACP uncertainty level curves

# The predictive uncertainty is represented by the width of the CP interval here.
fit <- gam(width ~ s(x1, x2), data = lacp_glm)
grid <- expand.grid(
  x1 = seq(min(data$x1)-.5, max(data$x1)+.5, length = 150),
  x2 = seq(min(data$x2)-.5, max(data$x2)+.5, length = 150)
)

grid$width_hat <- predict(fit, newdata = grid)

segments <- data.frame(
  x = rep(data_test$x1[out_query_idx], 3),
  y = rep(data_test$x2[out_query_idx], 3),
  xend = cf_whatif_glm$x1[1:3],
  yend = cf_whatif_glm$x2[1:3]
)

ggplot(grid, aes(x = x1, y = x2)) +
  geom_raster(aes(fill = width_hat), interpolate = TRUE) +
  geom_contour(
    aes(z = width_hat),
    color = "grey",
    alpha = 0.8,
    linewidth = .15
  ) +
  geom_point(
    data = out_query,
    aes(x = x1, y = x2, color = "Query instance", shape = "Query instance"),
    size = 4.5,
    stroke = 1,
    fill = "coral1"
  ) +
  geom_point(
    data = cf_whatif_glm |> filter(label == "cf"), #|> slice(1),
    aes(x = x1, y = x2, color = "Counterfactuals", shape = "Counterfactuals"),
    shape = 4,
    #color = "coral1",
    stroke = 1.2,
    size = 3,
    fill = "coral1"
  ) +
  geom_segment(
    data = segments,
    aes(x = x, y = y, xend = xend, yend = yend),
    linetype = "dashed",
    color = "coral1",
    linewidth = .6
  ) +
  geom_point(
    data = data,
    aes(x = x1, y = x2, color = factor(outlier), shape = "Observation"),
    size = 2
  ) +
  scale_color_manual(
    values = c(
      "0" = "black", 
      "1" = "darkred", 
      "Query instance" = "coral1", 
      "Counterfactuals" = "coral1"
    ),
    breaks = c("0", "1", "Query instance", "Counterfactuals"),
    labels = c("Inlier", "Outlier", "Query instance", "Counterfactuals"),
    name = "Observation type"
  ) +
  scale_shape_manual(
    values = c("Observation" = 16, "Query instance" = 21, "Counterfactuals" = 4),
    guide = "none"
  ) +
  scale_fill_viridis_c(
    name = "Predictive uncertainty",
    option = "viridis",
    breaks = c(1, 5, 9),
    labels = c("1", "5", "9"),
    guide = guide_colorbar(reverse = TRUE)
  ) +
  coord_cartesian(
    xlim = c(-2.5, 5.5),
    ylim = c(-3, 4.8)
  ) +
  labs(
    x = "X1",
    y = "X2"
  ) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14)
  )

ggsave("../figs/cfs-3-whatif-glm-outliers.pdf",
       width = 5.5,
       height = 3.5,
       units = "in")

#### Uncertainty-aware CE ####

# We can now calculate their uncertainty using LACP prediction set width:
lacp_glm_whatif <- split_cp_lacp(
  model = glm_model,
  predict_fun = glm_predict,
  data_cal = data_cal,
  data_test = whatif_data_glm 
)
cp_constraint_data <- lacp_glm_whatif |> filter(upper <= 1.5)
nrow(cp_constraint_data)

##### WhatIf #####

# We don't have instances satisfying the CP constraint so we calculate more CEs
whatif_glm <- WhatIfRegr$new(
  glm_predictor,
  n_counterfactuals = 6L,
  distance_function = "gower_c"
)

ces_whatif_glm <- whatif_glm$find_counterfactuals(
  out_query |> select(-c(y, outlier)),
  desired_outcome = c(0, 1.5)
)

whatif_data_glm <- ces_whatif_glm$data
lacp_glm_whatif <- split_cp_lacp(
  model = glm_model,
  predict_fun = glm_predict,
  data_cal = data_cal,
  data_test = whatif_data_glm 
)
cp_constraint_data <- lacp_glm_whatif |> filter(upper <= 1.5)
nrow(cp_constraint_data)

##### Plot #####
segments <- data.frame(
  x = rep(data_test$x1[out_query_idx], 3),
  y = rep(data_test$x2[out_query_idx], 3),
  xend = cp_constraint_data$x1[1:3],
  yend = cp_constraint_data$x2[1:3]
)

ggplot(grid, aes(x = x1, y = x2)) +
  geom_raster(aes(fill = width_hat), interpolate = TRUE) +
  geom_contour(
    aes(z = width_hat),
    color = "grey",
    alpha = 0.8,
    linewidth = .15
  ) +
  geom_point(
    data = out_query,
    aes(x = x1, y = x2, color = "Query instance", shape = "Query instance"),
    size = 4.5,
    stroke = 1,
    fill = "coral1"
  ) +
  geom_point(
    data = cp_constraint_data,
    aes(x = x1, y = x2, color = "Counterfactuals", shape = "Counterfactuals"),
    shape = 4,
    #color = "coral1",
    stroke = 1.2,
    size = 3,
    fill = "coral1"
  ) +
  geom_segment(
    data = segments,
    aes(x = x, y = y, xend = xend, yend = yend),
    linetype = "dashed",
    color = "coral1",
    linewidth = .6
  ) +
  geom_point(
    data = data,
    aes(x = x1, y = x2, color = factor(outlier), shape = "Observation"),
    size = 2
  ) +
  scale_color_manual(
    values = c(
      "0" = "black", 
      "1" = "darkred", 
      "Query instance" = "coral1", 
      "Counterfactuals" = "coral1"
    ),
    breaks = c("0", "1", "Query instance", "Counterfactuals"),
    labels = c("Inlier", "Outlier", "Query instance", "Counterfactuals"),
    name = "Observation type"
  ) +
  scale_fill_viridis_c(
    name = "Predictive uncertainty",
    option = "viridis",
    breaks = c(1, 5, 9),
    labels = c("1", "5", "9"),
    guide = guide_colorbar(reverse = TRUE)
  ) +
  scale_shape_manual(
    values = c("Observation" = 16, "Query instance" = 21, "Counterfactuals" = 4),
    guide = "none"
  ) +
  labs(
    x = "X1",
    y = "X2"
  ) +
  coord_cartesian(
    xlim = c(-2.5, 5.5),
    ylim = c(-3, 4.8)
  ) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14)
  )
ggsave(
  "../figs/cf-cp-constraint-whatif-glm-outliers.pdf",
  width = 5.5,
  height = 3.5,
  units = "in"
)
