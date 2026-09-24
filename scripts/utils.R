#### Packages ####
library(xgboost)
library(dplyr)
library(data.table)
library(tibble)

#### Data Simulation ####

#' Simulate synthetic data with inliers and outliers.
#'
#' Generates a synthetic dataset containing six continuous
#' predictors, two categorical predictors, a Poisson response,
#' and an indicator identifying generated outliers.
#'
#' The predictors of the outliers are shifted away from the
#' distribution of the inliers. Their response values are also
#' forced to be relatively large.
#'
#' @param n_inliers Number of regular observations. Default: 500.
#' @param n_outliers Number of outlier observations. Default: 10.
#' @param seed Random seed for reproducibility. Default: 2026.
#'
#' @return A tibble containing predictors from `x1` to `x8`, response
#'   `y`, and the binary `outlier` indicator.
#'
simulate_data <- function(n_inliers = 500, n_outliers = 10, seed = 2026) {
  set.seed(seed)
  # Continuous predictors for inliers
  x_inliers <- matrix(rnorm(n_inliers * 6), ncol = 6)
  colnames(x_inliers) <- paste0("x", 1:6)
  
  # Continuous predictors for outliers
  x_outliers <- matrix(rnorm(n_outliers * 6, mean = 4, sd = 1.2), ncol = 6)
  colnames(x_outliers) <- paste0("x", 1:6)
  
  # Combine
  x <- rbind(x_inliers, x_outliers)
  outlier <- c(rep(0, n_inliers), rep(1, n_outliers))
  
  n <- nrow(x)
  
  # We add a binary categorical variable
  cat_bin <- factor(sample(c("No", "Yes"), n, replace = TRUE))
  # We add a three-level categorical variable
  cat3 <- factor(sample(c("A", "B", "C"), n, replace = TRUE))
  
  # Linear predictor
  eta <- -1 +
    0.4 * x[,1] -
    0.3 * x[,2] +
    0.2 * x[,3] -
    0.1 * x[,4] -
    0.2 * x[,5] +
    0.3 * x[,6] +
    0.5 * (cat_bin == "Yes") +
    0.2 * (cat3 == "B") -
    0.3 * (cat3 == "C")
  
  mu <- exp(eta)
  
  # Poisson response
  y <- rpois(n, lambda = mu)
  
  # Force higher counts for outliers
  outlier_idx <- which(outlier == 1)
  y[outlier_idx] <- sample(
    seq(max(y) - 4, max(y)),
    size = n_outliers,
    replace = TRUE
  )
  
  tibble(
    x1 = x[,1],
    x2 = x[,2],
    x3 = x[,3],
    x4 = x[,4],
    x5 = x[,5],
    x6 = x[,6],
    x7 = cat_bin,
    x8 = cat3,
    y = y,
    outlier = outlier
  )
}

#' Split a dataset into training, calibration, and test sets.
#'
#' @param data Dataset to split.
#' @param train_prop Proportion assigned to the training set. Default: 60%.
#' @param calib_prop Proportion assigned to the calibration set. Default: 30%.
#' @param test_prop Proportion assigned to the test set. Default: 10%.
#' @param seed Random seed for reproducibility. Default: 2026.
#'
#' @return A list containing `data_train`, `data_calib`, and
#'   `data_test`.
#'
split_data <- function(data,
                       train_prop = 0.6,
                       calib_prop = 0.3,
                       test_prop = 0.1,
                       seed = 2026) {
  
  set.seed(seed)
  
  n <- nrow(data)
  
  # Shuffle indices
  idx <- sample(n)
  
  # Sizes
  n_train <- floor(train_prop * n)
  n_calib <- floor(calib_prop * n)
  
  # Indices
  train_idx <- idx[1:n_train]
  calib_idx <- idx[(n_train + 1):(n_train + n_calib)]
  test_idx <- idx[(n_train + n_calib + 1):n]
  
  list(
    data_train = data[train_idx, , drop = FALSE],
    data_calib = data[calib_idx, , drop = FALSE],
    data_test = data[test_idx, , drop = FALSE]
  )
}

#### Model Fitting ####

#' Run cross-validation (CV) for an XGBoost model.
#'
#' Fits an XGBoost model using 5-fold CV and
#' early stopping. The function evaluates a given set of
#' hyperparameters and returns the best validation score and
#' corresponding number of boosting rounds.
#'
#' Note: `xgb_train_mat` must exist in the calling environment.
#'
#' @param objective XGBoost objective function.
#' @param eval_metric Evaluation metric used during training.
#' @param max_depth Maximum tree depth.
#' @param eta Learning rate.
#' @param subsample Fraction of observations sampled for each tree.
#' @param colsample_bytree Fraction of predictors sampled for each tree.
#' @param min_child_weight Minimum sum of instance weights required
#'   in a child.
#' @param gamma Minimum loss reduction required for a split.
#' @param tree_method XGBoost tree construction method.
#'
#' @return A tibble containing the hyperparameters, best validation
#'   score, and number of boosting rounds.
#'
cv_fun <- function(objective, 
                   eval_metric,
                   max_depth, 
                   eta,
                   subsample, 
                   colsample_bytree,
                   min_child_weight, 
                   gamma,
                   tree_method) {
  
  params_list <- list(
    booster = "gbtree",
    objective = objective,
    eval_metric = eval_metric,
    max_depth = max_depth,
    eta = eta,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    tree_method = tree_method
  )
  
  cv <- xgb.cv(
    params = params_list,
    data = xgb_train_mat,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0,
    maximize = FALSE
  )
  
  best_score <- min(cv$evaluation_log$test_gamma_nloglik_mean)
  
  tibble(
    objective = objective,
    eval_metric = eval_metric,
    max_depth = max_depth,
    eta = eta,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    tree_method = tree_method,
    gamma_nloglik = best_score,
    best_nrounds = cv$niter
  )
}

#### Conformal Prediction ####

#' Construct locally adaptive conformal prediction (LACP) intervals.
#'
#' Implements split conformal prediction (SCP) with a locally adaptive
#' scale estimate. The absolute calibration residuals are first
#' modeled as a function of the predictors using XGBoost.
#' Calibration scores are then standardized by this estimated
#' local residual scale.
#'
#' The resulting prediction interval has the form
#'
#'   prediction +/- q_hat * sigma_hat(x),
#'
#' where `sigma_hat(x)` is the predicted local residual scale.
#'
#' @param model Fitted prediction model.
#' @param predict_fun Function used to obtain predictions from
#'   `model`.
#' @param data_calib Calibration dataset. The response must be
#'   stored in a column named `Y`.
#' @param data_test Test/query dataset.
#' @param alpha Miscoverage level. Default: 0.05.
#'
#' @return A tibble containing the point prediction, estimated
#'   local scale, lower and upper prediction bounds, interval
#'   width, and conformal quantile.
#'
split_cp_lacp <- function(
    model,
    predict_fun,
    data_calib,
    data_test,
    alpha = .05 # coverage level
) {
  
  # Calculating conformity score values on the calib. set
  calib_pred <- predict_fun(model, data_calib |> select(-Y))
  calib_residuals <- abs(data_calib$Y - calib_pred)
  
  # Dummy variables
  if ("Duration" %in% names(data_calib)) {
    x_calib_dummy <- model.matrix( ~ 0 + . , data = data_calib |> select(-c(Y, Duration)))
  } else {
    x_calib_dummy <- model.matrix( ~ 0 + . , data = data_calib |> select(-Y))
  }
  # Variance model with an XGBoost
  sigma_model <- xgboost(
    x = x_calib_dummy,
    y = calib_residuals,
    objective = "reg:squarederror",
    nrounds = 100,
    max_depth = 3,
    learning_rate = .05
  )
  
  sigma_calib <- predict(sigma_model, x_calib_dummy)
  # To prevent predicted values of residuals <= 0
  sigma_calib <- pmax(sigma_calib, 1e-6)
  
  # Conformity scores on the calib. set
  calib_scores <- calib_residuals / sigma_calib
  n_calib <- length(calib_scores)
  
  # Empirical quantile at the level 1-alpha
  qhat <- as.numeric(
    quantile(
      calib_scores,
      probs = ceiling((n_calib + 1) * (1 - alpha)) / n_calib
    )
  )
  
  # Calculation of the prediction sets for the test set
  if (is.null(data_test$Y)) {
    data_test$Y <- rep(NA, nrow(data_test))
  }
  pred_test <- predict_fun(model, data_test |> select(-Y))
  if ("Duration" %in% names(data_test)) {
    x_test_dummy <- model.matrix( ~ 0 + . , data = data_test |> select(-c(Y, Duration)))
  } else {
    x_test_dummy <- model.matrix( ~ 0 + . , data = data_test |> select(-Y))
  }
  sigma_pred_test <- predict(sigma_model, x_test_dummy)
  sigma_pred_test <- pmax(sigma_pred_test, 1e-6)
  # Lower bound of the prediction set (depending on x values)
  lower <- max(0, pred_test - qhat * sigma_pred_test)
  upper <- pred_test + qhat * sigma_pred_test
  
  tibble(
    pred = pred_test,
    sigma = sigma_pred_test,
    lower = lower,
    upper = upper,
    width = 2 * qhat * sigma_pred_test,
    q = qhat
  ) |> 
    bind_cols(data_test)
}

#### Model evaluation metrics ####

#' Gamma deviance for individual observations
#'
#' Computes the Gamma deviance between observed values `y`
#' and predicted means `mu`.
#'
gamma_deviance <- function(y, mu) {
  2 * ((y - mu) / mu - log(y / mu))
}

#' Mean Gamma deviance for individual observations
#'
#' Computes the averaged Gamma deviance between observed values `y`
#' and predicted means `mu`.
#'
mean_gamma_deviance <- function(y, pred) {
  mean(gamma_deviance(y, pred))
}

#' Mean absolute error
#'
mae <- function(y, pred) {
  mean(abs(y - pred))
}

#' Root mean squared error
#'
rmse <- function(y, pred) {
  sqrt(mean((y - pred)^2))
}

#' Evaluate a severity model
#'
#' Computes MAE, RMSE, and mean Gamma deviance for a severity
#' prediction model.
#'
evaluate_severity_model <- function(y, pred) {
  tibble(
    mae = mae(y, pred),
    rmse = rmse(y, pred),
    deviance = mean_gamma_deviance(y, pred)
  )
}

#' Mean Poisson deviance
#'
#' Computes the mean Poisson deviance between observed counts
#' and predicted Poisson means.
#'
poisson_deviance <- function(y, pred) {
  pred <- pmax(pred, 1e-10)
  
  mean(
    ifelse(
      y == 0,
      2 * pred,
      2 * (y * log(y / pred) - (y - pred))
    )
  )
}

#' Evaluate a Poisson model
#'
#' Computes MAE, RMSE, and Poisson deviance.
#'
evaluate_poisson_model <- function(y, pred) {
  tibble(
    mae = mae(y, pred),
    rmse = rmse(y, pred),
    deviance = poisson_deviance(y, pred)
  )
}

#### Counterfactual Explanations ####

#' Generate and evaluate counterfactual explanations (CEs).
#'
#' Searches for CEs for a query observation,
#' evaluates the resulting counterfactuals, and computes:
#'
#'   - predicted outcome,
#'   - LACP interval upper bound,
#'   - proximity: Gower distance from the query,
#'   - plausibility: Gower distance to the training data,
#'   - sparsity: number of unchanged variables,
#'   - validity of the CEs,
#'   - computation time,
#'   - the telematics variables: Pct_drive_rush, Accel_14miles, Brake_14miles.
#'
#' @param query Query observation for which counterfactuals are sought.
#' @param pred_query Prediction for the query observation.
#' @param y_u Desired upper outcome threshold.
#' @param predictor Model/predictor object used by the counterfactual
#'   package.
#' @param predict_fun Prediction function for the underlying model.
#' @param df_calib Calibration data for LACP.
#' @param model Fitted predictive model.
#' @param ce_object Counterfactual explanation object.
#'
#' @return A tibble containing the evaluated counterfactuals.
#'
calculate_ce <- function(
    query, 
    pred_query, 
    y_u,
    predictor,
    predict_fun,
    df_calib,
    model,
    ce_object
) {
  
  start <- Sys.time()
  ces_df <- ce_object$find_counterfactuals(
    query,
    desired_outcome = c(0, y_u)
  )
  end <- Sys.time()
  time = end - start
  
  if (nrow(ces_df$data) != 0) {
    eval_df <- ces_df$evaluate()
    df <- eval_df |>
      select(-minimality) |>
      mutate(pred = predictor$predict(eval_df)$..prediction)
    # Uncertainty
    lacp_cf <- split_cp_lacp(
      model, 
      predict_fun, 
      df_calib,
      df |> select(-c(
        dist_x_interest, dist_train, dist_target, no_changed, pred
      )),
      alpha = .2
    )
    # Final df
    res_ce <- df |> 
      left_join(
        lacp_cf |> 
          select(-c("pred", "sigma", "lower", "q", "Y", "width")),
        by = c("Duration", trad_variables, telematics_variables)
      ) |>
      mutate(
        valid = if_else(dist_target == 0, 1, 0),
        time = end - start
      ) |>
      select(all_of(c("pred", "upper", "dist_train", "dist_x_interest", 
                      "no_changed", "valid", "time", telematics_variables))) |>
      mutate(time = as.numeric(time))
    res_ce
    
  } else {
    
    res_ce <- tibble(
      pred = NA,
      upper = NA,
      dist_train = NA,
      dist_x_interest = NA, 
      no_changed = NA,
      valid = NA,
      time = NA,
      Pct_drive_rush = NA,
      Accel_14miles = NA,
      Brake_14miles = NA
    )
    res_ce
    
  }
}

##### Evaluation #####

# The following functions were retrieved from the R package counterfactuals.

#' Compute Gower distances between two datasets, allowing continuous and 
#' categorical variables to be handled simultaneously. For numerical variables, 
#' the range of each variable is used to normalize the corresponding 
#' differences.
#'
#' @param x First dataset.
#' @param y Second dataset.
#' @param data Reference dataset used to compute variable ranges.
#'
#' @return A matrix of Gower distances between observations in
#'   `x` and observations in `y`.
#'
distance_function <- function (x, y, data) {
  ranges = data[, sapply(.SD, function(x) 
    ifelse(is.numeric(x), max(x, na.rm = TRUE) - min(x, na.rm = TRUE), NA)
  )]
  dists = StatMatch::gower.dist(x, y, rngs = ranges, KR.corr = FALSE)
  dists
}

#' Wrapper for the distance function.
#'
#' @param distance_function Function used to compute distances.
#' @param x First dataset.
#' @param y Second dataset.
#' @param data Optional reference dataset used for scaling.
#'
#' @return A distance matrix.
#'
eval_distance = function(distance_function, x, y, data) {
  dist_matrix = distance_function(x, y, data)
  dist_matrix
}

#' Count the number of changed variables in each counterfactual.
#'
#' @param cfactuals Counterfactual observations.
#' @param x_interest Original/query observation.
#'
#' @return A vector containing the number of changed variables
#'   for each counterfactual.
#'
count_changes = function(cfactuals, x_interest) {
  cfactuals[, sum(.SD != x_interest), by = seq_len(nrow(cfactuals))][[2L]]
}

#' Compute selected metrics for a set of counterfactuals:
#'   - proximity: distance to the original query,
#'   - sparsity: number of changed variables,
#'   - plausibility: distance to the closest training observation.
#'
#' @param data Counterfactual observations.
#' @param idx_query Index identifying the original query.
#' @param distance_function Function used to compute distances.
#' @param x_interest Original/query observation.
#' @param X_train Features from the training dataset.
#' @param measures Character vector specifying which metrics
#'   to compute.
#'
#' @return A tibble containing the requested counterfactual
#'   evaluation metrics.
#'
evaluate = function(
    data,
    idx_query,
    distance_function,
    x_interest,
    X_train,
    measures = c("dist_x_interest", "no_changed", "dist_train")
) {
  
  data_t <- as.data.table(data)
  x_interest_t <- as.data.table(x_interest)
  X_train_t <- as.data.table(X_train)
  evals <- NULL
  
  if ("dist_x_interest" %in% measures) {
    evals$dist_x_interest = as.vector(eval_distance(distance_function, data_t, x_interest_t, X_train_t))
  }
  
  if ("no_changed" %in% measures) {
    evals$no_changed = count_changes(data_t, x_interest_t)
  }
  
  if ("dist_train" %in% measures) {
    dist_matrix = eval_distance(distance_function, data_t, X_train_t, X_train_t)
    evals$dist_train = t(apply(dist_matrix, 1L, function(x) sort(x)))[, 1L]
  }
  
  evals$idx_query <- idx_query
  tibble(
    idx_query = rep(evals$idx_query, length(evals$dist_x_interest)),
    dist_x_interest = evals$dist_x_interest,
    no_changed = evals$no_changed,
    dist_train = evals$dist_train
  )
  
}