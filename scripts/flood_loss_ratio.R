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
library(ranger)
library(tibble)
library(ggplot2)
library(scales)
library(shapviz)
library(lime)
# We take our extension of the R package counterfactuals 
#devtools::install_github("fer-agathe/counterfactuals")
library(counterfactuals)

source("scripts/utils.R")
set.seed(2026)

library(wesanderson)
colors_ <- wes_palette('Rushmore1')

#### Data pre-processing ####

# Load the NFIP Claims data set
#df_info <- fema_data_fields(data_set = "fimaNfipClaims")
#df_info$name
# call to the API to get the dataset
#df <- open_fema(data_set = "fimaNfipClaims", top_n = 1000)
#head(df)
# filtering data: Florida state + claims after 2023
#df_claim <- open_fema(data_set = "fimaNfipClaims",
#                      filters = list(
#                        state = "FL",
#                        yearOfLoss = ">= 2023"
#                        )
#                      )
#save(df_claim, file = "../out/fema_claims_FL.rda")

load("out/fema_claims_FL.rda")
nrow(df_claim) # 91525 rows 
ncol(df_claim) # 73 columns

# 1. We remove geographical variables except the flood area indicator.
# The removed variables are:
# "state", "reportedCity", "reportedZipCode", "countyCode", 
# "censusTract", "censusBlockGroupFips", "latitude", "longitude",
# "nfipRatedCommunityNumber", "nfipCommunityName".

# 2. We consider only gross losses related to building and content coverages. 
# The associated removed variables are: "amountPaidOnBuildingClaim", 
# "amountPaidOnContentsClaim", "amountPaidOnIncreasedCostOfComplianceClaim",
# "buildingDeductibleCode", "netBuildingPaymentAmount", 
# "contentsDeductibleCode", "netContentsPaymentAmount",  "iccCoverage", 
# "netIccPaymentAmount", "nonPaymentReasonContents", "nonPaymentReasonBuilding".

# 3. We assume all buildings have a 100% coverage and ignore the 
# insurance-coverage related variables so we remove: 
# "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage".

# 4. Other irrelevant variables in our case include: "asOfDate", 
# "crsClassificationCode", "originalNBDate", "rateMethod", "yearOfLoss",
# "ficoNumber", "floodEvent", "id", "condominiumCoverageTypeCode", 
# "replacementCostBasis".

to_remove <- c(
  "asOfDate", "crsClassificationCode", "originalNBDate", 
  "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim", 
  "amountPaidOnIncreasedCostOfComplianceClaim", "rateMethod",
  "yearOfLoss", "buildingDeductibleCode", "netBuildingPaymentAmount",
  "contentsDeductibleCode", "netContentsPaymentAmount", "ficoNumber",
  "floodEvent", "iccCoverage", "netIccPaymentAmount", 
  "nonPaymentReasonContents", "nonPaymentReasonBuilding",
  "nfipRatedCommunityNumber", "nfipCommunityName", "state", "reportedCity", 
  "reportedZipCode", "countyCode", "censusTract", "censusBlockGroupFips", 
  "latitude", "longitude", "id", "totalBuildingInsuranceCoverage", 
  "totalContentsInsuranceCoverage", "condominiumCoverageTypeCode", 
  "replacementCostBasis", "nfipCommunityNumberCurrent" 
)

df_claim <- df_claim |>
  select(-all_of(to_remove)) 
ncol(df_claim) # 39 columns left


# We preprocess the dataset by filtering observations based on certain features 
# and then removing those features. We also redefine some categorical features.
df <- df_claim |> 
  filter(
    agricultureStructureIndicator == "FALSE", # -3 obs.
    houseWorship == "FALSE", # -81 obs.
    smallBusinessIndicatorBuilding == "FALSE", # -2314 obs.
    buildingDescriptionCode %in% c("1", "8", "2", "7", "11", "3", "13", "NULL"),
    policyCount == "1",
    numberOfUnits == "1",
    stateOwnedIndicator == "FALSE"
    ) |>
  select(-c(
    "agricultureStructureIndicator", "houseWorship", "stateOwnedIndicator",
    "smallBusinessIndicatorBuilding", "policyCount", "numberOfUnits"
  )) |>
  mutate(
    buildingDescriptionCode = ifelse(
      buildingDescriptionCode == "NULL", "1", buildingDescriptionCode
      ),
    loss_month = month(dateOfLoss),
    loss_year = year(dateOfLoss),
    building_age = year(dateOfLoss) - year(originalConstructionDate)
    ) |>
  select(-c(
    "dateOfLoss", "originalConstructionDate", "postFIRMConstructionIndicator"
  ))

ncol(df) # 33 columns left
nrow(df) # 83% of obs. left (around 76000 obs.)

# We need to transform numerical-type variable into numerical variables since
# they are considered as character variables for now.
num_variables <- c(
  "elevationDifference", "baseFloodElevation", "lowestAdjacentGrade",
  "lowestFloorElevation", "numberOfFloorsInTheInsuredBuilding",
  "buildingDamageAmount", "buildingPropertyValue", "contentsDamageAmount",
  "contentsPropertyValue", "floodWaterDuration", "buildingReplacementCost",
  "contentsReplacementCost", "waterDepth", "building_age"
  )

df <- df |>
  mutate(across(all_of(num_variables), as.numeric))

# We remove variables with too many NA's.
tibble(
  variable = names(df),
  n_na = colSums(is.na(df)),
  pct_na = colMeans(is.na(df)) * 100
)
df <- df |>
  select(-c(
    "elevationDifference", "baseFloodElevation", "lowestAdjacentGrade",
    "lowestFloorElevation", "floodWaterDuration"
  ))

# We regroup the categories of some of the categorical variables.
df <- df |>
  mutate(
    basementEnclosureCrawlspaceType = ifelse(
      basementEnclosureCrawlspaceType == "NULL", "0", basementEnclosureCrawlspaceType
      ),
    flood_zone = ifelse(
      floodZoneCurrent == "NULL", ratedFloodZone, floodZoneCurrent
      )
    ) |>
  filter(causeOfDamage != "NULL") |>
  select(-c(
    "elevationCertificateIndicator", "disasterAssistanceCoverageRequired",
    "floodCharacteristicsIndicator", "ratedFloodZone", "floodZoneCurrent"
  )) |>
  mutate(
    loss_month = as.factor(loss_month),
    loss_year = as.factor(loss_year)
  )

# Replace NULL values from event designation number
unique(df$eventDesignationNumber[which(df$loss_year == 2023)])
unique(df$eventDesignationNumber[which(df$loss_year == 2024)])
unique(df$eventDesignationNumber[which(df$loss_year == 2025)])
unique(df$eventDesignationNumber[which(df$loss_year == 2026)])

df$eventDesignationNumber[which(df$loss_year == 2023)] <- "FL0123"

unique(df$loss_month[which(df$eventDesignationNumber == "FL0124")])
df$eventDesignationNumber[which(df$loss_year == 2024 & df$loss_month == 9)] <- "FL0124"

unique(df$loss_month[which(df$eventDesignationNumber == "FL0224")])
df$eventDesignationNumber[which(df$loss_year == 2024 & df$loss_month %in% c(10, 11))] <- "FL0224"

# We transform categorical variables into factor variables
df <- df |>
  mutate(
    across(where(is.character), as.factor)
    )

# We remove the variable "floodproofedIndicator" because only two obs. have the value TRUE
df <- df |>
  select(-floodproofedIndicator)

# Response variable (gross losses): we exclude ICC payments

#length(which(is.na(df$contentsPropertyValue)))
#length(which(is.na(df$contentsReplacementCost)))
#length(which(is.na(df$contentsPropertyValue) & is.na(df$contentsReplacementCost)))
#length(which(is.na(df$contentsDamageAmount)))

# We remove obs. with claims = 0 for the considered coverages here.
idx <- which(is.na(df$contentsReplacementCost) & !(is.na(df$contentsPropertyValue)))
df <- df[-idx, ]

#length(which(is.na(df$buildingPropertyValue)))
#length(which(is.na(df$buildingReplacementCost)))
#length(which(is.na(df$buildingPropertyValue) & is.na(df$buildingReplacementCost)))
#length(which(is.na(df$buildingDamageAmount)))

# We define the response variable.
df <- df |>
  filter(!(is.na(buildingDamageAmount) & is.na(contentsDamageAmount))) |>
  mutate(
    buildingDamageAmount = if_else(
      is.na(buildingDamageAmount), 0, buildingDamageAmount
    ),
    buildingPropertyValue = if_else(
      is.na(buildingPropertyValue), 0, buildingPropertyValue
    ),
    buildingReplacementCost = if_else(
      is.na(buildingReplacementCost), 0, buildingReplacementCost
    ),
    contentsDamageAmount = if_else(
      is.na(contentsDamageAmount), 0, contentsDamageAmount
    ),
    contentsPropertyValue = if_else(
      is.na(contentsPropertyValue), 0, contentsPropertyValue
    ),
    contentsReplacementCost = if_else(
      is.na(contentsReplacementCost), 0, contentsReplacementCost
    )
  ) |>
  mutate(Y = buildingDamageAmount + contentsDamageAmount) |>
  filter(Y > 0) |>
  select(-c(buildingDamageAmount, contentsDamageAmount))

#length(which(df$buildingPropertyValue == 0 & df$buildingReplacementCost > 0))
#length(which(df$contentsPropertyValue == 0 & df$contentsReplacementCost > 0))

# We do not keep both the replacement cost and the property value since they are correlated (approx. 0.85)
df <- df |>
  mutate(
    building_value = ifelse(
      buildingPropertyValue == 0 & buildingReplacementCost > 0,
      buildingReplacementCost,
      buildingPropertyValue
    ),
    contents_value = ifelse(
      contentsPropertyValue == 0 & contentsReplacementCost > 0,
      contentsReplacementCost,
      contentsPropertyValue
    )
  )

df <- df |>
  mutate(
    property_value = building_value + contents_value
  ) |>
  select(-c(
    "buildingPropertyValue", "contentsPropertyValue", "buildingReplacementCost",
    "contentsReplacementCost", "building_value", "contents_value"
  ))

# We remove obs. having contents and building claims greater than the value of the property.
idx <- which(df$property_value < df$Y)
df <- df[-idx, ]

# We assume all policies are exposed the whole years here.
# Final dataframe:
df <- df |>
  rename(
    basement = basementEnclosureCrawlspaceType,
    elevated_building = elevatedBuildingIndicator,
    contents_location = locationOfContents,
    nb_floors = numberOfFloorsInTheInsuredBuilding,
    non_profit = nonProfitIndicator,
    obstruction = obstructionType,
    occupancy = occupancyType,
    primary_residence = primaryResidenceIndicator,
    damage_cause = causeOfDamage,
    event_num = eventDesignationNumber,
    water_depth = waterDepth,
    building_code = buildingDescriptionCode,
    rental = rentalPropertyIndicator
  )

# Last preprocessing steps: removing outliers and some features.
df <- df |>
  select(-water_depth) |>
  filter(building_age >= 0) |>
  select(-building_code) |> 
  select(-obstruction) |> 
  mutate(nb_floors = as.factor(nb_floors))

# We group categories of flood zones, occupancy
df <- df |>
  mutate(
    flood_zone = case_when(
      flood_zone %in% c(
        "V", "VE", paste0("V0", 1:9), paste0("V", 10:30)
        ) ~ "V_zone",
      
      flood_zone %in% c(
        "AE", paste0("A0", 1:9), paste0("A", 10:30)
        ) ~ "AE_zone",
      
      flood_zone %in% c(
        "A", "AH", "AHB",
        "AO", "AOB", "A99"
      ) ~ "Other_A_zone",
      
      flood_zone %in% c(
        "AR", "AR Dual Zones"
      ) ~ "AR_zone",
      
      flood_zone %in% c(
        "B", "C", "X"
      ) ~ "Low_moderate_zone",
      
      flood_zone == "D" ~ "Unknown",
      
      TRUE ~ NA_character_
    )
  ) |> 
  mutate(flood_zone = as.factor(flood_zone))

df <- df |>
  mutate(
    occupancy = case_when(
      occupancy == 11 ~ "Single_family_residential",
      occupancy %in% c(15, 16) ~ "Multi_unit_residential",
      occupancy %in% c(14, 17) ~ "Mobile_home",
      occupancy %in% c(18, 19) ~ "Building_non_residential",
      TRUE ~ "Other"
    ),
    occupancy = factor(occupancy)
  )

df <- df |>
  mutate(
    nb_floors = case_when(
      occupancy == "Mobile_home" ~ "1",
      TRUE ~ nb_floors
    ),
    nb_floors = factor(nb_floors)
  )

# Variable types
num_variables <- names(df)[sapply(df, is.numeric)]
categ_variables <- setdiff(names(df), num_variables)
num_variables <- setdiff(num_variables, "Y")

# Potential actionable features
actionable_variables <- c(
  "basement", "elevated_building", "contents_location"
)

# 1: basementEnclosureCrawlspaceType: Partially actionable	Changing a 
# basement/crawlspace type requires construction work. Possible but expensive; 
# usually a long-term mitigation action.

# 2: elevatedBuildingIndicator: Elevating a building is one of the main flood 
# mitigation measures. Expensive, but actionable.

# 3: locationOfContents: Very actionable. Moving contents from basement/lower 
# levels to higher floors reduces flood losses.

# Final dataframe description
cat("Number of features:", ncol(df)-1, "\nNumber of policies:", nrow(df), "\n")

# Distribution of Y
hist(df$Y, breaks = 40)
summary(df$Y)
boxplot(df$Y)
df |> arrange(desc(Y)) |> slice(1:5) |> select(Y, event_num)

# Modeling loss ratios rather than gross losses
df <- df |> mutate(loss_ratio = Y/property_value)

# We regroup contents location categories
df$contents_location[which(df$contents_location == "NULL" & 
                             df$occupancy == "Mobile_home")] = "7"

df$contents_location[which(df$contents_location == "NULL" &
                             df$basement == "1")] = "1"
df$contents_location[which(df$contents_location == "NULL" &
                             df$basement == "4")] = "1"
df$contents_location[which(df$contents_location == "NULL" &
                             df$basement %in% c("0", "2") &
                              df$nb_floors == "1")] = "3"
df$contents_location[which(df$contents_location == "NULL" &
                             df$basement %in% c("0", "2") &
                             df$nb_floors == "2")] = "4"
df$contents_location[which(df$contents_location == "NULL" &
                             df$basement %in% c("0", "2") &
                             df$nb_floors == "3")] = "5"
df$contents_location <- droplevels(df$contents_location)

df <- df |>
  mutate(contents_location = case_when(
    contents_location == "6" ~ "Mobile_home",
    contents_location %in% c("1", "3", "7") ~ "Lowest_floor_only",
    contents_location %in% c("2", "4") ~ "Lowest_floor_and_above",
    contents_location == "5" ~ "Upper_floors",
    TRUE ~ NA_character_
  ))
df$contents_location <- factor(
  df$contents_location,
  levels = c(
    "Lowest_floor_only",
    "Lowest_floor_and_above",
    "Upper_floors",
    "Mobile_home"
  ))

#### Splitting the dataset ####
# We split the dataset into:
# a training set, to fit the severity models (GLM Gamma + XGBoost),
# a calibration set, to fit the Conformal Prediction (CP) quantities,
# a test set, to calculate counterfactual explanations and associated uncertainties.
n <- nrow(df)
# Random permutation of row indices
set.seed(2026)
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

#### XGBoost model fitting ####

df_train <- df_train |> select(-Y) |> rename(Y = loss_ratio)
df_calib <- df_calib |> select(-Y) |> rename(Y = loss_ratio)
df_test <- df_test |> select(-Y) |> rename(Y = loss_ratio)

df_train |> group_by(event_num) |> summarise(claim = mean(Y))

# Training set
df_train_ <- df_train |>
  filter(event_num != "NULL") |>
  select(-event_num, -loss_year, -loss_month)
train_property_value <- df_train_$property_value
df_train_ <- df_train_ |> select(-property_value)

# Calibration and test sets
df_calib_ <- df_calib |> 
  filter(event_num != "NULL") |>
  select(-event_num, -loss_year, -loss_month)
calib_property_value <- df_calib_$property_value
df_calib_ <- df_calib_ |> select(-property_value)

df_test_ <- df_test |> 
  #filter(event_num == "FL0124") |> 
  filter(event_num != "NULL") |>
  select(-event_num, -loss_year, -loss_month)
test_property_value <- df_test_$property_value
df_test_ <- df_test_ |> select(-property_value)

# Save for DiCE in Python
write.csv(df_train_, "out/flood_df_train.csv", row.names = FALSE)
write.csv(df_calib_, "out/flood_df_calib.csv", row.names = FALSE)
write.csv(df_test_,  "out/flood_df_test.csv",  row.names = FALSE)

# XGBoost
df_train_dummy <- as.data.frame(model.matrix( ~ 0 +. , data = df_train_))
xgb_df_train <- df_train_dummy |> select(-Y)
xgb_train_mat <- xgb.DMatrix(
  as.matrix(xgb_df_train),
  label = df_train_dummy$Y
)

# Hyperparameter grid
#grid <- expand.grid(
#  objective = "reg:gamma",
#  eval_metric = "gamma-nloglik",
#  max_depth = c(3, 6, 9),
#  eta = c(0.01, 0.05, 0.1),
#  subsample = c(0.7, 1),
#  colsample_bytree = c(0.7, 1),
#  min_child_weight = c(1, 5),
#  gamma = c(0, 1),
#  tree_method = "hist",
#  stringsAsFactors = FALSE
#)

# We can now run grid search using 5-fold CV (with parallelization)
#results <- with_progress({
#  p <- progressor(along = 1:nrow(grid))
#  pmap_dfr(grid, function(...) {
#    p()   # update progress bar
#    cv_fun(...)
#  })
#})
#save(results, file = "../out/res_param_xgb_fema_loss_ratio.RData")

load("out/res_param_xgb_fema_loss_ratio.RData") # name: results
# Select best hyperparameters from CV
best_params <- results |>
  arrange(gamma_nloglik) |>
  slice(1)
# Final hyperparameters to fit the model on all the training set
final_params <- list(
  booster = "gbtree",
  objective = best_params$objective,
  eval_metric = best_params$eval_metric,
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma,
  tree_method = best_params$tree_method
)
# XGBoost fitted model
xgb_model <- xgb.train(
  params = final_params,
  data = xgb_train_mat,
  nrounds = best_params$best_nrounds,
  verbose = 0
)
# Save model for Python
xgb.save(xgb_model, "out/flood_xgb_model.json")

# Transform the calibration and test dataframes into the XGBoost matrix for R
df_calib_dummy <- as.data.frame(model.matrix( ~ 0 +. , data = df_calib_))
xgb_df_calib <- df_calib_dummy |> select(-Y)
xgb_calib_mat <- xgb.DMatrix(
  as.matrix(xgb_df_calib),
  label = df_calib_dummy$Y
)
xgb_pred_calib <- predict(xgb_model, newdata = xgb_calib_mat)

df_test_dummy <- as.data.frame(model.matrix( ~ 0 +. , data = df_test_))
xgb_df_test <- df_test_dummy |> select(-Y)
xgb_test_mat <- xgb.DMatrix(
  as.matrix(xgb_df_test),
  label = df_test_dummy$Y
)
xgb_pred_test <- predict(xgb_model, newdata = xgb_test_mat)

# Predictions on the training set
xgb_pred_train <- predict(xgb_model, newdata = xgb_train_mat)

# Save XGB matrices for Python
write.csv(xgb_df_train, "out/flood_xgb_train_matrix.csv", row.names = FALSE)
write.csv(xgb_df_test, "out/flood_xgb_test_matrix.csv", row.names = FALSE)

#### RF model fitting ####
# 5-fold CV
#ctrl <- trainControl(
#  method = "cv",
#  number = 5
#)

# Hyperparameter grid
#grid <- expand.grid(
#  mtry = c(5, 10, 15),
#  splitrule = "variance",
#  min.node.size = c(5, 10, 20)
#)

# Train the model
#rf_model <- train(
#  Y ~ .,
#  data = df_train_,
#  method = "ranger",
#  trControl = ctrl,
#  tuneGrid = grid,
#  num.trees = 500,
#  importance = "permutation",
#  metric = "RMSE"
#)

#save(rf_model, file = "../out/rf_model_loss_ratio.RData")
#load("out/rf_model_loss_ratio.RData")

#rf_pred_train <- predict(rf_model, newdata = df_train_)
#rf_pred_calib <- predict(rf_model, newdata = df_calib_)
#rf_pred_test <- predict(rf_model, newdata = df_test_)

#### Result Evaluation ####
plot(df_test_$Y, xgb_pred_test, xlim = c(0,1), ylim = c(0,1), pch = 16)
#points(df_test_$Y, rf_pred_test, col = "darkgreen", pch = 4)
lines(c(0,7e5), c(0,7e5), lty = 2, col = "gray")

# Metrics
res_train <- evaluate_severity_model(df_train_$Y, xgb_pred_train) |>
  mutate(subset = "overall", set = "train", model = "xgb") #|>
  #bind_rows(
  #  evaluate_severity_model(df_train_$Y, rf_pred_train) |>
  #    mutate(subset = "overall", set = "train", model = "rf")
 # )
res_train

res_calib <- evaluate_severity_model(df_calib_$Y, xgb_pred_calib) |>
  mutate(subset = "overall", set = "calib", model = "xgb") #|>
  #bind_rows(
  #  evaluate_severity_model(df_calib_$Y, rf_pred_calib) |>
  #    mutate(subset = "overall", set = "calib", model = "rf")
  #)
res_calib

res_test <- evaluate_severity_model(df_test_$Y, xgb_pred_test) |>
  mutate(subset = "overall", set = "test", model = "xgb") #|>
  #bind_rows(
  #  evaluate_severity_model(df_test_$Y, rf_pred_test) |>
  #    mutate(subset = "overall", set = "test", model = "rf")
  #)
res_test

results <- res_train |>
  bind_rows(res_calib) |>
  bind_rows(res_test) |>
  select(c(set, model, subset, deviance, mae, rmse))
results

#### Conformal Prediction ####

# Prediction function for applying the methods of the R package counterfactuals
predict_fun_xgb <- function(model, newdata) {
  newdata_dummy <- as.data.frame(model.matrix( ~ 0 +. , data = newdata))
  xgb_data <- newdata_dummy
  xgb_mat <- xgb.DMatrix(
    as.matrix(xgb_data)
  )
  predict(model, xgb_mat)
}

predictor_xgb <- Predictor$new(
  model = xgb_model,
  data = df_train_ |> select(-Y),
  y = df_train_$Y,
  predict.function = predict_fun_xgb
)

# Uncertainty in the test set
lacp_xgb <- split_cp_lacp(
  model = xgb_model,
  predict_fun = predict_fun_xgb,
  data_calib = df_calib_,
  data_test = df_test_,
  alpha = .2
)
qhat <- unique(lacp_xgb$q)

#### Counterfactual Explanations ####

# Immutable features
fixed_variables <- c(
  "nb_floors", "non_profit", "occupancy", "primary_residence", "damage_cause",
  "rental", "building_age", "flood_zone"
)

##### 1st query #####
idx_query <- which(
  df_test_$Y > quantile(df_test_$Y, 0.75) &
    xgb_pred_test > quantile(xgb_pred_test, 0.75)
)[1]

query <- df_test_[idx_query,]
query
xgb_pred_query <- xgb_pred_test[idx_query]
df_test_$Y[idx_query]
query_property_value <- test_property_value[idx_query]

# Empirical CDF
F_pred_test <- ecdf(xgb_pred_test)
F_pred_test(xgb_pred_query)

# Verification of the prediction function -> CORRECT
c(xgb_pred_query, predictor_xgb$predict(query))

# Uncertainty of the query instance
upper_query <- lacp_xgb$upper[idx_query]
width_query <- lacp_xgb$width[idx_query]
# Compared to the average uncertainty
summary(lacp_xgb$width)
width_query >= quantile(lacp_xgb$width, probs = .75)
F_width_test <-  ecdf(lacp_xgb$width)
F_width_test(width_query)

###### WhatIf ######
whatif_xgb <- WhatIfRegr$new(
  predictor_xgb,
  n_counterfactuals = 100L,
  distance_function = "gower_c",
  fixed_features = fixed_variables,
  epsilon = .3
)

ces_whatif_xgb <- whatif_xgb$find_counterfactuals(
  query,
  desired_outcome = c(0, .41)
)
nrow(ces_whatif_xgb$data)
eval_df_whatif_xgb <- ces_whatif_xgb$evaluate() 

all_df_whatif_xgb <- eval_df_whatif_xgb |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_xgb$predict(eval_df_whatif_xgb)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = xgb_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_whatif_xgb <- split_cp_lacp(
  model = xgb_model,
  predict_fun = predict_fun_xgb,
  data_cal = df_calib_,
  data_test = all_df_whatif_xgb |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )) |>
    distinct(),
  alpha = .20
)

all_df_whatif_xgb <- all_df_whatif_xgb |> 
  left_join(
    lacp_cf_whatif_xgb |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
      "basement", "elevated_building", "contents_location", "nb_floors",
      "non_profit", "occupancy", "primary_residence", "damage_cause", 
      "rental", "building_age", "flood_zone"
      )
  )
all_df_whatif_xgb$width[which(all_df_whatif_xgb$label == "query")] <- width_query
all_df_whatif_xgb$upper[which(all_df_whatif_xgb$label == "query")] <- upper_query

# Without the CP constraint
unaware_cf_whatif_xgb <- all_df_whatif_xgb |>
  filter(label == "cf") |>
  mutate(sum_dist = dist_x_interest + dist_train + dist_target) |>
  arrange(sum_dist) |>
  slice(1:2) |>
  bind_rows(
    all_df_whatif_xgb |>
      filter(label == "query")
  ) |>
  select(all_of(c("label", actionable_variables, "pred", "upper", "width"))) |>
  mutate(claim = pred * query_property_value)
unaware_cf_whatif_xgb

# With the CP constraint
aware_cf_whatif_xgb <- all_df_whatif_xgb |>
  filter(label == "cf") |>
  arrange(upper) |>
  slice(1:2) |>
  bind_rows(
    all_df_whatif_xgb |>
      filter(label == "query")
  ) |>
  select(all_of(c("label", actionable_variables, "pred", "upper", "width")))|>
  mutate(claim = pred * query_property_value)
aware_cf_whatif_xgb

###### NICE ######

nice_xgb <- NICERegr$new(
  predictor_xgb,
  optimization = "proximity",
  x_nn_correct = TRUE,
  return_multiple = TRUE,
  finish_early = FALSE,
  distance_function = "gower_c",
  fixed_features = fixed_variables
)

ces_nice_xgb <- nice_xgb$find_counterfactuals(
  query,
  desired_outcome = c(0, .41)
)
nrow(ces_nice_xgb$data)

eval_df_nice_xgb <- ces_nice_xgb$evaluate() 

all_df_nice_xgb <- eval_df_nice_xgb |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_xgb$predict(eval_df_nice_xgb)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = xgb_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_nice_xgb <- split_cp_lacp(
  model = xgb_model,
  predict_fun = predict_fun_xgb,
  data_cal = df_calib_,
  data_test = all_df_nice_xgb |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )) |>
    distinct(),
  alpha = .20
)

all_df_nice_xgb <- all_df_nice_xgb |> 
  left_join(
    lacp_cf_nice_xgb |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
      "basement", "elevated_building", "contents_location", "nb_floors",
      "non_profit", "occupancy", "primary_residence", "damage_cause", 
      "rental", "building_age", "flood_zone"
    )
  )
all_df_nice_xgb$width[which(all_df_nice_xgb$label == "query")] <- width_query
all_df_nice_xgb$upper[which(all_df_nice_xgb$label == "query")] <- upper_query

# Only one CE
all_df_nice_xgb |>
  select(all_of(c("label", actionable_variables, "pred", "upper", "width")))

###### MOC ######
moc_xgb <- MOCRegr$new(
  predictor_xgb,
  fixed_features = fixed_variables,
  distance_function = "gower_c",
  p_mut = .73,
  p_mut_gen = .5,
  p_rec = .71,
  p_rec_gen = .62,
  mu = 5,
  n_generations = 10
)

ces_moc_xgb<- moc_xgb$find_counterfactuals(
  query,
  desired_outcome = c(0, .41)
)

ces_moc_xgb$subset_to_valid()
nrow(ces_moc_xgb$data)
eval_df_moc_xgb <- ces_moc_xgb$evaluate() 

all_df_moc_xgb <- eval_df_moc_xgb |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_xgb$predict(eval_df_moc_xgb)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = xgb_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_moc_xgb <- split_cp_lacp(
  model = xgb_model,
  predict_fun = predict_fun_xgb,
  data_cal = df_calib_,
  data_test = all_df_moc_xgb |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )) |>
    distinct(),
  alpha = .20
)

all_df_moc_xgb <- all_df_moc_xgb |> 
  left_join(
    lacp_cf_moc_xgb |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
      "basement", "elevated_building", "contents_location", "nb_floors",
      "non_profit", "occupancy", "primary_residence", "damage_cause", 
      "rental", "building_age", "flood_zone"
    )
  )
all_df_moc_xgb$width[which(all_df_moc_xgb$label == "query")] <- width_query
all_df_moc_xgb$upper[which(all_df_moc_xgb$label == "query")] <- upper_query

# Only two CEs
all_df_moc_xgb |>
  select(all_of(c("label", actionable_variables, "pred", "upper", "width"))) |>
  mutate(claim = pred * query_property_value)

##### 2nd query #####

idx_query <- which(
  df_test_$Y > quantile(df_test_$Y, 0.5) &
    xgb_pred_test > quantile(xgb_pred_test, 0.5) & 
    df_test_$basement == "4"
)[1]

query <- df_test_[idx_query,]
query
xgb_pred_query <- xgb_pred_test[idx_query]
df_test_$Y[idx_query]
query_property_value <- test_property_value[idx_query]

# Empirical cdf
F_pred_test(xgb_pred_query)

# Verification of the prediction function -> CORRECT
c(xgb_pred_query, predictor_xgb$predict(query))

# Uncertainty of the query instance
upper_query <- lacp_xgb$upper[idx_query]
width_query <- lacp_xgb$width[idx_query]
# Compared to the average uncertainty
summary(lacp_xgb$width)
width_query >= quantile(lacp_xgb$width, probs = .75)
F_width_test(width_query)

###### WhatIf ######
whatif_xgb <- WhatIfRegr$new(
  predictor_xgb,
  n_counterfactuals = 100L,
  distance_function = "gower_c",
  fixed_features = fixed_variables,
  epsilon = .3
)

ces_whatif_xgb <- whatif_xgb$find_counterfactuals(
  query,
  desired_outcome = c(0, .36)
)
nrow(ces_whatif_xgb$data)
eval_df_whatif_xgb <- ces_whatif_xgb$evaluate() 

all_df_whatif_xgb <- eval_df_whatif_xgb |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_xgb$predict(eval_df_whatif_xgb)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = xgb_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_whatif_xgb <- split_cp_lacp(
  model = xgb_model,
  predict_fun = predict_fun_xgb,
  data_cal = df_calib_,
  data_test = all_df_whatif_xgb |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )) |>
    distinct(),
  alpha = .2
)

all_df_whatif_xgb <- all_df_whatif_xgb |> 
  left_join(
    lacp_cf_whatif_xgb |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
      "basement", "elevated_building", "contents_location", "nb_floors",
      "non_profit", "occupancy", "primary_residence", "damage_cause", 
      "rental", "building_age", "flood_zone"
    )
  )
all_df_whatif_xgb$width[which(all_df_whatif_xgb$label == "query")] <- width_query
all_df_whatif_xgb$upper[which(all_df_whatif_xgb$label == "query")] <- upper_query

# Without the CP constraint
unaware_cf_whatif_xgb <- all_df_whatif_xgb |>
  filter(label == "cf") |>
  arrange(desc(pred)) |>
  slice(1) |>
  bind_rows(
    all_df_whatif_xgb |>
      filter(label == "query")
    ) |>
  select(all_of(c("label", actionable_variables, "pred", "upper", "width"))) |>
  mutate(claim = pred * query_property_value)
unaware_cf_whatif_xgb

# With the CP constraint
aware_cf_whatif_xgb <- all_df_whatif_xgb |>
  filter(label == "cf") |>
  arrange(upper) |>
  slice(1:4) |>
  bind_rows(
    all_df_whatif_xgb |>
      filter(label == "query")
  ) |>
  select(all_of(c("label", actionable_variables, "pred", "upper", "width")))|>
  mutate(claim = pred * query_property_value)
aware_cf_whatif_xgb

###### NICE ######

nice_xgb <- NICERegr$new(
  predictor_xgb,
  optimization = "proximity",
  x_nn_correct = TRUE,
  return_multiple = TRUE,
  finish_early = FALSE,
  distance_function = "gower_c",
  fixed_features = fixed_variables
)

ces_nice_xgb <- nice_xgb$find_counterfactuals(
  query,
  desired_outcome = c(0, .36)
)
nrow(ces_nice_xgb$data)

eval_df_nice_xgb <- ces_nice_xgb$evaluate() 

all_df_nice_xgb <- eval_df_nice_xgb |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_xgb$predict(eval_df_nice_xgb)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = xgb_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_nice_xgb <- split_cp_lacp(
  model = xgb_model,
  predict_fun = predict_fun_xgb,
  data_cal = df_calib_,
  data_test = all_df_nice_xgb |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )) |>
    distinct(),
  alpha = .2
)

all_df_nice_xgb <- all_df_nice_xgb |> 
  left_join(
    lacp_cf_nice_xgb |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
      "basement", "elevated_building", "contents_location", "nb_floors",
      "non_profit", "occupancy", "primary_residence", "damage_cause", 
      "rental", "building_age", "flood_zone"
    )
  )
all_df_nice_xgb$width[which(all_df_nice_xgb$label == "query")] <- width_query
all_df_nice_xgb$upper[which(all_df_nice_xgb$label == "query")] <- upper_query

# Only two CEs
all_df_nice_xgb |>
  slice(1) |>
  select(all_of(c("label", actionable_variables, "pred", "upper", "width")))

###### MOC ######

moc_xgb <- MOCRegr$new(
  predictor_xgb,
  fixed_features = fixed_variables,
  distance_function = "gower_c",
  p_rec = 0.71,
  p_rec_gen = 0.62,
  p_mut = 0.73,
  p_mut_gen = 0.5,
  mu = 5,
  n_generations = 10
)

ces_moc_xgb<- moc_xgb$find_counterfactuals(
  query,
  desired_outcome = c(0, .36)
)

ces_moc_xgb$subset_to_valid()
nrow(ces_moc_xgb$data)
eval_df_moc_xgb <- ces_moc_xgb$evaluate() 

all_df_moc_xgb <- eval_df_moc_xgb |>
  select(-c("no_changed", "minimality")) |>
  mutate(
    pred = predictor_xgb$predict(eval_df_moc_xgb)$..prediction,
    label = "cf"
  ) |> 
  bind_rows(
    query |> select(-Y) |> mutate(pred = xgb_pred_query, label = "query")
  )

# Associated uncertainty
lacp_cf_moc_xgb <- split_cp_lacp(
  model = xgb_model,
  predict_fun = predict_fun_xgb,
  data_cal = df_calib_,
  data_test = all_df_moc_xgb |>
    filter(label == "cf") |>
    select(-c(
      "dist_x_interest", "dist_train", "dist_target", "pred", "label"
    )) |>
    distinct(),
  alpha = .2
)

all_df_moc_xgb <- all_df_moc_xgb |> 
  left_join(
    lacp_cf_moc_xgb |> select(-c("pred", "sigma", "lower", "q", "Y")),
    by = c(
      "basement", "elevated_building", "contents_location", "nb_floors",
      "non_profit", "occupancy", "primary_residence", "damage_cause", 
      "rental", "building_age", "flood_zone"
    )
  )
all_df_moc_xgb$width[which(all_df_moc_xgb$label == "query")] <- width_query
all_df_moc_xgb$upper[which(all_df_moc_xgb$label == "query")] <- upper_query

# Only 3 CEs
all_df_moc_xgb |>
  select(all_of(c("label", actionable_variables, "pred", "upper", "width"))) |>
  mutate(claim = pred * query_property_value)

# With the R package counterfactuals
ces_whatif_xgb$plot_parallel(feature_names = actionable_variables)

# Preparing data to plot
data <- all_df_whatif_xgb |>
  bind_rows(all_df_nice_xgb) #|>
  #mutate(pred = pred * query_property_value)

features_to_plot <- c(actionable_variables, "pred")

df <- data |>
  select(all_of(c(features_to_plot, "label"))) |>
  rename(pred. = pred) |>
  distinct()

claim_query <- df |>
  filter(label == "query") |>
  pull(pred.)

# Fixed vertical range
range <- .5

# Number of modalities
n_basement <- n_distinct(df_test_$basement)
n_elev     <- n_distinct(df_test_$elevated_building)
n_loc      <- n_distinct(df_test_$contents_location)

basement_scale <- range / n_basement
elev_scale     <- range / n_elev
loc_scale      <- range / n_loc

df_to_plot <- df |>
  mutate(
    basement = recode(
      as.character(basement),
      "4" = claim_query,
      "2" = claim_query - basement_scale,
      "1" = claim_query - 2 * basement_scale,
      "0" = claim_query - 3 * basement_scale
    ),
    
    elevated_building = case_when(
      elevated_building == FALSE ~ claim_query,
      elevated_building == TRUE  ~ claim_query - elev_scale
    ),
    
    contents_location = recode(
      contents_location,
      "Lowest_floor_only"       = claim_query,
      "Lowest_floor_and_above"  = claim_query - loc_scale,
      "Upper_floors"            = claim_query - 2 * loc_scale,
      "Mobile_home"             = claim_query - 3 * loc_scale
    )
  ) |>
  slice(c(5, 10, 3, 4, 9))

mat <- df_to_plot |>
  select(-label) |>
  as.matrix()

colnames(mat)[2] <- "elevation"
colnames(mat)[3] <- "contents_loc"

line_colors <- c(
  "query" = "#1F1F1F",
  "CE 1"   = "#D55E00",
  "CE 2"   = "#0072B2",
  "CE 3"   = "#009E73",
  "CE 4"   = "#CC79A7"
)

x <- seq_len(ncol(mat))

###### Figure 3 #######

pdf("figs/flood-decision-plot.pdf", width = 7, height = 4)

par(mar = c(5, 1, 2, 5))

plot(NA, xlim = c(0.75, ncol(mat) + 0.25), ylim = c(0, .45), xaxt = "n", 
     yaxt = "n", xlab = "", ylab = "", bty = "n")

abline(h = seq(0, .4, by = .1), col = "grey90", lty = 1)

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

axis(4, at = seq(0, .4, by = .1), 
     labels = comma(seq(0,.4, by = .1)), las = 1, lwd = 0, lwd.ticks = 1,
     cex.axis = 0.85)

text(x = 1, y = mat[c(1,3,5), 1], labels = c("0", "1", "3"), pos = 3, cex = 0.8)

text(x = 2, y = mat[3, 2], labels = "no", pos = 3, cex = 0.8)
text(x = 2, y = mat[1, 2], labels = "yes", pos = 4, cex = 0.8)

text(x = 3, y = mat[3, 3]-.015, labels = "lowest floor and above", pos = 1, cex = 0.75)
text(x = 3, y = mat[5, 3], labels = "lowest floor only", pos = 3, cex = 0.75)

library(latex2exp)

legend_labels <- c(
  TeX("$x_2$"),
  TeX("$x'_{2,1}$"),
  TeX("$x'_{2,2}$"),
  TeX("$x'_{2,3}$"),
  TeX("$x'_{2,4}$")
)

legend("bottom", inset = c(0, -0.35), #legend = names(line_colors),
       legend = legend_labels,
       col = line_colors, lty = 1,lwd = c(3, 2, 2, 2, 2), 
       pch = 16, bty = "n", cex = 1, 
       horiz = TRUE, xpd = NA)

dev.off()

###### SHAP ######

# XGBoost test data without Y
X_test <- xgb_df_test
# SHAP values averaged over the entire test set 
shap_xgb <- shapviz(
  xgb_model,
  X_pred = as.matrix(X_test)
)
sv_importance(shap_xgb)

# Plot for the 2nd query instance
shap_query <- shap_xgb[idx_query]
sv_waterfall(shap_query) # log scale

# SHAP decomposition
margin <- shap_xgb$baseline + sum(shap_xgb$S[idx_query, ])
# retour sur l'échelle de Y
pred_shap <- exp(margin)

# Personalized plot to deal with Gamma regression function
# Log scale: query SHAP values
shap_values <- shap_xgb$S[idx_query, ]

# Feature names
feature_names <- names(shap_values)

# Log scale baseline
baseline_log <- shap_xgb$baseline

# Normal scale prediction
pred_query <- xgb_pred_test[idx_query]

# Top 10 most important features
top10_idx <- order(abs(shap_values), decreasing = TRUE)[1:10]

shap_values_top10 <- shap_values[top10_idx]
feature_names_top10 <- feature_names[top10_idx]

# Cumulative prediction on the original scale
cumulative_log_top10 <- c(
  baseline_log,
  baseline_log + cumsum(shap_values_top10)
)

cumulative_pred_top10 <- exp(cumulative_log_top10)

# Contributions on the original scale
contributions_top10 <- diff(cumulative_pred_top10)

####### Table 12 #######
# Create table
waterfall_table <- data.frame(
  feature = feature_names_top10,
  contribution = contributions_top10,
  contribution_cum = cumulative_pred_top10[-1]
) |>
  mutate(
    increase_decrease = case_when(
      contribution > 0 ~ "Increase",
      contribution < 0 ~ "Decrease",
      TRUE ~ "No change"
    )
  )
waterfall_table

###### LIME ######
predict_xgb <- function(model, newdata) {
  predict(
    model,
    newdata = xgb.DMatrix(as.matrix(newdata))
  )
}

explainer <- lime(
  x = X_test,
  model = xgb_model,
  predict_model = predict_xgb
)

explanation <- explain(
  x = X_test[idx_query, , drop = FALSE],
  explainer = explainer,
  n_features = 10
)

####### Table 13 #######
lime_table <- explanation |>
  arrange(desc(abs(feature_weight))) |>
  mutate(
    contribution = feature_weight,
    contribution_cum = model_intercept + cumsum(feature_weight)
  ) |>
  select(
    feature,
    contribution,
    contribution_cum
  )

lime_table
