# In this file, we calculate the Gower distance over CEs from DiCE in the 
# telematics data for post-hoc plausibility filtering.

# Import packages
library(dplyr)
library(tidyr)
library(iml)
source("scripts/utils.R")

#### First query instance ####
# DiCE results
ces_dice <- read.csv("out/flood_cf_dice_first_query.csv")
nrow(ces_dice)

##### LACP on DiCE #####
ces_dice <- ces_dice |>
  mutate(
    basement = factor(case_when(
      basement0 == 1 ~ "0",
      basement1 == 1 ~ "1",
      basement2 == 1 ~ "2",
      basement4 == 1 ~ "4"
    ), levels = c("0", "1", "2", "4")),
    elevated_building = as.factor(ifelse(elevated_buildingTRUE == 1, TRUE, FALSE)),
    contents_location = factor(case_when(
      contents_locationLowest_floor_and_above == 1 ~ "Lowest_floor_and_above",
      contents_locationUpper_floors == 1 ~ "Upper_floors",
      contents_locationMobile_home == 1 ~ "Mobile_home",
      TRUE ~ "Lowest_floor_only"
    ), levels = c("Lowest_floor_only", "Lowest_floor_and_above", "Upper_floors", "Mobile_home")),
    nb_floors = factor(case_when(
      nb_floors2 == 1 ~ "2",
      nb_floors2 == 1 ~ "3",
      TRUE ~ "1"
    ), levels = c("1", "2", "3")),
    non_profit = as.factor(ifelse(non_profitTRUE == 1, TRUE, FALSE)),
    occupancy = factor(case_when(
      occupancyMobile_home == 1 ~ "Mobile_home",
      occupancyMulti_unit_residential == 1 ~ "Multi_unit_residential",
      occupancySingle_family_residential == 1 ~ "Single_family_residential",
      TRUE ~ "Building_non_residential"
    ), levels = c("Building_non_residential", "Mobile_home", "Multi_unit_residential", "Single_family_residential")),
    primary_residence = as.factor(ifelse(primary_residenceTRUE == 1, TRUE, FALSE)),
    damage_cause = factor(case_when(
      damage_cause1 == 1 ~ "1",
      damage_cause2 == 1 ~ "2",
      damage_cause3 == 1 ~ "3",
      damage_cause4 == 1 ~ "4",
      damage_cause7 == 1 ~ "7",
      damage_causeA == 1 ~ "A",
      TRUE ~ "0"
    ), levels = c("0", "1", "2", "3", "4", "7", "9", "A")),
    rental = as.factor(ifelse(rentalTRUE == 1, TRUE, FALSE)),
    flood_zone = factor(case_when(
      flood_zoneLow_moderate_zone == 1 ~ "Low_moderate_zone",
      flood_zoneOther_A_zone == 1 ~ "Other_A_zone",
      flood_zoneV_zone == 1 ~ "V_zone",
      TRUE ~ "AE_zone"
    ), levels = c("AE_zone", "Low_moderate_zone", "Other_A_zone", "V_zone"))
  ) |>
  select(c(
    idx_query, basement, elevated_building, contents_location, nb_floors, non_profit, 
    occupancy, primary_residence, damage_cause, rental, building_age, 
    flood_zone, pred
  ))

df_train <- read.csv("out/flood_df_train.csv")
df_train <- df_train |>
  mutate(
    basement = as.factor(basement),
    elevated_building = as.factor(elevated_building),
    contents_location = factor(contents_location, levels = c(
      "Lowest_floor_only", "Lowest_floor_and_above", "Upper_floors", "Mobile_home"  
    )),
    nb_floors = as.factor(nb_floors),
    non_profit = as.factor(non_profit),
    occupancy = as.factor(occupancy),
    primary_residence = as.factor(primary_residence),
    damage_cause = factor(damage_cause, levels = c("0", "1", "2", "3", "4", "7", "9", "A")),
    rental = as.factor(rental),
    flood_zone = as.factor(flood_zone),
  )

ces_dice <- ces_dice |>
  mutate(across(
    where(is.factor),
    ~ factor(.x, levels = levels(df_train[[cur_column()]]))
  ))

# XGBoost
df_train_dummy <- as.data.frame(model.matrix( ~ 0 +. , data = df_train))
xgb_df_train <- df_train_dummy |> select(-Y)
xgb_train_mat <- xgb.DMatrix(
  as.matrix(xgb_df_train),
  label = df_train_dummy$Y
)

# XGBoost fitted model
xgb_model <- xgb.load("out/flood_xgb_model.json")

df_calib <- read.csv("out/flood_df_calib.csv")
df_test <- read.csv("out/flood_df_test.csv")

df_calib <- df_calib |>
  mutate(
    basement = as.factor(basement),
    elevated_building = as.factor(elevated_building),
    contents_location = factor(contents_location, levels = c(
      "Lowest_floor_only", "Lowest_floor_and_above", "Upper_floors", "Mobile_home"  
    )),
    nb_floors = as.factor(nb_floors),
    non_profit = as.factor(non_profit),
    occupancy = as.factor(occupancy),
    primary_residence = as.factor(primary_residence),
    damage_cause = factor(damage_cause, levels = c("0", "1", "2", "3", "4", "7", "9", "A")),
    rental = as.factor(rental),
    flood_zone = as.factor(flood_zone),
  )

df_test <- df_test |>
  mutate(
    basement = as.factor(basement),
    elevated_building = as.factor(elevated_building),
    contents_location = factor(contents_location, levels = c(
      "Lowest_floor_only", "Lowest_floor_and_above", "Upper_floors", "Mobile_home"  
    )),
    nb_floors = as.factor(nb_floors),
    non_profit = as.factor(non_profit),
    occupancy = as.factor(occupancy),
    primary_residence = as.factor(primary_residence),
    damage_cause = factor(damage_cause, levels = c("0", "1", "2", "3", "4", "7", "9", "A")),
    rental = as.factor(rental),
    flood_zone = as.factor(flood_zone),
  )

df_calib_dummy <- as.data.frame(model.matrix( ~ 0 +. , data = df_calib))
xgb_df_calib <- df_calib_dummy |> select(-Y)
xgb_calib_mat <- xgb.DMatrix(
  as.matrix(xgb_df_calib),
  label = df_calib_dummy$Y
)
xgb_pred_calib <- predict(xgb_model, newdata = xgb_calib_mat)

df_test_dummy <- as.data.frame(model.matrix( ~ 0 +. , data = df_test))
xgb_df_test <- df_test_dummy |> select(-Y)
xgb_test_mat <- xgb.DMatrix(
  as.matrix(xgb_df_test),
  label = df_test_dummy$Y
)
xgb_pred_test <- predict(xgb_model, newdata = xgb_test_mat)

# Predictions on the training set
xgb_pred_train <- predict(xgb_model, newdata = xgb_train_mat)

# Predictor functions
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
  data = df_train |> select(-Y),
  y = df_train$Y,
  predict.function = predict_fun_xgb
)

# Uncertainty on the CEs
data_dice <- ces_dice |>
  drop_na() |>
  select(-pred)

lacp_dice <- split_cp_lacp(
  xgb_model, 
  predict_fun_xgb, 
  df_calib,
  data_dice |> select(-idx_query),
  alpha = .2
)

indices <- unique(ces_dice$idx_query)
df_y_u <- data.frame(
  idx_query = indices,
  y_u = .9 * xgb_pred_test[indices]
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
  x_interest = df_test[indices,] |> select(-Y), 
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
    "basement", "elevated_building", "contents_location",
    "reliable", "idx_query"
  ))
res_dice

ordered_res_dice <- res_dice |> arrange(dist_train) |> slice(1:5)
max_dist <- max(ordered_res_dice$dist_train)

res_dice |> filter(dist_train <= max_dist)

#### Second query instance ####
# DiCE results
ces_dice <- read.csv("out/flood_cf_dice_second_query.csv")
nrow(ces_dice)

##### LACP on DiCE #####
ces_dice <- ces_dice |>
  mutate(
    basement = factor(case_when(
      basement0 == 1 ~ "0",
      basement1 == 1 ~ "1",
      basement2 == 1 ~ "2",
      basement4 == 1 ~ "4"
    ), levels = c("0", "1", "2", "4")),
    elevated_building = as.factor(ifelse(elevated_buildingTRUE == 1, TRUE, FALSE)),
    contents_location = factor(case_when(
      contents_locationLowest_floor_and_above == 1 ~ "Lowest_floor_and_above",
      contents_locationUpper_floors == 1 ~ "Upper_floors",
      contents_locationMobile_home == 1 ~ "Mobile_home",
      TRUE ~ "Lowest_floor_only"
    ), levels = c("Lowest_floor_only", "Lowest_floor_and_above", "Upper_floors", "Mobile_home")),
    nb_floors = factor(case_when(
      nb_floors2 == 1 ~ "2",
      nb_floors2 == 1 ~ "3",
      TRUE ~ "1"
    ), levels = c("1", "2", "3")),
    non_profit = as.factor(ifelse(non_profitTRUE == 1, TRUE, FALSE)),
    occupancy = factor(case_when(
      occupancyMobile_home == 1 ~ "Mobile_home",
      occupancyMulti_unit_residential == 1 ~ "Multi_unit_residential",
      occupancySingle_family_residential == 1 ~ "Single_family_residential",
      TRUE ~ "Building_non_residential"
    ), levels = c("Building_non_residential", "Mobile_home", "Multi_unit_residential", "Single_family_residential")),
    primary_residence = as.factor(ifelse(primary_residenceTRUE == 1, TRUE, FALSE)),
    damage_cause = factor(case_when(
      damage_cause1 == 1 ~ "1",
      damage_cause2 == 1 ~ "2",
      damage_cause3 == 1 ~ "3",
      damage_cause4 == 1 ~ "4",
      damage_cause7 == 1 ~ "7",
      damage_causeA == 1 ~ "A",
      TRUE ~ "0"
    ), levels = c("0", "1", "2", "3", "4", "7", "9", "A")),
    rental = as.factor(ifelse(rentalTRUE == 1, TRUE, FALSE)),
    flood_zone = factor(case_when(
      flood_zoneLow_moderate_zone == 1 ~ "Low_moderate_zone",
      flood_zoneOther_A_zone == 1 ~ "Other_A_zone",
      flood_zoneV_zone == 1 ~ "V_zone",
      TRUE ~ "AE_zone"
    ), levels = c("AE_zone", "Low_moderate_zone", "Other_A_zone", "V_zone"))
  ) |>
  select(c(
    idx_query, basement, elevated_building, contents_location, nb_floors, non_profit, 
    occupancy, primary_residence, damage_cause, rental, building_age, 
    flood_zone, pred
  ))

ces_dice <- ces_dice |>
  mutate(across(
    where(is.factor),
    ~ factor(.x, levels = levels(df_train[[cur_column()]]))
  ))

# Uncertainty on the CEs
data_dice <- ces_dice |>
  drop_na() |>
  select(-pred)

lacp_dice <- split_cp_lacp(
  xgb_model, 
  predict_fun_xgb, 
  df_calib,
  data_dice |> select(-idx_query),
  alpha = .2
)

indices <- unique(ces_dice$idx_query)
df_y_u <- data.frame(
  idx_query = indices,
  y_u = .9 * xgb_pred_test[indices]
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
  x_interest = df_test[indices,] |> select(-Y), 
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
    "basement", "elevated_building", "contents_location",
    "reliable", "idx_query"
  ))
res_dice

ordered_res_dice <- res_dice |> arrange(dist_train) |> slice(1:5)
max_dist <- max(ordered_res_dice$dist_train)

res_dice |> filter(dist_train <= max_dist)