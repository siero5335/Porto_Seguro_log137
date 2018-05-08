library(data.table)
library(Matrix)
library(caret)
library(dplyr)
library(Rtsne)
library(MLmetrics)
library(missRanger)
library(DataExplorer)
library(verification)
library(rBayesianOptimization)
library(readr)
library(lightgbm)
library(xgboost)

# Read train and test data
dtrain <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/train.csv')
dtest <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test.csv')
id <- dtest$id

#Feature engineering
dtrain[, amount_nas := rowSums(dtrain == -1, na.rm = T)]
dtrain[, high_nas := ifelse(amount_nas>4,1,0)]
dtrain[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
dtrain[, ps_car_13_ps_reg_01 := ps_car_13*ps_reg_01]
dtrain[, ps_car_13_ps_reg_02 := ps_car_13*ps_reg_02]
dtrain[, ps_ind_02_cat_ps_car_11_cat_m := ps_ind_02_cat*ps_car_11_cat]
dtrain[, ps_ind_15_ps_car_06_cat_m := ps_ind_15*ps_car_06_cat]
dtrain[, ps_car_15_ps_calc_10 := ps_car_15*ps_calc_10]
dtrain[, ps_car_01_cat_ps_calc_07 := ps_car_01_cat*ps_calc_07]
dtrain[, ps_calc_06_ps_calc_14 := ps_calc_06*ps_calc_14]
dtrain[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
dtrain[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]
dtrain[, ps_calc_bin_sum := ps_calc_15_bin + ps_calc_16_bin + ps_calc_17_bin + ps_calc_18_bin + ps_calc_19_bin + ps_calc_20_bin]


dtrain$ps_ind_14 <- NULL
dtrain$ps_ind_11_bin <- NULL
dtrain$ps_ind_13_bin <- NULL


dtest[, amount_nas := rowSums(dtest == -1, na.rm = T)]
dtest[, high_nas := ifelse(amount_nas>4,1,0)]
dtest[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
dtest[, ps_car_13_ps_reg_01 := ps_car_13*ps_reg_01]
dtest[, ps_car_13_ps_reg_02 := ps_car_13*ps_reg_02]
dtest[, ps_ind_02_cat_ps_car_11_cat_m := ps_ind_02_cat*ps_car_11_cat]
dtest[, ps_ind_15_ps_car_06_cat_m := ps_ind_15*ps_car_06_cat]
dtest[, ps_car_15_ps_calc_10 := ps_car_15*ps_calc_10]
dtest[, ps_car_01_cat_ps_calc_07 := ps_car_01_cat*ps_calc_07]
dtest[, ps_calc_06_ps_calc_14 := ps_calc_06*ps_calc_14]
dtest[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
dtest[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]
dtest[, ps_calc_bin_sum := ps_calc_15_bin + ps_calc_16_bin + ps_calc_17_bin + ps_calc_18_bin + ps_calc_19_bin + ps_calc_20_bin]

dtest$ps_ind_14 <- NULL
dtest$ps_ind_11_bin <- NULL
dtest$ps_ind_13_bin <- NULL


# collect names of all categorical variables
cat_vars <- names(dtrain)[grepl('_cat$', names(dtrain))]
bin_vars <- names(dtrain)[grepl('_bin$', names(dtrain))]

# turn categorical features into factors
dtrain[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]
dtest[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]
dtrain[, (bin_vars) := lapply(.SD, factor), .SDcols = bin_vars]
dtest[, (bin_vars) := lapply(.SD, factor), .SDcols = bin_vars]


dtrain[, c(38:56)] <- NULL
dtest[, c(37:55)] <- NULL

dtrain$ps_car_11_cat <- NULL
dtest$ps_car_11_cat <- NULL

#Extree
set.seed(71)
ctrl <- trainControl(method = "cv",
                     number = 3,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

set.seed(71)
train_grid = expand.grid(mtry = 5:15, splitrule = "extratrees")

dtrain$target <- factor(dtrain$target)
feature.names=names(dtrain)

for (f in feature.names) {
  if (class(dtrain[[f]])=="factor") {
    levels <- unique(c(dtrain[[f]]))
    dtrain[[f]] <- factor(dtrain[[f]],
                          labels=make.names(levels))
  }
}

feature.names=names(dtest)

for (f in feature.names) {
  if (class(dtest[[f]])=="factor") {
    levels <- unique(c(dtest[[f]]))
    dtest[[f]] <- factor(dtest[[f]],
                         labels=make.names(levels))
  }
}

extree_fit = train(dtrain[,3:49], dtrain$target,
                 method = "ranger", 
                 tuneGrid = train_grid,
                 trControl=ctrl,
                 metric = "ROC")

extree_predict_train_res = predict(extree_fit, dtrain[, 3:49])
extree_predict_test_res = predict(extree_fit, dtest[, 2:48])



#rf
set.seed(72)
ctrl <- trainControl(method = "cv",
                     number = 3,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

set.seed(71)
train_grid = expand.grid(mtry = 5:15, splitrule = "gini")

dtrain$target <- factor(dtrain$target)


rf_fit = train(dtrain[,3:49], dtrain$target,
                   method = "ranger", 
                   tuneGrid = train_grid,
                   trControl=ctrl,
                   metric = "ROC")

rf_predict_train_res = predict(rf_fit, dtrain[, 3:49])
rf_predict_test_res = predict(rf_fit, dtest[, 2:48])

write_csv(as.data.frame(rf_predict_train_res), "train_pred_targets_rf.csv")
write_csv(as.data.frame(rf_predict_test_res), "test_pred_targets_rf.csv")
write_csv(as.data.frame(extree_predict_train_res), "train_pred_targets_extree.csv")
write_csv(as.data.frame(extree_predict_test_res), "test_pred_targets_extree.csv")