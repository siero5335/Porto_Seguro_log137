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

dtrain_xgb <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/train_pred_targets_xgboost_stratifiedCVopt3mix_oliver.csv')
dtest_xgb <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test_pred_targets_xgboost_stratifiedCVopt3mix_oliver.csv')

dtrain_light <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/train_pred_targets_lgboost_stratifiedCVopt3mix_oriver.csv')
dtest_light <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test_pred_targets_lgboost_stratifiedCVopt3mix_oriver.csv')

dtrain_ridge <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/train_pred_targets_ridge.csv')
dtest_ridge <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test_pred_targets_ridge.csv')

dtrain_rf <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/train_pred_targets_rf.csv')
dtest_rf <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test_pred_targets_rf.csv')

dtrain_extree <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/train_pred_targets_extree.csv')
dtest_extree <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test_pred_targets_extree.csv')

id <- dtest$id

dtrain <- data.frame(dtrain[, 1:2], 
                      dtrain_xgb, 
                      dtrain_light,
                      dtrain_ridge,
                      dtrain_rf,
                      dtrain_extree)

dtest <- data.frame(dtest[, 1], 
                      dtest_xgb, 
                      dtest_light,
                      dtest_ridge,
                      dtest_rf,
                      dtest_extree)


colnames(dtrain) <- c("id","target","V1","V2",
                      "V3", "V4", "V5", "V6", "V7"
                      , "V8", "V9")

colnames(dtest) <- c("id","V1","V2",
                      "V3", "V4", "V5", "V6", "V7"
                      , "V8", "V9")

set.seed(71)
ctrl <- trainControl(method = "cv",
                     number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

set.seed(71)
train_grid = expand.grid(alpha = 10^ (1:10 * -1), lambda = 10^ (1:10 * -1))

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

ridge_fit = train(target ~ .,
                  data = dtrain[, 2:10],
                  method = "glmnet", 
                  tuneGrid = train_grid,
                  trControl=ctrl,
                  metric = "ROC")

res_test_stack = predict(ridge_fit, dtest[, 2:10], type="prob")

sub <- data.table(id, target=res_test_stack$X2)
write.table(sub, "stacktest2.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)
