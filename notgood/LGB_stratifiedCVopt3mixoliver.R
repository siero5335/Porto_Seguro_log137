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
dtrain[, ps_car_02_cat_ps_reg_01 := ps_car_02_cat*ps_reg_01]
dtrain[, ps_car_04_cat_ps_reg_02 := ps_car_04_cat*ps_reg_01]
dtrain[, ps_ind_02_cat_ps_car_11_cat_m := ps_ind_02_cat*ps_car_11_cat]
dtrain[, ps_ind_15_ps_car_06_cat_m := ps_ind_15*ps_car_06_cat]
dtrain[, ps_car_15_ps_calc_10 := ps_car_15*ps_calc_10]
dtrain[, ps_car_01_cat_ps_calc_07 := ps_car_01_cat*ps_calc_07]
dtrain[, ps_calc_06_ps_calc_14 := ps_calc_06*ps_calc_14]
dtrain[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
dtrain[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]
dtrain[, ps_calc_bin_sum := ps_calc_15_bin + ps_calc_16_bin + ps_calc_17_bin + ps_calc_18_bin + ps_calc_19_bin + ps_calc_20_bin]

dtest[, amount_nas := rowSums(dtest == -1, na.rm = T)]
dtest[, high_nas := ifelse(amount_nas>4,1,0)]
dtest[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
dtest[, ps_car_02_cat_ps_reg_01 := ps_car_02_cat*ps_reg_01]
dtest[, ps_car_04_cat_ps_reg_02 := ps_car_04_cat*ps_reg_01]
dtest[, ps_ind_02_cat_ps_car_11_cat_m := ps_ind_02_cat*ps_car_11_cat]
dtest[, ps_ind_15_ps_car_06_cat_m := ps_ind_15*ps_car_06_cat]
dtest[, ps_car_15_ps_calc_10 := ps_car_15*ps_calc_10]
dtest[, ps_car_01_cat_ps_calc_07 := ps_car_01_cat*ps_calc_07]
dtest[, ps_calc_06_ps_calc_14 := ps_calc_06*ps_calc_14]
dtest[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
dtest[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]
dtest[, ps_calc_bin_sum := ps_calc_15_bin + ps_calc_16_bin + ps_calc_17_bin + ps_calc_18_bin + ps_calc_19_bin + ps_calc_20_bin]

#remove features: https://www.kaggle.com/ogrellier/noise-analysis-of-porto-seguro-s-features
dtrain$ps_calc_18_bin <- NULL
dtrain$ps_ind_13_bin <- NULL
dtrain$ps_calc_10 <- NULL
dtrain$ps_calc_01 <- NULL
dtrain$ps_calc_02 <- NULL
dtrain$ps_calc_03 <- NULL
dtrain$ps_calc_13 <- NULL
dtrain$ps_calc_08 <- NULL
dtrain$ps_calc_07 <- NULL
dtrain$ps_calc_12 <- NULL
dtrain$ps_calc_04 <- NULL
dtrain$ps_calc_17_bin <- NULL
dtrain$ps_car_10_cat <- NULL
dtrain$ps_calc_14 <- NULL
dtrain$ps_calc_11 <- NULL
dtrain$ps_calc_06 <- NULL
dtrain$ps_calc_16_bin <- NULL
dtrain$ps_calc_19_bin <- NULL
dtrain$ps_calc_20_bin <- NULL
dtrain$ps_calc_15_bin <- NULL
dtrain$ps_ind_11_bin <- NULL
dtrain$ps_ind_10_bin<- NULL

dtest$ps_calc_18_bin <- NULL
dtest$ps_ind_13_bin <- NULL
dtest$ps_calc_10 <- NULL
dtest$ps_calc_01 <- NULL
dtest$ps_calc_02 <- NULL
dtest$ps_calc_03 <- NULL
dtest$ps_calc_13 <- NULL
dtest$ps_calc_08 <- NULL
dtest$ps_calc_07 <- NULL
dtest$ps_calc_12 <- NULL
dtest$ps_calc_04 <- NULL
dtest$ps_calc_17_bin <- NULL
dtest$ps_car_10_cat <- NULL
dtest$ps_calc_14 <- NULL
dtest$ps_calc_11 <- NULL
dtest$ps_calc_06 <- NULL
dtest$ps_calc_16_bin <- NULL
dtest$ps_calc_19_bin <- NULL
dtest$ps_calc_20_bin <- NULL
dtest$ps_calc_15_bin <- NULL
dtest$ps_ind_11_bin <- NULL
dtest$ps_ind_10_bin<- NULL

dtrain[dtrain == "-1"] <-NA
dtest[dtest == "-1"] <-NA

# collect names of all categorical variables
cat_vars <- names(dtrain)[grepl('_cat$', names(dtrain))]
bin_vars <- names(dtrain)[grepl('_bin$', names(dtrain))]

# turn categorical features into factors
dtrain[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]
dtest[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]
dtrain[, (bin_vars) := lapply(.SD, factor), .SDcols = bin_vars]
dtest[, (bin_vars) := lapply(.SD, factor), .SDcols = bin_vars]

# stratifiedCV: "https://github.com/pzisis/str-random-sampling/blob/master/strrandsamplingcrossval.R"
strrandsamplingcrossval <- function(classes,nfolds = 3,data = NULL,seedno = NA) {
  ## The function performs stratified random sampling for n-fold 
  ## cross-validation in the sampling set with classes as given in the input 
  ## vector with the classes for all samples. It returns a list with the
  ## indices of the samples assigned to each of the nfolds sets.
  ## 
  ## Inputs:
  ## classes: vector with n elements, where n stands for the total number of
  ##          to be split in training and testing sets. The vector might have 
  ##          numerical or categorical data.
  ## nfolds: number of folds (sets) which the total number of samples will
  ##         be split in
  ## seedno: number of seed to reproduce the results amonf repetitive runs
  ##
  ## Output:
  ## A list with nfolds elements, each one containing the indices of the 
  ## samples assigned to the respective fold (set)
  
  if(nfolds < 1) stop("wrong number of folds")
  numSamples = length(classes)		# the number of samples
  if(nfolds > numSamples) warning("number of folds larger than number of samples, some folds will be empty")
  
  # Find the unique classes
  classesUnique = unique(classes)
  numClasses = length(classesUnique)	# number of classes
  
  # Find the indices of the folds
  id <- NULL
  for (i in 1:nfolds){
    id[[i]] <- vector()     # empty vector for the samples indices
  }
  count <- 0      # counter to help result in folds with equal number of samples
  for (i in 1:numClasses){
    classI <- classesUnique[i]
    indClass <- which(classes==classI)  
    numClassI <- length(indClass)       # number of samples of class i
    numInFold <- floor(numClassI/nfolds)      # minimum number of samples in each fold 
    if (!is.na(seedno)) set.seed(seedno+i)
    indSorted <- sample(1:numClassI, numClassI, replace=FALSE)  # suffle the samples
    
    # populate the folds
    if (numInFold > 0){
      for (j in 1:nfolds){
        id[[j]] <- c(id[[j]],indClass[indSorted[(numInFold*(j-1)+1):(numInFold*j)]])
      }
    }
    # assign the remaining samples
    if (numInFold*nfolds < numClassI){
      for (j in (numInFold*nfolds+1):numClassI){
        idSample <- (count%%nfolds) + 1
        id[[idSample]] <- c(id[[idSample]],indClass[indSorted[j]])
        count <- count + 1
      }
    }
  }
  
  id
}

STCV <- strrandsamplingcrossval(dtrain$target, nfolds = 3, seedno = 70)


# Lightboost
x_train <- dtrain[, 2:ncol(dtrain)]

x_train1 <- rbind(dtrain[STCV[[1]], 2:ncol(dtrain)], dtrain[STCV[[2]], 2:ncol(dtrain)])
x_train2 <- rbind(dtrain[STCV[[1]], 2:ncol(dtrain)], dtrain[STCV[[3]], 2:ncol(dtrain)])
x_train3 <- rbind(dtrain[STCV[[2]], 2:ncol(dtrain)], dtrain[STCV[[3]], 2:ncol(dtrain)])

x_val1 <- dtrain[STCV[[3]], 2:ncol(dtrain)]
x_val2 <- dtrain[STCV[[2]], 2:ncol(dtrain)]
x_val3 <- dtrain[STCV[[1]], 2:ncol(dtrain)]

dtrainreflgb <- lgb.Dataset(data=data.matrix(x_train[, 2:49]), label=x_train$target)

lgbtrain1 <- lgb.Dataset(data=data.matrix(x_train1[, 2:49]), label=x_train1$target)
lgbval1 <- lgb.Dataset(data=data.matrix(x_val1[, 2:49]), label=x_val1$target)
lgbtrain2 <- lgb.Dataset(data=data.matrix(x_train2[, 2:49]), label=x_train2$target)
lgbval2 <- lgb.Dataset(data=data.matrix(x_val2[, 2:49]), label=x_val2$target)
lgbtrain3 <- lgb.Dataset(data=data.matrix(x_train3[, 2:49]), label=x_train3$target)
lgbval3 <- lgb.Dataset(data=data.matrix(x_val3[, 2:49]), label=x_val3$target)
lgbtest <- lgb.Dataset(data=data.matrix(dtest[, 2:49]))

rm(x_train1, x_train2, x_train3, 
   x_val1, x_val2, x_val3)
gc()


#"Params tunes for xgb"
cv_folds <- KFold(x_train$target, nfolds = 4,
                  stratified = TRUE, seed = 71)

lgb_cv_bayes <- function(lamnda_l1,
                         feature_fraction, 
                         min_gain_to_split,
                         bagging_fraction) {
  cv <- lgb.cv(params = list(objective = "binary",  boosting_type = "gbdt", 
                             learning_rate = 0.002, tree_learner = "serial",
                             metric="auc",
                             max_depth = 4,
                             min_gain_to_split = min_gain_to_split,
                             feature_fraction = feature_fraction, 
                             bagging_fraction = bagging_fraction),
               data = dtrainreflgb, nrounds = 10000, 
               folds = cv_folds, early_stopping_rounds = 200, verbose = 0,
               categorical_feature = c(3, 5:11, 14:16, 20:29)
  )
  list(Score = 
         mean(
           c(
             max(cv$record_evals$valid$auc$eval[[1]]), 
             max(cv$record_evals$valid$auc$eval[[2]]),
             max(cv$record_evals$valid$auc$eval[[3]]),
             max(cv$record_evals$valid$auc$eval[[4]])
            )
             )
       )
}

OPT_Res <- BayesianOptimization(lgb_cv_bayes,
                                bounds = list(lamnda_l1 = c(0L, 40L),
                                              feature_fraction = c(0.5, 0.8),
                                              bagging_fraction = c(0.5, 0.8),
                                              min_gain_to_split = c(1, 3)),
                                init_grid_dt = NULL, init_points = 10, n_iter = 20,
                                acq = "ucb", kappa = 2.576, eps = 0.0,
                                verbose = TRUE)

##Best Parameters Found: 
##lamnda_l1 feature_fraction bagging_fraction min_gain_to_split     Value
##20        0.5042881        0.5024351          2.957638 0.6016684

#"Params for xgb"
params <- list(objective = "binary",  
               boosting_type = "gbdt", 
               learning_rate = 0.002, 
               tree_learner = "serial",
               metric="auc",
               num_leaves = 2048, 
               min_gain_to_split = 2.957638,
               feature_fraction = 0.5042881, 
               bagging_fraction = 0.5024351,
               lambda_l1 = 20,
               early_stopping_rounds = 100, 
               verbose = 0,
               categorical_feature = c(3, 5:11, 14:16, 20:29)
)



#"lgb model"
set.seed(71)

lgb_model1 <- lgb.train(data = lgbtrain1,
                        valids = list(eval = lgbval1),
                        params = params,
                        nrounds = 8000,
                        early_stopping_round = 100,
                        verbose = 1
)

set.seed(71)

lgb_model2 <- lgb.train(data = lgbtrain2,
                        valids = list(eval = lgbval2),
                        params = params,
                        nrounds = 10000,
                        early_stopping_round = 300,
                        verbose = 1
)

set.seed(71)
lgb_model3 <- lgb.train(data = lgbtrain3,
                        valids = list(eval = lgbval3),
                        params = params,
                        nrounds = 10000,
                        early_stopping_round = 300,
                        verbose = 1
)

#"Predict and output csv"
preds1 <- predict(lgb_model1,data = data.matrix(dtest), n = lgb_model1$best_iter)
preds2 <- predict(lgb_model2,data = data.matrix(dtest), n = lgb_model2$best_iter)
preds3 <- predict(lgb_model3,data = data.matrix(dtest), n = lgb_model3$best_iter)

preds <- (preds1 + preds2 + preds3)/3

sub <- data.table(id, target=preds)
write.table(sub, "lgb_stratifiedCVBopt3mix_oliver.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)

trainpreds1 <- predict(lgb_model1,data = data.matrix(x_train), n = lgb_model1$best_iter)
trainpreds2 <- predict(lgb_model2,data = data.matrix(x_train), n = lgb_model2$best_iter)
trainpreds3 <- predict(lgb_model3,data = data.matrix(x_train), n = lgb_model3$best_iter)

train_pred_targets_lgb <- cbind(trainpreds1, 
                                trainpreds2,
                                trainpreds3)


test_pred_targets_lgb <- cbind(preds1, 
                               preds2,
                               preds3)

write_csv(as.data.frame(train_pred_targets_lgb), "train_pred_targets_lgboost_stratifiedCVopt3mix_oriver.csv")
write_csv(as.data.frame(test_pred_targets_lgb), "test_pred_targets_lgboost_stratifiedCVopt3mix_oriver.csv")
