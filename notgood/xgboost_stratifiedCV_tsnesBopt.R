library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)
library(Rtsne)
library(MLmetrics)
library(missRanger)
library(DataExplorer)
library(verification)
library(rBayesianOptimization)

# Read train and test data
dtrain <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/train.csv')
dtest <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test.csv')
id <- dtest$id

#Feature engineering
dtrain[, amount_nas := rowSums(dtrain == -1, na.rm = T)]
dtrain[, high_nas := ifelse(amount_nas>4,1,0)]
dtrain[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
dtrain[, ps_ind_02_cat_ps_car_11_cat := ps_ind_02_cat*ps_car_11_cat]
dtrain[, ps_ind_15_ps_car_06_cat := ps_ind_15*ps_car_06_cat]
dtrain[, ps_car_15_ps_calc_10 := ps_car_15*ps_calc_10]
dtrain[, ps_car_01_cat_ps_calc_07 := ps_car_01_cat*ps_calc_07]
dtrain[, ps_calc_06_ps_calc_14 := ps_calc_06*ps_calc_14]
dtrain[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
dtrain[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]

dtrain2 <- missRanger(dtrain[ ,3:ncol(dtrain)], pmm.k = 10, num.trees = 1000, seed = 71)
dtrainPCA <- prcomp(dtrain2, scale=T)

set.seed(71)
dtraintsne <- Rtsne(as.matrix(dtrain2), dims = 3, perplexity = 30, initial_dims = 50,
                    theta = 0.5, check_duplicates = TRUE, pca = TRUE, verbose=TRUE, max_iter = 500)


dtrain$imp_ps_car_03_cat <- dtrain2$ps_car_03_cat
dtrain$imp_ps_car_05_cat <- dtrain2$ps_car_05_cat
dtrain$imp_ps_reg_03 <- dtrain2$ps_reg_03
dtrain$imp_ps_car_14 <- dtrain2$ps_car_14
dtrain$imp_ps_car_07_cat <- dtrain2$ps_car_07_cat
dtrain$imp_ps_ind_05_cat <- dtrain2$ps_ind_05_cat
dtrain$imp_ps_car_09_cat <- dtrain2$ps_car_09_cat
dtrain$imp_ps_ind_02_cat <- dtrain2$ps_ind_02_cat
dtrain$imp_ps_car_01_cat <- dtrain2$ps_car_01_cat
dtrain$imp_ps_ind_04_cat <- dtrain2$ps_ind_04_cat
dtrain$imp_ps_car_02_cat <- dtrain2$ps_car_02_cat
dtrain$ps_car_11 <- dtrain2$ps_car_11
dtrain$ps_car_12 <- dtrain2$ps_car_12


dtest[, amount_nas := rowSums(dtest == -1, na.rm = T)]
dtest[, high_nas := ifelse(amount_nas>4,1,0)]
dtest[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
dtest[, ps_ind_02_cat_ps_car_11_cat := ps_ind_02_cat*ps_car_11_cat]
dtest[, ps_ind_15_ps_car_06_cat := ps_ind_15*ps_car_06_cat]
dtest[, ps_car_15_ps_calc_10 := ps_car_15*ps_calc_10]
dtest[, ps_car_01_cat_ps_calc_07 := ps_car_01_cat*ps_calc_07]
dtest[, ps_calc_06_ps_calc_14 := ps_calc_06*ps_calc_14]
dtest[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
dtest[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]

dtest2 <- missRanger(dtest[, 3:ncol(dtest)], pmm.k = 10, num.trees = 1000, seed = 71)
dtestPCA <- prcomp(dtest2, scale=T)

set.seed(71)
dtestsne <- Rtsne(as.matrix(dtest2), dims = 3, perplexity = 30, initial_dims = 50,
                  theta = 0.5, check_duplicates = TRUE, pca = TRUE, verbose=TRUE, max_iter = 500)

dtest$imp_ps_car_03_cat <- dtest2$ps_car_03_cat
dtest$imp_ps_car_05_cat <- dtest2$ps_car_05_cat
dtest$imp_ps_reg_03 <- dtest2$ps_reg_03
dtest$imp_ps_car_14 <- dtest2$ps_car_14
dtest$imp_ps_car_07_cat <- dtest2$ps_car_07_cat
dtest$imp_ps_ind_05_cat <- dtest2$ps_ind_05_cat
dtest$imp_ps_car_09_cat <- dtest2$ps_car_09_cat
dtest$imp_ps_ind_02_cat <- dtest2$ps_ind_02_cat
dtest$imp_ps_car_01_cat <- dtest2$ps_car_01_cat
dtest$imp_ps_ind_04_cat <- dtest2$ps_ind_04_cat
dtest$imp_ps_car_02_cat <- dtest2$ps_car_02_cat
dtest$ps_car_11 <- dtest2$ps_car_11
dtest$ps_car_12 <- dtest2$ps_car_12

rm(dtrain2, dtest2);gc()

# collect names of all categorical variables
cat_vars <- names(dtrain)[grepl('_cat$|_bin$', names(dtrain))]

# turn categorical features into factors
dtrain[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]
dtest[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]

dtrain <- cbind(dtrain, dtraintsne$Y)
dtest <- cbind(dtest, dtestsne$Y)

rm(dtrainPCA, dtestPCA);gc()



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

STCV <- strrandsamplingcrossval(dtrain$target, nfolds = 3, seedno = 114514)



# Convert target factor levels to 0 = "No" and 1 = "Yes" to avoid this error when predicting class probs:
# https://stackoverflow.com/questions/18402016/error-when-i-try-to-predict-class-probabilities-in-r-caret

x_train <- dtrain[, 2:ncol(dtrain)]

x_train1 <- rbind(dtrain[STCV[[1]], 2:ncol(dtrain)], dtrain[STCV[[2]], 2:ncol(dtrain)])
x_train2 <- rbind(dtrain[STCV[[1]], 2:ncol(dtrain)], dtrain[STCV[[3]], 2:ncol(dtrain)])
x_train3 <- rbind(dtrain[STCV[[2]], 2:ncol(dtrain)], dtrain[STCV[[3]], 2:ncol(dtrain)])

x_val1 <- dtrain[STCV[[3]], 2:ncol(dtrain)]
x_val2 <- dtrain[STCV[[2]], 2:ncol(dtrain)]
x_val3 <- dtrain[STCV[[1]], 2:ncol(dtrain)]

dtrain <- xgb.DMatrix(data=data.matrix(x_train[, 2:82]), label=x_train$target)

dtrain1 <- xgb.DMatrix(data=data.matrix(x_train1[, 2:82]), label=x_train1$target)
dval1 <- xgb.DMatrix(data=data.matrix(x_val1[, 2:82]), label=x_val1$target)
dtrain2 <- xgb.DMatrix(data=data.matrix(x_train2[, 2:82]), label=x_train2$target)
dval2 <- xgb.DMatrix(data=data.matrix(x_val2[, 2:82]), label=x_val2$target)
dtrain3 <- xgb.DMatrix(data=data.matrix(x_train3[, 2:82]), label=x_train3$target)
dval3 <- xgb.DMatrix(data=data.matrix(x_val3[, 2:82]), label=x_val3$target)
dtest <- xgb.DMatrix(data=data.matrix(dtest[, 2:82]))


rm(x_train1, x_train2, x_train3, 
   x_val1, x_val2, x_val3)
gc()


#"Params tunes for xgb"
cv_folds <- KFold(x_train$target, nfolds = 3,
                  stratified = TRUE, seed = 71)
xgb_cv_bayes <- function(max.depth, min_child_weight, subsample) {
  cv <- xgb.cv(params = list(booster = "gbtree", eta = 0.01,
                             max_depth = max.depth,
                             min_child_weight = min_child_weight,
                             subsample = subsample, colsample_bytree = 0.3,
                             lambda = 1, alpha = 0.1,
                             objective = "binary:logistic",
                             eval_metric = "auc"),
               data = dtrain, nround = 2000,
               folds = cv_folds, prediction = TRUE, showsd = TRUE,
               early_stopping_round = 50, maximize = TRUE, verbose = 0)
  list(Score = cv$evaluation_log[, max(test_auc_mean)],
       Pred = cv$pred)
}

OPT_Res <- BayesianOptimization(xgb_cv_bayes,
                                bounds = list(max.depth = c(4L, 5L, 6L, 7L, 8L, 9L, 10L),
                                              min_child_weight = c(5L, 8L, 10L),
                                              subsample = c(0.5, 0.8)),
                                init_grid_dt = NULL, init_points = 10, n_iter = 20,
                                acq = "ucb", kappa = 2.576, eps = 0.0,
                                verbose = TRUE)

##Best Parameters Found: 
##Round = 10	max.depth = 4.0000	min_child_weight = 6.0000	subsample = 0.6254	Value = 0.6319                                 
                                
#"Params for xgb"
param <- list(booster="gbtree",
              objective="binary:logistic",
              max_depth = 4,
              eta = 0.01,
              gamma = 1,
              max_delta_step = 5,
              subsample = 0.63,
              colsample_bytree = 0.8,
              min_child_weight = 6
)



#"xgb model"
set.seed(71)
watchlist <- list(eval=dval1)
xgb_model1 <- xgb.train(data = dtrain1,
                        params = param,
                        nrounds = 5000,
                        eval_metric = "auc",
                        maximize = TRUE,
                        watchlist = watchlist,
                        early_stopping_round = 50,
                        verbose = 1
)

watchlist <- list(eval=dval2)
xgb_model2 <- xgb.train(data = dtrain2,
                        params = param,
                        nrounds = 5000,
                        eval_metric = "auc",
                        maximize = TRUE,
                        watchlist = watchlist,
                        early_stopping_round = 50,
                        verbose = 1
)

watchlist <- list(eval=dval3)
xgb_model3 <- xgb.train(data = dtrain3,
                        params = param,
                        nrounds = 5000,
                        eval_metric = "auc",
                        maximize = TRUE,
                        watchlist = watchlist,
                        early_stopping_round = 50,
                        verbose = 1
)

#"Predict and output csv"
preds1 <- data.table(target=predict(xgb_model1,dtest))
preds2 <- data.table(target=predict(xgb_model2,dtest))
preds3 <- data.table(target=predict(xgb_model3,dtest))

preds <- (preds1 + preds2 + preds3)/3

sub <- data.table(id, target=preds)
write.table(sub, "xgboost_stratifiedCV_tsnesBopt.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)
