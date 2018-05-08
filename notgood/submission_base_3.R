library(data.table)
library(caret)
library(Rtsne)
library(dplyr)
library(tidyr)
library(xgboost)
library(Matrix)

train <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/train.csv')
test <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test.csv')

train_Id <- train$id
test_Id <- test$id



set.seed(71)
trainIndex <- createDataPartition(train$target, p = .6, 
                                  list = FALSE, 
                                  times = 1)

train_y <- train[ trainIndex,]$target
val_y <- train[ -trainIndex,]$target

train_x <- subset(train[ trainIndex,], select = -c(id, target))
train_x <- sparse.model.matrix(~., data = train_x)

val_x <- subset(train[ -trainIndex,], select = -c(id, target))
val_x <- sparse.model.matrix(~., data = val_x)

test_x <- subset(test, select = -c(id))
test_x <- sparse.model.matrix(~., data = test_x)

param <- list("objective" = "binary:logistic",
              "eta" = 0.02,
              "min_child_weight" = 10,
              "subsample" = .9,
              "colsample_bytree" = .9,
              "scale_pos_weight" = 3,
              "max_depth" = 4)

num_rounds <- 1000

xgtest <- xgb.DMatrix(data = test_x)
xgtrain <- xgb.DMatrix(data = train_x, label= train_y)
xgval <-  xgb.DMatrix(data = val_x, label= val_y)

watchlist <- list(train=xgtrain, val=xgval)

# build Gini functions for use in custom xgboost evaluation metric
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = (1:nrow(df))/nrow(df)
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

# wrap up into a function to be called within xgboost.train
evalgini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- NormalizedGini(as.numeric(labels),as.numeric(preds))
  return(list(metric = "Gini", value = err))
}

# Now fit again but this time evaulate using Normalized Gini
set.seed(71)
bst <- xgb.train(params = param, data = xgtrain, feval = evalgini, 
                 nround=num_rounds,  watchlist=watchlist, early_stopping_rounds = 10, maximize = TRUE)

bst$best_iteration

imp<-xgb.importance(names(xgtrain),model=bst)
print(imp)
xgb.plot.importance(imp)

preds <- predict(bst, newdata = xgtest, type = "prob")

sub <- data.frame(id = test$id, target = preds)


write.csv(sub, 'submission_xggini448.csv', row.names = FALSE)
