library(data.table)
library(missRanger)


# Read train and test data
train <- fread(paste0("/Users/siero5335/Desktop/Safe Driver Prediction/train.csv"), sep=",", na.strings = "", stringsAsFactors=T)
test <- fread(paste0("/Users/siero5335/Desktop/Safe Driver Prediction/test.csv"), sep=",", na.strings = "", stringsAsFactors=T)
id <- test$id

misstrain <- train[, 3:59]
misstest <- test[, 2:58]

misstrain[misstrain == "-1"] <-NA
misstest[misstest == "-1"] <-NA

misstrain <- missRanger(misstrain, pmm.k = 3, num.trees = 100, seed = 71)
misstrain <- data.frame(train[, 1:2], misstrain)
write.csv(misstrain, "misstrain.csv")
rm(misstrain)
gc()

misstest <- missRanger(misstest, pmm.k = 3, num.trees = 100, seed = 71)
misstest <- data.frame(test[, 1], misstest)

write.csv(misstest, "misstest.csv")