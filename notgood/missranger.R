library(data.table)
library(missRanger)


# Read train and test data
train <- fread(paste0("/Users/siero5335/Desktop/Safe Driver Prediction/train.csv"), sep=",", na.strings = "", stringsAsFactors=T)
test <- fread(paste0("/Users/siero5335/Desktop/Safe Driver Prediction/test.csv"), sep=",", na.strings = "", stringsAsFactors=T)
id <- dtest$id

missdtrain <- train[, 3:59]
missdtest <- test[, 2:58]

missdtrain[missdtrain == "-1"] <-NA
missdtest[missdtest == "-1"] <-NA

missdtrain <- missRanger(missdtrain, pmm.k = 3, num.trees = 100, seed = 71)
missdtest <- missRanger(missdtest, pmm.k = 3, num.trees = 100, seed = 71)

write.csv(missdtrain, "missdtrain.csv")
write.csv(missdtest, "missdtest.csv")