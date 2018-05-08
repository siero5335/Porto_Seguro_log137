library(data.table)
library(missRanger)
library(DataExplorer)

cat("Load data")
train <- fread(paste0("/Users/siero5335/Desktop/Safe Driver Prediction/train.csv"), sep=",", na.strings = "", stringsAsFactors=T)
test <- fread(paste0("/Users/siero5335/Desktop/Safe Driver Prediction/test.csv"), sep=",", na.strings = "", stringsAsFactors=T)

cat("Combine train and test files")
test$target <- NA
data <- rbind(train, test)
rm(train,test);gc()

cat("Feature engineering")
data[, amount_nas := rowSums(data == -1, na.rm = T)]
data[, high_nas := ifelse(amount_nas>4,1,0)]
data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
data[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]

data[data =="-1"] <- NA

PlotMissing(data[,1:64])

cat("missing treatment")
data <- missRanger(data, pmm.k = 3, num.trees = 100, seed = 71)

write.csv(data, "missing treated data,csv")