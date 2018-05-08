dtest <- fread('/Users/siero5335/Desktop/Safe Driver Prediction/test.csv')
id <- dtest$id

sub <- data.table(id, target=0)
write.table(sub, "0.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)
