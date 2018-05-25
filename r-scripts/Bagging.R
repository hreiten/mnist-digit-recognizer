set.seed(1000)

library(ggplot2)
library(randomForest)
library(tikzDevice)
library(caret)
library(xtable)

source("HelpFunctions.R")
exportspath <- "../exports/tree_based_methods/bagging/"

# read in data
data <- read.csv("../data/Train_Digits_20171108.csv")
data$Digit <- as.factor(data$Digit)
sample <- sample(1:nrow(data), 0.8*nrow(data))
train <- data[sample,]
test <- data[-sample,]

ntree = 1000

## fit bagging model ##
bag <- randomForest(Digit ~ ., data = train,
                    ntree = ntree,
                    mtry = (ncol(train) - 1),
                    importance=TRUE,
                    na.action = na.exclude,
                    do.trace = floor(ntree/10)
                    )

# plot the OOB error evolution
err_df <- data.frame(trees = 1:length(bag$err.rate[,"OOB"]), err = bag$err.rate[,"OOB"])
pl <- ggplot(err_df) +
  geom_line(aes(x = trees, y = err)) +
  xlab("Number of trees") + ylab("OOB error") + ggtitle("") +
  theme_bw()
exportPlotToLatex(pl, exportspath, "bag_oob_err.tex")

# predict values on test set
pred <- predict(bag, newdata = test)
pred_df <- data.frame(pred = pred, true = test$Digit)

# write confusion matrix to tex file
confM <- makeConfusionMatrix(pred,test$Digit)
exportTableToLatex(confM, exportspath, "bag_confusionmatrix.tex")

# save data to .csv-files
write.table(err_df, file = paste(exportspath, "bag_oob_error.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
write.table(pred_df, file = paste(exportspath, "bag_predictions.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
