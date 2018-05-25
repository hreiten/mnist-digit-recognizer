rm(list=ls(all = T))
set.seed(1000)
Sys.setenv(TZ="Africa/Johannesburg")

library(ggplot2)
library(tikzDevice)
library(xtable)
library(caret)
library(gbm)

source("HelpFunctions.R")
exportspath <- "../exports/tree_based_methods/boosting/"

# read in data
data <- read.csv("../data/Train_Digits_20171108.csv")
data$Digit <- as.factor(data$Digit)

sample <- sample(1:nrow(data), 0.8*nrow(data))
train <- data[sample,]
test <- data[-sample,]

n.trees = 10000
cv.folds = 4
int.depth = 2
n.cores = 4
shrinkage = 0.01

# remove insignificant predictors to speed things up
cutPredictors <- getNZVPredictors(train)
train.scaled <- getScaledAndTrimmedData(train, cutPredictors)
test.scaled <- getScaledAndTrimmedData(test, cutPredictors)

boost <- gbm(Digit ~., data = train.scaled, distribution = "multinomial",
              n.trees = n.trees, interaction.depth = int.depth,
              cv.folds = cv.folds, shrinkage = shrinkage, n.cores = n.cores)

# plot the CV error as a function of the number of trees
best.iter <- gbm.perf(boost, method="cv", plot.it = F)
err_df <- data.frame(trees = 1:length(boost$cv.error), err = boost$cv.error)
pl <- ggplot(err_df) +
  geom_line(aes(x = trees, y = err)) +
  xlab("Number of trees") + ylab("CV error") +
  theme_bw()
exportPlotToLatex(pl, exportspath, "boost_cverr.tex")


# predict values for the test set
predict_values <- function(model, newdata, n.trees = best.iter){
  pred <- predict(model, newdata = newdata, n.trees = n.trees, type="response")
  return(apply(pred, 1, function(x) which.max(x) - 1))

  # factors = 0:9
  # return(sapply(pred, function(x) factors[which.min(abs(x - factors))]))
}


pred <- predict_values(boost, newdata = test.scaled, n.trees = best.iter)
pred_df <- data.frame(pred = pred, true = test.scaled$Digit)

confM <- makeConfusionMatrix(pred, test.scaled$Digit); print(confM)
exportTableToLatex(confM, exportspath, "boost_confusionmatrix.tex")

# save data to .csv-files
write.table(err_df, file = paste(exportspath, "boost_cv_error.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
write.table(pred_df, file = paste(exportspath, "boost_predictions.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
