rm(list=ls())
set.seed(1)
Sys.setenv(TZ="Africa/Johannesburg")

library(ggplot2)
library(caret)
library(tikzDevice)
library(xtable)
library(class)

source("HelpFunctions.R")
exportspath <- "../exports/KNN/"

# read data
data <- read.csv("../data/Train_Digits_20171108.csv")
data$Digit <- as.factor(data$Digit)

sample <- sample(1:nrow(data), 0.8*nrow(data))
train <- data[sample,]
test <- data[-sample,]


# ============== #
# HELP FUNCTIONS #
# ============== #

# returns predictions found with knn()
executeKNN <- function(train, test, k, numComponents){
  # find near zero predictors and delete these
  cutPredictors <- getNZVPredictors(train)

  # normalize data and remove near zero predictors
  train.scale <- getScaledAndTrimmedData(train, cutPredictors)
  test.scale <- getScaledAndTrimmedData(test, cutPredictors)

  # apply PCA
  train.pca <- applyPCA(train.scale)

  # represent the training and testing sets with principal components
  train_as_pca <- getDataAsPCA(train.scale, train.pca, numComponents)
  test_as_pca <- getDataAsPCA(test.scale, train.pca, numComponents)

  # make predictions
  pred = knn(train = train_as_pca[,-1], test = test_as_pca[,-1], cl = train_as_pca$Digit, k = k)
  return(pred)
}

# cross validation including the use of principal components
cross_validate_pca <- function(data, num_runs, numFolds, numComponents, k){
  avg_errors <- c()
  for (run in 1:num_runs) {

    shuffle <- sample(1:nrow(data),length(1:nrow(data)))
    folds <- split(shuffle, 1:numFolds)

    errors <- c()
    for (fold in folds){
      cv.test = data[fold,]
      cv.train = data[-fold,]

      pred <- executeKNN(cv.train, cv.test, k, numComponents)
      err = 1 - confusionMatrix(pred, cv.test$Digit)$overall["Accuracy"]

      errors <- c(errors, err)
    }
    avg_errors <- c(avg_errors, mean(errors))

  }
  return(mean(avg_errors))
}

# ===================== #
# END OF HELP FUNCTIONS #
# ===================== #


# find the optimal value for k using cross validation
numComponents = 25
k_df = data.frame(k = seq(1,101,by=2), err = NA)
for (i in 1:nrow(k_df)){
  k = k_df$k[i]; print(k)
  k_df$err[i] = cross_validate_pca(data = train, num_runs = 1, numFolds = 5, numComponents = numComponents, k = k)
}

# plot the CV error for every K
pl <- ggplot(k_df) +
  geom_line(aes(x = k, y = err)) +
  geom_point(aes(x = k, y = err)) +
  xlab("K") + ylab("CV error") +
  theme_bw()
pl
exportPlotToLatex(pl, exportspath, "knn_pca_cverr.tex")

# choose the best k
optimalK = k_df$k[which.min(k_df$err)]

# make new predictions with the optimal K
pred <- executeKNN(train, test, optimalK, numComponents)
pred_df <- data.frame(pred = pred, true = test$Digit)
confM <- makeConfusionMatrix(pred, test$Digit); print(confM)
exportTableToLatex(confM, exportspath, "knn_pca_confusionmatrix.tex")

# write to .csv-files
write.table(k_df, file = paste0(exportspath, "knn_pca_cv_error.csv"), row.names=F, col.names=T, sep = ",", append=F)
write.table(pred_df, file = paste0(exportspath, "knn_pca_predictions.csv"), row.names=F, col.names=T, sep = ",", append=F)
