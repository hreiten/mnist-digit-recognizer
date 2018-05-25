rm(list=ls(all = T))
set.seed(1000)
Sys.setenv(TZ="Africa/Johannesburg")

library(ggplot2)
library(randomForest)
library(tikzDevice)
library(caret)
library(xtable)

source("HelpFunctions.R")
exportspath <- "../exports/tree_based_methods/randomforest/"

# read in data
data <- read.csv("../data/Train_Digits_20171108.csv")
data$Digit <- as.factor(data$Digit)

sample <- sample(1:nrow(data), 0.8*nrow(data))
train <- data[sample,]
test <- data[-sample,]

ntree = 1000

# ## RANDOM FOREST IN PARALLEL ##
# # will not get OOB measures, so will not use it in this excercise
#
# library(doParallel)
# ncores <- detectCores()
# cl <- makeCluster(ncores)
# registerDoParallel(cl)
#
# rf <- foreach(ntree=rep(floor(ntree/ncores), ncores), .combine = combine, .packages = "randomForest") %dopar% {
#   tit_rf <- randomForest(Digit ~ ., data = train, ntree = ntree, importance=TRUE, na.action = na.exclude)
# }
# stopCluster(cl)


## RANDOM FOREST NORMAL ##
rf <- randomForest(Digit ~ ., data = train,
                   ntree = ntree,
                   importance=TRUE,
                   na.action = na.exclude,
                   do.trace = floor(ntree/10))

err_df <- data.frame(trees = 1:length(rf$err.rate[,"OOB"]), err = rf$err.rate[,"OOB"])
pl <- ggplot(err_df) +
  geom_line(aes(x = trees, y = err)) +
  xlab("Number of trees") + ylab("OOB error") + ggtitle("") +
  theme_bw()
exportPlotToLatex(pl, exportspath, "rf_oob_err.tex")


# Grow new Random Forest with optimal number of trees
ntree = 500
rf <- randomForest(Digit ~ ., data = train,
                   ntree = ntree,
                   importance=TRUE,
                   na.action = na.exclude,
                   do.trace = floor(ntree/10))


# predict values on test set
pred <- predict(rf, newdata = test)
pred_df <- data.frame(pred = pred, true = test$Digit)
confM <- makeConfusionMatrix(pred,test$Digit)
exportTableToLatex(confM, exportspath, "rf_confusionmatrix.tex")

# save data to .csv-files
write.table(err_df, file = paste(exportspath, "rf_oob_error.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
write.table(pred_df, file = paste(exportspath, "rf_predictions.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
