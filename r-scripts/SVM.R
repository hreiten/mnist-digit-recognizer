rm(list=ls())
set.seed(1000)
Sys.setenv(TZ="Africa/Johannesburg")

library(ggplot2)
library(caret)
library(tikzDevice)
library(xtable)
library(e1071)

source("HelpFunctions.R")
exportspath <- "../exports/svm/"
exportspath <- "../test/"

# load data and split into train/test
data <- read.csv("../data/Train_Digits_20171108.csv")
data$Digit <- as.factor(data$Digit)

sample <- sample(1:nrow(data), 0.8*nrow(data))
train <- data[sample,]
test <- data[-sample,]




# ========================= #
# Preprocessing of the data #
# ========================= #

# find near zero variance predictors in training set
cutPredictors <- getNZVPredictors(data = train)

# remove near zero variance predictors from the data, then normalize.
train.scaled <- getScaledAndTrimmedData(data = train, cutPredictors)
test.scaled <- getScaledAndTrimmedData(data = test, cutPredictors)





# ============ #
# Applying PCA #
# ============ #

# apply PCA to training data
train.pca <- applyPCA(train.scaled)

# represent the training and testing sets with principal components
train_as_pca <- getDataAsPCA(train.scaled, train.pca, 50)
test_as_pca <- getDataAsPCA(test.scaled, train.pca, 50)

# plot variance explained per included principal components
varExpl <- summary(train.pca)$importance[2,]
varExplSummed <- cumsum(varExpl)
varExpl_df <- data.frame(Comp = 1:length(train.pca$sdev), VarExpl = varExpl, CumVarExpl = varExplSummed)

pl <- ggplot(varExpl_df[1:100,]) +
  geom_point(aes(x = Comp, y = CumVarExpl)) +
  scale_y_continuous(limits=c(0,1)) +
  theme_bw() + xlab("Number of components") + ylab("Summed variance explained")
exportPlotToLatex(pl, exportspath, "svm_PCA_varExplainedTop100.tex")

# plot the data described by the first two principal components
namePC1 <- paste("PC1 (",round(varExpl[1],4)*100,"\\%)",sep="")
namePC2 <- paste("PC2 (",round(varExpl[2],4)*100,"\\%)",sep="")
pl <- ggplot(data = train_as_pca) +
  geom_point(aes(x = PC1, y = PC2, col=Digit)) +
  scale_color_discrete(name = "Digit") +
  xlab(namePC1) + ylab(namePC2) +
  theme_bw()
exportPlotToLatex(pl, exportspath, "svm_digit_twoPCA.tex")





# =============================================== #
# Finding optimal number of components to include #
# =============================================== #
number_of_runs = 10
components_vec = 2:33

evaluateCVofComponents <- function(number_of_runs, components_vec){
  error_df <- data.frame(components = components_vec, cv_err = NA)
  for (i in 1:length(components_vec)){
    chosenNumberOfComponents = components_vec[i]

    # train the model
    train_as_pca = getDataAsPCA(scaledData = train.scaled, pca = train.pca, numComponents = chosenNumberOfComponents)

    # tune and fit a model on the training set
    cv_errors <- c()
    for (run in 1:number_of_runs){
      svm.tune <- tune.svm(Digit ~ ., data = train_as_pca, cost = 2, kernel = "radial")
      cv_errors <- c(cv_errors, svm.tune$best.performance)
    }

    # save the mean error measure to the error_df
    error_df$cv_err[i] <- mean(cv_errors)

    # print progress to console
    print(sprintf("Number of components: %.0f", chosenNumberOfComponents))
    print(sprintf("CV Error: %.5f", error_df$cv_err[i]))
    print("", quote = F)

  }
  return(error_df)
}

error_df <- evaluateCVofComponents(number_of_runs = number_of_runs, components_vec = components_vec)
error_df$cv_err <- as.numeric(error_df$cv_err)

# plot the CV error per included component
pl <- ggplot(error_df) +
  geom_point(aes(x = components, y = cv_err)) +
  geom_line(aes(x = components, y = cv_err)) +
  scale_x_continuous(breaks = c(seq(0,nrow(error_df),5))) +
  theme_bw() +
  xlab("Number of components") + ylab("Error") + ggtitle("")
exportPlotToLatex(pl, exportspath, "svm_errors_per_components.tex")

# based on the plot, choosing to include 25 of principal components in the model
chosenNumberOfComponents = 25

# retrain the model
train_as_pca <- getDataAsPCA(train.scaled, train.pca, chosenNumberOfComponents)
test_as_pca <- getDataAsPCA(test.scaled, train.pca, chosenNumberOfComponents)





# ==================================== #
# Finding the optimal regularization C #
# ==================================== #
# will fit a model N times for a given C and return the mean cv error for each C.

number_of_runs = 10
trainingData = train_as_pca
cost_params = seq(0.5,15,by=0.5)

compareCostParameter <- function(number_of_runs, trainingData = train, cost_params){

  errors <- matrix(0L, nrow = length(cost_params), ncol = number_of_runs, byrow=T,
                   dimnames = list(paste0("C",cost_params),paste0("run",1:number_of_runs)))
  for (run in 1:number_of_runs){
    print(run)
    grid <- tune.svm(Digit ~ ., data = trainingData, cost = cost_params, kernel ="radial")
    df <- data.frame(cost = grid$performances$cost, cv_err = grid$performances$error)
    errors[,run] = df$cv_err
  }

  avg_errors <- data.frame(cost = cost_params, cv_err = rowMeans(errors))
  bestCost = avg_errors$cost[which(avg_errors$cv_err == min(avg_errors$cv_err))]

  return(list(avg_errors = avg_errors, bestCost = bestCost))
}

cost_errors <- compareCostParameter(number_of_runs, trainingData, cost_params)
bestCost = cost_errors$bestCost

# plot the expected CV error per C
pl <- ggplot(cost_errors$avg_errors) +
  geom_line(aes(x = cost, y = cv_err)) +
  geom_point(aes(x = cost, y = cv_err)) +
  geom_point(aes(x = bestCost, y = cv_err[which(cv_err == min(cv_err))]), col="red") +
  geom_vline(xintercept = bestCost, linetype = "dotted", col ="red") +
  scale_x_continuous(breaks=c(bestCost,seq(0,15,by=5))) +
  xlab("C") + ylab("CV Error") + ggtitle("") +
  theme_bw()
exportPlotToLatex(pl, exportspath, "svm_CvsCVerr.tex")






# ======================================== #
# Fitting a new model with best parameters #
# ======================================== #

# fit a model with the best cost parameter
svm.fit <- svm(Digit ~ ., data = train_as_pca, cost = bestCost, kernel="radial")

# evaluate the fit on the testing set
pred <- predict(svm.fit, test_as_pca, type="response")
pred_df <- data.frame(pred = pred, true = test$Digit)

confM <- makeConfusionMatrix(pred,test$Digit)
exportTableToLatex(confM, exportspath, "svm_confusionmatrix.tex")


# write csv files to store the data
write.table(error_df, file = paste(exportspath, "svm_cv_error.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
write.table(cost_errors$avg_errors, file = paste(exportspath, "svm_cost_error.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
write.table(pred_df, file = paste(exportspath, "svm_predictions.csv", sep=""), row.names=F, col.names=T, sep = ",", append=F)
