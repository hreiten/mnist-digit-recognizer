require(caret)
require(ggplot2)
require(tikzDevice)
require(xtable)


# ====================== #
# GENERAL HELP FUNCTIONS #
# ====================== #

makeConfusionMatrix <- function(pred, true){
  mod <- confusionMatrix(pred,true)
  confM <- mod$table
  confM <- rbind(confM, Totals = colSums(confM))
  sums <- rowSums(confM)

  confM <- as.data.frame.matrix(confM)
  confM <- cbind(confM, Error = 0, Rate = NA)
  correct_vec = numeric(nrow(confM))
  wrong_vec = numeric(nrow(confM))

  for (row in 1:(nrow(confM)-1)){
    correct <- as.numeric(confM[row,row])
    wrong <- sums[row] - correct
    correct_vec[row] = correct
    wrong_vec[row] = wrong

    confM$Error[row] = round(wrong/(correct+wrong),5)
    confM$Rate[row] = paste0(wrong, " / ", (correct+wrong))
  }
  confM$Error[nrow(confM)] <- sum(wrong_vec) / length(pred)
  confM$Rate[nrow(confM)] <- paste0(sum(wrong_vec), " / ", length(pred))

  return(confM)
}


# will export a table or a data frame to tex format
# must be in the format of the confusion matrix above.
exportTableToLatex <- function(table, folder_path, name){
  complete_filepath = paste0(folder_path, name)
  print(xtable(table,
             display = c("d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "f", "s"),
             digits = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0)),
      file = complete_filepath,
      only.contents = T,
      include.rownames = T)
}


# will export a ggplot to tex format
exportPlotToLatex <- function(plot, folder_path, name){
  complete_filepath = paste0(folder_path, name)

  tikz(file = complete_filepath, width = 6, height = 4)
    print(plot)
  dev.off()
}


# ================== #
# PCA HELP FUNCTIONS #
# ================== #

# will return the near zero variance predictors of 'data'
getNZVPredictors <- function(data){
  zeroVarPred <- nearZeroVar(data[,-1], freqCut=10000/1, saveMetrics = TRUE, unique = 1/7)
  return(cutPredictors <- rownames(zeroVarPred[zeroVarPred$nzv == TRUE,]))
}

# will remove cutPredictors from 'data' and normalize it
getScaledAndTrimmedData <- function(data, cutPredictors){

  data <- data[,-which(colnames(data) %in% cutPredictors)]
  data[,-1] <- data[,-1]/255

  return(data)
}

# returns PCA of the data input
applyPCA <- function(scaledTrimmedData){

  # apply PCA
  data.cov <- cov(scaledTrimmedData[,-1])
  pca <- prcomp(data.cov)

  return(pca)
}

# will return the data input represented with principal components
getDataAsPCA <- function(scaledData, pca, numComponents){
  labels = scaledData$Digit
  data <- scaledData[,-1] # remove response variables

  score <- as.matrix(data) %*% pca$rotation[,1:numComponents]
  dataAsPCA <- cbind(Digit = labels, as.data.frame(score))
  return(dataAsPCA)
}
