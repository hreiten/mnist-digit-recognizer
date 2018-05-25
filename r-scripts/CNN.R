rm(list=ls())
set.seed(1000)
Sys.setenv(TZ="Africa/Johannesburg")

library(mxnet)
library(ggplot2)
library(xtable)
library(tikzDevice)
library(caret)

source("HelpFunctions.R")
exportspath <- "../exports/NN/"

# read in data
alldata <- read.csv("../data/Train_Digits_20171108.csv")
alldata$Digit <- as.factor(alldata$Digit)

sample <- sort(sample(1:nrow(alldata), 0.8*nrow(alldata)))
train <- alldata[sample[1:(0.8*length(sample))],]
val <- alldata[sample[(nrow(train)+1):length(sample)],]
test <- alldata[-sample,]

train.m <- data.matrix(train)
val.m <- data.matrix(val)
test.m <- data.matrix(test)

# define predictors and response for training, validation and testing
train.x <- t(train.m[,-1])
train.y <- train.m[,1] - 1
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))

val.x <- t(val.m[,-1])
val.y <- val.m[,1] - 1
val.array <- val.x
dim(val.array) <- c(28, 28, 1, ncol(val.x))


test.x <- t(test.m[,-1])
test.y <- test.m[,1]
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))






# ========================== #
# Setup the CNN architecture #
# ========================== #
data <- mx.symbol.Variable('data')

# first convolution
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# second convolution
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# first fully-connected layer
# use data from convolutions (flatten)
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")


# second fully-connected layer
fc2 <- mx.symbol.FullyConnected(data = tanh3, num_hidden=10)

# the output will be a 10-way softmax.
softmax <- mx.symbol.SoftmaxOutput(data = fc2)

# use CPU device
devices <- mx.cpu()
mx.set.seed(1000)

# define a logger to keep track of errors
logger <- mx.metric.logger$new()




# =============================== #
# Hyperparameter search in a grid #
# =============================== #

# perform a grid search for the best parameters
grid <- data.frame(id = 1:6, rate = c(0.01,0.01,0.001,0.001,0.005,0.005), array_size = c(50,100), e_val = NA)

for (i in 1:nrow(grid)){
  rate = grid$rate[i]
  size = grid$array_size[i]

  cnn.model <- mx.model.FeedForward.create(softmax,
                                           X = train.array, y = train.y,
                                           eval.data = list(data=val.array, label=val.y),
                                           ctx=devices,
                                           num.round=40,
                                           array.batch.size=size,
                                           learning.rate=rate,
                                           momentum=0.9,
                                           eval.metric=mx.metric.accuracy,
                                           optimizer = "sgd",
                                           epoch.end.callback=mx.callback.log.train.metric(5,logger))

  grid$e_val[i] = 1-logger$eval[length(logger$eval)]

}

# export grid to latex
print(xtable(grid,
             display = c("d", "d", "f", "d", "f"),
             digits = c(0, 0, 3, 0, 4)),
      file = paste0(exportspath, "cnn_gridsearch.tex"),
      only.contents = T,
      include.rownames = F)

# plot the grid
pl<-ggplot(grid) +
  geom_point(aes(x = id, y = e_val)) +
  theme_bw() + xlab("Model") + ylab("$E_{val}$")
exportPlotToLatex(pl, exportspath, "cnn_griderrors.tex")






# ======================================== #
# Train a new CNN with the best parameters #
# ======================================== #

# just a run-through to plot the validation error as a function of epochs
sortedGrid <- grid[order(grid$e_val),]
bestParameters <- sortedGrid[1,c(2,3)]
bestParameters <- data.frame(rate = 0.01, array_size = 100)

logger <- mx.metric.logger$new()
best.model.val <- mx.model.FeedForward.create(softmax,
                                         X = train.array, y = train.y,
                                         eval.data = list(data=val.array, label=val.y),
                                         ctx=devices,
                                         num.round=40,
                                         array.batch.size=bestParameters$array_size,
                                         learning.rate=bestParameters$rate,
                                         momentum=0.9,
                                         optimizer = "sgd",
                                         eval.metric=mx.metric.accuracy,
                                         epoch.end.callback=mx.callback.log.train.metric(5,logger))


err_df <- data.frame(runs = 1:40, e_val = 1 - logger$eval)
pl <- ggplot(err_df) +
  geom_line(aes(x = runs, y = e_val)) +
  theme_bw() + xlab("Epochs") + ylab("$E_{val}$")
exportPlotToLatex(pl, exportspath, "cnn_error.tex")


# train A NEW model with THE WHOLE training data, no validation set
train <- alldata[sample,]
train.m <- data.matrix(train)
train.x <- t(train.m[,-1])
train.y <- train.m[,1] - 1
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))

best.model <- mx.model.FeedForward.create(softmax,
                                         X = train.array, y = train.y,
                                         ctx=devices,
                                         num.round=40,
                                         array.batch.size=bestParameters$array_size,
                                         learning.rate=bestParameters$rate,
                                         momentum=0.9,
                                         optimizer = "sgd",
                                         eval.metric=mx.metric.accuracy,
                                         epoch.end.callback=mx.callback.log.train.metric(100))


# predict on the test set
pred <- predict(best.model, test.array)
pred_digits <- max.col(t(pred)) - 1

confM <- makeConfusionMatrix(pred_digits,test$Digit)
print(confM)
exportTableToLatex(confM, exportspath, "CNN_confusionmatrix.tex")







# ================================= #
# MAKE PREDICITONS ON REAL TEST SET #
# ================================= #
# As this method was the one who produced the best results of all ML techniques, use this
# to predict on the final test set

# read in data again
train <- read.csv("../data/Train_Digits_20171108.csv")
test <- read.csv("../data/Test_Digits_20171108.csv")
train$Digit = as.factor(train$Digit)
test$Digit = as.factor(test$Digit)

train.m <- data.matrix(train)
test.m <- data.matrix(test)

# define predictors and response
train.x <- t(train.m[,-1])
train.y <- train.m[,1] - 1
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))

test.x <- t(test.m[,-1])
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))

# train model on whole testing data
# use the parameters found in validation
final.model <- mx.model.FeedForward.create(softmax,
                                          X = train.array, y = train.y,
                                          ctx=devices,
                                          num.round=40,
                                          array.batch.size=bestParameters$array_size,
                                          learning.rate=bestParameters$rate,
                                          momentum=0.9,
                                          optimizer = "sgd",
                                          eval.metric=mx.metric.accuracy,
                                          epoch.end.callback=mx.callback.log.train.metric(100))


# predict on the test set
pred <- predict(final.model, test.array)
pred_digits <- max.col(t(pred)) - 1
pred_isUneven <- pred_digits %% 2

pred_df <- data.frame(row_id = 1:2500, pred_digit = pred_digits, pred_isUneven = pred_isUneven)
write.table(pred_df, file = "../exports/finalpredictions.csv", row.names=F, col.names=T, sep = ",", append=F)
