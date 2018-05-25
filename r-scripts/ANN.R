rm(list=ls())
set.seed(1000)
Sys.setenv(TZ="Africa/Johannesburg")

library(ggplot2)
library(h2o)
library(tikzDevice)
library(caret)
library(xtable)

source("HelpFunctions.R")
exportspath = "../exports/NN/"

# read in data
data <- read.csv("../data/Train_Digits_20171108.csv")
data$Digit <- as.factor(data$Digit)

sample <- sample(1:nrow(data), 0.8*nrow(data))
train <- data[sample,]
test <- data[-sample,]


# initialize h2o
localH2O = h2o.init(nthreads = -1, max_mem_size = '4G')

# add data to h2o environment
train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)

# define predictors and response variables
response = colnames(train)[1]
predictors = colnames(train)[2:ncol(train)]


# using a grid search to find the best combination of parameter tunings
epochs_opt = c(100)
hidden_opt <- list(c(100), c(200), c(400), c(400, 200), c(500, 100), c(300,300))
rate_opt <- c(0.1, 0.01, 0.001)
stopping_tolerance_opt <- c(0.0001)
hyper_params = list(epochs = epochs_opt,
                    hidden = hidden_opt,
                    rate = rate_opt,
                    stopping_tolerance = stopping_tolerance_opt)

nn.grid <- h2o.grid("deeplearning",
                    x = predictors, y = response,
                    training_frame = train.h2o,
                    nfolds = 5,
                    seed = 1000,
                    input_dropout_ratio = 0.20,
                    activation = "Tanh",
                    stopping_metric = "misclassification",
                    stopping_rounds = 2,
                    l1 = 1e-5,
                    hyper_params = hyper_params)

# sort grid by increasing CV error and extract the best model
# nn.grid.ordered <- h2o.getGrid(nn.grid@grid_id, sort_by = "err", decreasing=FALSE);
nn.grid.ordered <- h2o.getGrid(nn.grid@grid_id, sort_by = "mse", decreasing = FALSE);
nn.grid.ordered

# plot the grid sorted by mse
pl <- ggplot(nn.grid.ordered@summary_table, aes(x=1:length(nn.grid.ordered@model_ids), y=as.numeric(err))) +
  geom_point() +
  geom_point(colour = "red", aes(x = 1, y = as.numeric(err[1]))) + # colour the best model in red
  xlab("Nth model with lowest CV Error") + ylab("CV error") + ggtitle("") +
  theme_bw()
exportPlotToLatex(pl, exportspath, "ANN_cverr_gridmodel.tex")


# extract the best model
best_model <- h2o.getModel(nn.grid.ordered@model_ids[[1]])
summary(best_model)

# metrics of best model on TRAINING set
h2o.mse(best_model, xval=TRUE) # mean CV error
best_model@model$cross_validation_metrics_summary # All CV error data

# make predictions and write confusion matrix to file
pred <- h2o.predict(best_model, newdata = test.h2o)$predict
pred <- as.numeric(as.matrix(pred))

confM <- makeConfusionMatrix(pred, test$Digit)
exportTableToLatex(confM, exportspath, "ANN_confusionmatrix.tex")


# write parameters to tex-file
grid_df <- as.data.frame(nn.grid.ordered@summary_table)[,-5]
grid_df$epochs <- as.numeric(grid_df$epochs)
grid_df$mse <- as.numeric(grid_df$mse)

# write grid to tex-file
print(xtable(grid_df,
           display = c("d","f", "s", "f", "e", "f"),
           digits = c(0,2,0,2,1,5)),
           align = c(rep("c",ncol(grid_df)+1)),
    file = paste0(exportspath, "ANN_gridresults.tex"),
    only.contents = T,
    include.rownames = T)


h2o.shutdown(prompt = F)
