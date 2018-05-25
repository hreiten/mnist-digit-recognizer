rm(list=ls(all = T))
set.seed(1000)
Sys.setenv(TZ="Africa/Johannesburg")

library(ggplot2)
library(randomForest)
library(tikzDevice)
library(caret)

source("HelpFunctions.R")
exportspath <- "../exports/tree_based_methods/randomforest/"

# read in csv error files
rf <- read.csv("../exports/tree_based_methods/randomforest/rf_oob_error.csv")
bag <- read.csv("../exports/tree_based_methods/bagging/bag_oob_error.csv")

if(nrow(rf) != nrow(bag)){
  stop("ERROR: Different sizes in bagging and randomforest")
}

df <- data.frame(trees = rf$trees, rf_err = rf$err, bag_err = bag$err)

pl <- ggplot(df) +
  geom_line(aes(x = trees, y = rf_err, col = "rf")) +
  geom_line(aes(x = trees, y = bag_err, col = "bag")) +
  scale_color_discrete(name="", labels =c("Bagging", "Random Forest")) +
  theme_bw() + theme(legend.position = c(0.85, 0.9),
                     legend.background = element_rect(fill=alpha('white', 0))) +
  xlab("Number of trees") + ylab("OOB error")
exportPlotToLatex(pl, exportspath, "RFvsBagging_OOB.tex")
