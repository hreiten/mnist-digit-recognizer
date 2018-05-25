rm(list=ls())
set.seed(1000)
Sys.setenv(TZ="Africa/Johannesburg")

library(ggplot2)
library(caret) # used to make a confusion matrix
library(tikzDevice) # used to compile ggplots in latex
library(xtable) # used to compile tables and dataframes in latex
library(tree)

source("HelpFunctions.R")
exportspath <- "../exports/tree_based_methods/classificationtrees/"

# read in data
data <- read.csv("../data/Train_Digits_20171108.csv")
data$Digit <- as.factor(data$Digit)

# divide into training and testing sets
sample <- sample(1:nrow(data), 0.8*nrow(data))
train <- data[sample,]
test <- data[-sample,]

# save the pruned tree to file
fit <- tree(Digit ~ ., data = train, mindev = 1/200)
png(paste0(exportspath, "classtree_unpruned.png"), width = 1000, height = 600)
  plot(fit); text(fit, cex = 0.6)
dev.off()

cv.fit <- cv.tree(fit)
df <- data.frame(nodes = cv.fit$size, err = cv.fit$dev)

# assign the best node where the tree should be splitted from
best = 22

# plot the deviance per tree
pl <- ggplot(df) +
  geom_line(aes(x = nodes, y = err)) +
  geom_point(aes(x = nodes, y = err)) +
  geom_vline(xintercept = best, linetype = "dotted", col = "red") +
  theme_bw() +
  xlab("Number of terminal nodes") + ylab("Deviance")
exportPlotToLatex(pl, exportspath, "classtree_errorPerTree.tex")


# Will prune the tree down to 'best' trees
cv.fit$k[1] <- 0
alpha <- round(cv.fit$k)
fit.pruned <- prune.tree(fit, best = best)

# save the pruned tree to file
png(paste0(exportspath, "classtree_pruned.png"), width = 1000, height = 600)
  plot(fit.pruned); text(fit.pruned, cex = 0.6)
dev.off()


# evaluate values on test set
pred <- predict(fit.pruned, newdata = test, type="class")
pred_df <- data.frame(pred = pred, true = test$Digit)

confM <- makeConfusionMatrix(pred,test$Digit)
exportTableToLatex(confM, exportspath, "classtree_confusionmatrix.tex")
