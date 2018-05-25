rm(list=ls())
options('width'=200)

source("HelpFunctions.R")
exportspath = "../exports/digits/"
data <- read.csv("../data/Train_Digits_20171108.csv")
images <- data[,-1]

rotate <- function(x) t(apply(x, 2, rev))

png(filename = paste0(exportspath, "30digits.png"), height = 600, width = 1000)
  opar <- par(no.readonly = T)
  par(mfrow=c(8,16),mar=c(.1,.1,.1,.1))
  for (i in 1:(8*16)){
    # make image matrix
    m <- matrix(images[i,], nrow=28, ncol=28)
    m <- apply(m, c(1,2), function(x) as.numeric(x)/255)
    m <- rotate(m)
    image(m, col = grey(seq(0, 1, length = 256)),
          xaxt='n', ann=FALSE, yaxt='n')
  }
  par(opar)
dev.off()



# make histogram of the distribution of digits in the dataset
tab = table(data$Digit)
color = hcl(h = 15, l = 65, c = 100)
pl <- ggplot(data = data) +
  geom_bar(aes(x = as.factor(Digit), fill=Digit), fill = color) +
  theme_bw() + xlab("Digits") + ylab("Count")


exportPlotToLatex(pl, exportspath, "digits_histogram.tex")
