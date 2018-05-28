# mnist-digit-recognizer

The aim of this repository is to compare the performance of several different learning algorithms tasked with classifying a set of handwritten digits and determine if they are even or odd. The training and testing sets used are found in the _data_-folder.

See Kaggle for more information on the problem: http://www.kaggle.com/c/digit-recognizer

The following machine learning methods are implemented and compared: 
* __Tree-based Methods__
    * Classification Trees
    * Random Forest
    * Bagging
    * Boosting
* __Nearest Neighbors__
    * K Nearest Neighbors
* __Support Vector Machines__
* __Neural Networks__
    * Artificial Neural Networks (ANN)
    * Convolutional Neural Networks (CNN)

Each technique is evaluated in the following manner:<br/>
__1. Splitting data into training/testing sets__<br/>
The training set will be divided into a training set (80%) and a test set (20%). The test set will not be used in the training process, and is only there to evaluate the finally chosen hypothesis. If required, 20% of the training data will be extracted to form a validation set.<br/><br/>
__2. Finding the optimal parameters using validation__<br/>
Some form of validation will be used to choose the optimal parameters for each model. Parameters can be regularization factors, learning rates, required complexities etc. The validation will most often be in the form of 10-fold Cross Validation, OOB error, otherwise own validation sets. The model with the optimal parameters is chosen with respect to Eval(h), and the model with the best parameters is denoted g∗ for each method.<br/><br/>
__3. Evaluating the best model on the test set__<br/>
The best hypothesis, g∗, will then be evaluated on the test set and produce Eout(g∗). The value of Eout(g∗) is what will rank the different machine learning methods to each other. The lower Eout(g∗), the better.<br/>
