# ClassifyReviewSentiments
This Python program classifies positive or negative sentiment reviews from IMDb, Yelp, and Amazon.

## Input

```data/amazon_cells_labelled.txt```
```data/imdb_labelled.txt```
```data/yelp_labelled.txt```

## Output

The x-axis is the False Positive Rate, and the y-axis is the True Positive Rate in all the graphs. The following graphs are outputted to your screen:

Gaussian Naive Bayes without feature selection (Training Set) ROC
Gaussian Naive Bayes without feature selection (Validation Set) ROC
Gaussian Naive Bayes without feature selection (Testing Set) ROC
KNN-5 without feature selection (Training Set) ROC
KNN-5 without feature selection (Validation Set) ROC
KNN-5 without feature selection (Testing Set) ROC
SVC-scale without feature selection (Training Set) ROC
SVC-scale without feature selection (Validation Set) ROC
SVC-scale without feature selection (Testing Set) ROC
Gaussian Naive Bayes with stop words and top-1000 (Training Set) ROC
Gaussian Naive Bayes with stop words and top-1000 (Validation Set) ROC
Gaussian Naive Bayes with stop words and top-1000 (Testing Set) ROC
KNN-7 with stop words and top-1000 (Training Set) ROC
KNN-7 with stop words and top-1000 (Validation Set) ROC
KNN-7 with stop words and top-1000 (Testing Set) ROC
SVC-scale with stop words and top-1000 (Training Set) ROC
SVC-scale with stop words and top-1000 (Validation Set) ROC
SVC-scale with stop words and top-1000 (Testing Set) ROC

The following values are also printed out in your Command Prompt after the graphs:

Accuracy: the number of correct predictions divided by the total number of predictions.
Precision: the true positive divided by the true positive plus the false positive
Recall: the true positive divided by true positive plus false negative
Specificity: calculated as the number of correct negative predictions divided by the total number of negatives
AUROC is the area under the ROC curve and an estimator of the model's performance.
Offline efficiency: the time needed to train the model in seconds
Online efficiency: the time needed to predict the test set in seconds.

## Running The Project In Windows

In the project's root directory, type ```pip install -r requirements.txt``` to your Command Prompt to install all dependencies.

Type ```python main.py``` to run the project.

## How Does The Program Run

The review sentiments are gathered from the GitHub repository https://github.com/microsoft/ML-Server-Python-Samples/tree/master/microsoftml/202/data/sentiment_analysis. The data is first loaded into a Python module and divided into 60% training, 20% test, and 20% validation sets. KNN, GaussianNB, and SVC are used as predictors. 

First, two feature vectors are created, one with feature selection for stop words from sklearn's list and the other without the stop words. Some stop words are pronouns such as "he," "she," and "is," and they do not contribute to the accuracy. The experiment starts with the feature vector without feature selection.

Parameters, such as the number of neighbors in KNN and whether the scale or auto should be used in SVC, are determined with the validation set, which chooses the parameter that yields the highest accuracy. Afterwards, the models are trained with the training set, and the stats are measured using the test set. The offline and online efficiency is calculated at this stage. The confusion matrix is created, and the stats are calculated. The roc curve is plotted first and displayed on the screen for the six experiments (3 experiments for each feature vector). All stats are printed out afterwards.