import re, numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

# Return the accuracy, recall, precision, and specificity
def ConfusionMatrixCalculations(CM):
    return [(CM[0,0]+CM[1,1])/sum(sum(CM)), CM[0,0]/(CM[0,0]+CM[1,0]), CM[0,0]/(CM[0,0]+CM[0,1]), CM[1,1]/(CM[1,0]+CM[1,1])]

def GenerateGraphs(TestY, PredictY, Title):
    fpr, tpr, _ = roc_curve(TestY, PredictY)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(Title)
    plt.legend()
    plt.show()

def PerformExperiment(TrainX, TestX, ValidationX, TrainY, TestY, ValidationY, Algorithm, Title):

    # Train the model and predict the necessary values
    ExperimentEfficiency = []
    StartTime=time()
    Algorithm.fit(TrainX, TrainY)
    ExperimentEfficiency.append(round(time()-StartTime, 3))
    StartTime=time()
    PredictTestY = Algorithm.predict(TestX)
    ExperimentEfficiency.append(round(time()-StartTime, 3))
    PredictTrainingY = Algorithm.predict(TrainX)
    PredictValidationY = Algorithm.predict(ValidationX)

    # Generate the stats and the graphs
    CMTraining = confusion_matrix(TrainY,PredictTrainingY)
    CMValidation = confusion_matrix(ValidationY,PredictValidationY)
    CMTest = confusion_matrix(TestY,PredictTestY)
    GenerateGraphs(TrainY, PredictTrainingY, Title + " (Training Set)")
    GenerateGraphs(ValidationY, PredictValidationY, Title + " (Validation Set)")
    GenerateGraphs(TestY, PredictTestY, Title + " (Testing Set)")
    return [[ConfusionMatrixCalculations(CMTraining), roc_auc_score(TrainY, PredictTrainingY)], [ConfusionMatrixCalculations(CMValidation), roc_auc_score(ValidationY, PredictValidationY)], [ConfusionMatrixCalculations(CMTest), roc_auc_score(TestY, PredictTestY)], ExperimentEfficiency]

def ConductExperiments(Features, Labels, Identifier):
    
    # Dataset Split 60% Training, 20% Test, 20% Validation
    TrainX, TestX, TrainY, TestY = train_test_split(Features, Labels, test_size=0.2)
    TrainX, ValidationX, TrainY, ValidationY = train_test_split(TrainX, TrainY, test_size=0.25)
    N = FindBestKNN(TrainX,TrainY,ValidationX,ValidationY)
    Gamma = FindBestSVC(TrainX,TrainY,ValidationX,ValidationY)
    return [PerformExperiment(TrainX, TestX, ValidationX, TrainY, TestY, ValidationY, GaussianNB(), "Gaussian Naive Bayes " + Identifier), PerformExperiment(TrainX, TestX, ValidationX, TrainY, TestY, ValidationY, KNeighborsClassifier(n_neighbors=N), "KNN-" + str(N) + " " + Identifier), PerformExperiment(TrainX, TestX, ValidationX, TrainY, TestY, ValidationY, SVC(gamma=Gamma, probability=True), "SVC-" + Gamma + " " + Identifier)]

def FindBestKNN(TrainX,TrainY,ValidationX,ValidationY):
    MaxAccuracy = 0
    Result = 0
    print("Performance training with KNN is taking place...")
    N = list(range(10)[1::2])
    for i in N:
        PredictY=KNeighborsClassifier(n_neighbors=i).fit(TrainX, TrainY).predict(ValidationX)
        Accuracy = accuracy_score(ValidationY, PredictY)
        if  Accuracy > MaxAccuracy:
            MaxAccuracy =Accuracy
            Result = i
    return Result

def FindBestSVC(TrainX,TrainY,ValidationX,ValidationY):
    MaxAccuracy = 0
    Result = 0
    print("Performance training with SVC is taking place...")
    for i in  ['auto', 'scale']:
        PredictY=SVC(gamma=i, probability=True).fit(TrainX, TrainY).predict(ValidationX)
        Accuracy = accuracy_score(ValidationY, PredictY)
        if  Accuracy > MaxAccuracy:
            MaxAccuracy =Accuracy
            Result = i
    return Result

# Print out each stat to the command prompt
def ReportStatistics(Title, Data, Time):
    print(Title)
    print("Accuracy: %.2f" % Data[0][0])
    print("Precision: %.2f" % Data[0][1])
    print("Recall: %.2f" % Data[0][2])
    print("Specificity: %.2f" % Data[0][3])
    print("AUROC: %.2f" % Data[1])
    print("Offline Efficiency Cost: %.2f seconds" %  Time[0])
    print("Online Efficiency Cost: %.2f seconds" %  Time[1] + "\n\n")
    
# Load the data to Python and combine them
Amazon = list(filter(None, re.split("[\t\n]",open("data/amazon_cells_labelled.txt", "r").read())))
IMDB = list(filter(None, re.split("[\t\n]",open("data/imdb_labelled.txt", "r").read())))
Yelp = list(filter(None, re.split("[\t\n]",open("data/yelp_labelled.txt", "r").read())))
Amazon.extend(IMDB)
Amazon.extend(Yelp)

# Feature Construction and Feature Selection with top-1000 and stop words
Labels = list(map(int, Amazon[1::2]))
FV1 = CountVectorizer().fit_transform(Amazon[0::2])
FV2 = CountVectorizer(max_features = 1000, stop_words = list(ENGLISH_STOP_WORDS)).fit_transform(Amazon[0::2])

# Perform the experiments
FV1ExperimentResults = ConductExperiments(FV1.toarray(), Labels, "without feature selection")
FV2ExperimentResults = ConductExperiments(FV2.toarray(), Labels, "with stop words and top-1000")

# Report the outcome of the experiments
ReportStatistics("Gaussian Naive Bayes without feature selection (Training Set):", FV1ExperimentResults[0][0], FV1ExperimentResults[0][3])
ReportStatistics("Gaussian Naive Bayes without feature selection (Validation Set):", FV1ExperimentResults[0][1], FV1ExperimentResults[0][3])
ReportStatistics("Gaussian Naive Bayes without feature selection (Testing Set):", FV1ExperimentResults[0][2], FV1ExperimentResults[0][3])
ReportStatistics("KNN without feature selection (Training Set):", FV1ExperimentResults[1][0], FV1ExperimentResults[1][3])
ReportStatistics("KNN without feature selection (Validation Set):", FV1ExperimentResults[1][1], FV1ExperimentResults[1][3])
ReportStatistics("KNN without feature selection (Testing Set):", FV1ExperimentResults[1][2], FV1ExperimentResults[1][3])
ReportStatistics("SVC without feature selection (Training Set):", FV1ExperimentResults[2][0], FV1ExperimentResults[2][3])
ReportStatistics("SVC without feature selection (Validation Set):", FV1ExperimentResults[2][1], FV1ExperimentResults[2][3])
ReportStatistics("SVC without feature selection (Testing Set):", FV1ExperimentResults[2][2], FV1ExperimentResults[2][3])
ReportStatistics("Gaussian Naive Bayes with stop words and top-1000 (Training Set):", FV2ExperimentResults[0][0], FV2ExperimentResults[0][3])
ReportStatistics("Gaussian Naive Bayes with stop words and top-1000 (Validation Set):", FV2ExperimentResults[0][1], FV2ExperimentResults[0][3])
ReportStatistics("Gaussian Naive Bayes with stop words and top-1000 (Testing Set):", FV2ExperimentResults[0][2], FV2ExperimentResults[0][3])
ReportStatistics("KNN with stop words and top-1000 (Training Set):", FV2ExperimentResults[1][0], FV2ExperimentResults[1][3])
ReportStatistics("KNN with stop words and top-1000 (Validation Set):", FV2ExperimentResults[1][1], FV2ExperimentResults[1][3])
ReportStatistics("KNN with stop words and top-1000 (Testing Set):", FV2ExperimentResults[1][2], FV2ExperimentResults[1][3])
ReportStatistics("SVC with stop words and top-1000 (Training Set):", FV2ExperimentResults[2][0], FV2ExperimentResults[2][3])
ReportStatistics("SVC with stop words and top-1000 (Validation Set):", FV2ExperimentResults[2][1], FV2ExperimentResults[2][3])
ReportStatistics("SVC with stop words and top-1000 (Testing Set):", FV2ExperimentResults[2][2], FV2ExperimentResults[2][3])
