import Classification
import datetime, time
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sbn

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#filename = 'Chicago_Crime.csv'
filename = '~/Dropbox/APPM/APPM4580/FinalProject/Chicago_Crime.csv'
types = {
    'IUCR': 'category',
    'Primary Type' : 'category',
    'Description': 'category',
    'Location Description': 'category',
    'Block': 'category',
    'Beat': 'category',
    'District': 'category',
    'Community Area': 'category',
    'Domestic': 'category'
}
n = 40000
s = 42

clf = Classification.classification(filename, types, n, s)
data = clf.process_data()
full_set = data
IUCR = data[['IUCR', 'Primary Type', 'Description']].sort_values(['IUCR'])
geog = data[['Block', 'Beat', 'District', 'Community Area', 'Latitude', 'Longitude']]
data = data.drop(['Date', 'Time', 'Primary Type', 'Description', 'Block', 'District', 'Community Area', 'Beat'], axis = 1)

ohe = pd.get_dummies(data)

Xs = list(ohe.columns)[1:]
Y = 'Arrest'
test_prob = 0.25

#logistic regression on full One Hot Model, unscaled
logReg = LogisticRegression(solver = 'lbfgs')
logReg, logReg_scores, logReg_cm = clf.classification(logReg, ohe, Xs, Y, test_prob)
print('\nLogistic Regression: ')
#print(logReg_cm)
print(logReg_scores)
print(' ')

# Logistic Regression with PCA
logReg_pca = LogisticRegression(solver = 'lbfgs')
logReg_pca, logReg_pca_scores, logReg_pca_cm = clf.classification(logReg_pca, ohe, Xs, Y, test_prob, dim_reduction = 'pca', scaled = True)
print('\nPCA + Logistic Regression: ')
#print(logReg_pca_cm)
print(logReg_pca_scores)
print(' ')

# Logistic Regression with LDA
logReg_lda = LogisticRegression(solver = 'lbfgs')
logReg_lda, logReg_lda_scores, logReg_lda_cm = clf.classification(logReg_lda, ohe, Xs, Y, test_prob, dim_reduction = 'lda', scaled = True)
print('\nLDA + Logistic Regression: ')
#print(logReg_lda_cm)
print(logReg_lda_scores)
print(' ')

# SVM with Linear Kernel
svmLin_lda = svm.SVC(kernel = 'linear', gamma = 0.001, C = 5.0)
svmLin_lda, svmLin_lda_scores, svmLin_lda_cm = clf.classification(svmLin_lda, ohe, Xs, Y, test_prob, classifier = 'svmLin', dim_reduction = 'lda', scaled = True)
print('LDA + Linear SVC (C = 5.0, gamma = 0.001): ')
#print(svmLin_lda_cm)
print(svmLin_lda_scores)
print(' ')

svmLin_pca = svm.SVC(kernel = 'linear', gamma = 0.001, C = 5.0)
svmLin_pca, svmLin_pca_scores, svmLin_pca_cm = clf.classification(svmLin_pca, ohe, Xs, Y, test_prob, classifier = 'svmLin', dim_reduction = 'pca', scaled = True)
print('PCA + Linear SVC (C = 5.0, gamma = 0.001): ')
#print(svmLin_pca_cm)
print(svmLin_pca_scores)
print(' ')

# SVM with Radial Kernel
svmRad = svm.SVC(kernel = 'rbf', gamma = 0.001, C = 1.0)
svmRad, svmRad_lda_scores, svmRad_lda_cm = clf.classification(svmRad, ohe, Xs, Y, test_prob, classifier = 'svmRad', dim_reduction = 'lda', scaled = True)
print('\nLDA + Radial SVM: ')
print(svmRad_lda_cm)
print(svmRad_lda_scores)
print(' ')

svmRad1 = svm.SVC(kernel = 'rbf', gamma = 0.001, C = 5.0)
svmRad1, svmRad_lda_scores, svmRad_lda_cm = clf.classification(svmRad1, ohe, Xs, Y, test_prob, classifier = 'svmRad', dim_reduction = 'lda', scaled = True)
print('\nLDA + Radial SVM: ')
print(svmRad_lda_cm)
print(svmRad_lda_scores)
print(' ')

svmRad = svm.SVC(kernel = 'rbf', gamma = 0.1, C = 1.0)
svmRad, svmRad_lda_scores, svmRad_lda_cm = clf.classification(svmRad, ohe, Xs, Y, test_prob, classifier = 'svmRad', dim_reduction = 'lda', scaled = True)
print('\nLDA + Radial SVM: ')
print(svmRad_lda_cm)
print(svmRad_lda_scores)
print(' ')

svmRad1 = svm.SVC(kernel = 'rbf', gamma = 0.1, C = 5.0)
svmRad1, svmRad_lda_scores, svmRad_lda_cm = clf.classification(svmRad1, ohe, Xs, Y, test_prob, classifier = 'svmRad', dim_reduction = 'lda', scaled = True)
print('\nLDA + Radial SVM: ')
print(svmRad_lda_cm)
print(svmRad_lda_scores)
print(' ')

svmRad2 = svm.SVC(kernel = 'rbf', gamma = 0.001, C = 1.0)
svmRad2, svmRad_lda_scores, svmRad_lda_cm = clf.classification(svmRad2, ohe, Xs, Y, test_prob, classifier = 'svmRad', dim_reduction = 'pca', scaled = True)
print('\nPCA + Radial SVM: ')
print(svmRad_lda_cm)
print(svmRad_lda_scores)
print(' ')

svmRad3 = svm.SVC(kernel = 'rbf', gamma = 0.001, C = 5.0)
svmRad3, svmRad_lda_scores, svmRad_lda_cm = clf.classification(svmRad3, ohe, Xs, Y, test_prob, classifier = 'svmRad', dim_reduction = 'pca', scaled = True)
print('\nPCA + Radial SVM: ')
print(svmRad_lda_cm)
print(svmRad_lda_scores)
print(' ')

'''
Resources:
1. Linear regression in python: 
    https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/Logistic_Regression/LogisticRegression_toy_digits_Codementor.ipynb
2. PCA in python:
    https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
3. How to use seaborn:
    https://seaborn.pydata.org/tutorial/categorical.html
4. SVMs in python:
    http://scikit-learn.org/stable/auto_examples/plot_multilabel.html#sphx-glr-auto-examples-plot-multilabel-py
'''
