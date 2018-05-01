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
from sklearn.random_projection import johnson_lindenstrauss_min_dim as jlmd, SparseRandomProjection
from sklearn import svm

filename = 'Chicago_Crime.csv'
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
n = 1000
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
print(logReg_cm)
print(logReg_scores)

#logistic regression on full One Hot Model, scaled
logReg_sc = LogisticRegression(solver = 'lbfgs')
logReg_sc, logReg_sc_scores, logReg_sc_cm = clf.classification(logReg_sc, ohe, Xs, Y, test_prob, scaled = True)
print('\nScaled Logistic Regression: ')
print(logReg_sc_cm)
print(logReg_sc_scores)

# Logistic Regression with PCA
logReg_pca = LogisticRegression(solver = 'lbfgs')
logReg_pca, logReg_pca_scores, logReg_pca_cm = clf.classification(logReg_pca, ohe, Xs, Y, test_prob, dim_reduction = 'pca', scaled = True)
print('\nPCA + Logistic Regression: ')
print(logReg_pca_cm)
print(logReg_pca_scores)

# Logistic Regression with LDA
logReg_lda = LogisticRegression(solver = 'lbfgs')
logReg_lda, logReg_lda_scores, logReg_lda_cm = clf.classification(logReg_lda, ohe, Xs, Y, test_prob, dim_reduction = 'lda', scaled = True)
print('\nLDA + Logistic Regression: ')
print(logReg_lda_cm)
print(logReg_lda_scores)

# LDA
lda = LinearDiscriminantAnalysis()
lda, lda_scores, lda_cm = clf.classification(lda, ohe, Xs, Y, test_prob, classifier = 'lda', scaled = True)
print('\nLDA: ')
print(lda_cm)
print(lda_scores)

# QDA 
qda = QuadraticDiscriminantAnalysis()
qda, qda_scores, qda_cm = clf.classification(qda, ohe, Xs, Y, test_prob, classifier = 'qda', scaled = True)
print('\nQDA: ')
print(qda_cm)
print(qda_scores)

# KNN
knn_pca = KNeighborsClassifier(n_neighbors = 5)
knn_pca, knn_pca_scores, knn_pca_cm = clf.classification(knn_pca, ohe, Xs, Y, test_prob, classifier = 'knn', dim_reduction = 'pca', scaled = True)
print('\nPCA + KNN: ')
print(knn_pca_cm)
print(knn_pca_scores)

knn_lda = KNeighborsClassifier(n_neighbors = 5)
knn_lda, knn_lda_scores, knn_lda_cm = clf.classification(knn_lda, ohe, Xs, Y, test_prob, classifier = 'knn', dim_reduction = 'lda', scaled = True)
print('\nLDA + KNN: ')
print(knn_lda_cm)
print(knn_lda_scores)

# SVM with Linear Kernel + LDA
svmLin_lda = svm.SVC(kernel = 'linear', gamma = 0.001, C = 1.0)
svmLin_lda, svmLin_lda_scores, svmLin_lda_cm = clf.classification(svmLin_lda, ohe, Xs, Y, test_prob, classifier = 'svmLin', dim_reduction = 'lda', scaled = True)
print('\nLDA + Linear SVC: ')
print(svmLin_lda_cm)
print(svmLin_lda_scores)

# SVM with Radial Kernel
svmRad = svm.SVC(kernel = 'rbf', gamma = 0.001, C = 1.0)
svmRad, svmRad_lda_scores, svmRad_lda_cm = clf.classification(svmRad, ohe, Xs, Y, test_prob, classifier = 'svmRad', dim_reduction = 'lda', scaled = True)
print('\nLDA + Radial SVM: ')
print(svmRad_lda_cm)
print(svmRad_lda_scores)

#huge_clf = Classification.classification(filename, types, -1, s)
#huge_set = huge_clf.process_data()
#huge_set.drop(['Date', 'Time', 'Primary Type', 'Description', 'Beat', 'Block', 'District', 'Community Area'], axis = 1)
#huge_ohe = pd.get_dummies(huge_set)

#Xs = list(huge_ohe.columns)[1:]
#srp = SparseRandomProjection()
#srp, sparse_proj_X = huge_clf.sparse_random_projection(srp, huge_ohe[Xs], huge_ohe[Y])
#print(srp.components_)



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
