import datetime
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import random
import seaborn as sbn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import time

date_parse = lambda t: datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
sbn.set(style="whitegrid", color_codes=True)
np.set_printoptions(threshold=np.nan)

class classification:
    def __init__(self, filename, types, n, s = None):
        self.filename = filename
        self.types = types
        self.n = n
        self.s = s

    def process_data(self):
        self.s = int(random.uniform(0, 10000)) if self.s is None else self.s
        self.n = int(1e5) if self.n < 0 else self.n
        print('Processing Data...\nRandom State: {0}\nSample Size: {1}'.format(self.s, self.n))
        data = pd.read_csv(
            self.filename,
            parse_dates = ['Date'],
            date_parser = date_parse,
            dtype = self.types
        )
        data = data.sample(n = self.n, random_state = self.s)
        data['Arrest'] = data.apply(lambda t: 1 if t['Arrest'] else 0, axis = 1)
        data['Time'] = data.apply(lambda t: t['Date'].replace(second = 0).time(), axis = 1)
        data['Hour'] = data.apply(lambda t: t['Date'].strftime('%H'), axis = 1).astype('category')
        data['Month'] = data.apply(lambda t: t['Date'].strftime('%m'), axis = 1).astype('category')
        data['Day'] = data.apply(lambda t: t['Date'].strftime('%d'), axis = 1).astype('category')
        data['Year'] = data['Year'].astype('category')
        data['IUCR'] = data['IUCR'].cat.remove_unused_categories()
        data['Location Description'] = data['Location Description'].cat.remove_unused_categories()
    
        data = data[['Date', 'Month', 'Day', 'Year', 'Hour', 'Time', 'Arrest', 'IUCR', 'Primary Type', 'Description', 'Domestic', 'Location Description', 'Block', 'Beat', 'District', 'Community Area', 'Latitude', 'Longitude']].sort_values(['Month', 'Day', 'Time', 'Year'])
        print('Done Processing')
        return data
    
    def confusion_mtx(self, conf_mtx, title):
        plt.figure(figsize=(9,9))
        sbn.heatmap(conf_mtx, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
    
        plt.title(title, size = 15);
        #plt.show();

    def classification(self, clf,  data, Xs, Y, test_prob, var_retained = 0.9, classifier = 'logReg', dim_reduction = 'None', scaled = True):
        x_trn, x_tst, y_trn, y_tst = train_test_split(data[Xs], data['Arrest'], test_size = test_prob, random_state = 16)
        print('\nTrue Number of Arrests: {0}'.format(sum(y_tst)))
        print('Number of Components: {0}'.format(len(Xs)))
        print('Classifier: ' + classifier)
        print('Dimensional Reduction Technique: ' + dim_reduction)
        title = ''
        if (scaled):
            scaler = StandardScaler()
            scaler.fit(x_trn)
    
            x_trn = scaler.transform(x_trn)
            x_tst = scaler.transform(x_tst)
    
        if (dim_reduction == 'pca'):
            pca = PCA(var_retained)
            pca.fit(x_trn)
    
            x_trn = pca.transform(x_trn)
            x_tst = pca.transform(x_tst)
    
            title += 'PCA + '
            print('Number of Components (PCA): {0}'.format(pca.n_components_))

        elif (dim_reduction == 'lda'):
            lda = LinearDiscriminantAnalysis(n_components = 2)
            x_trn = lda.fit_transform(x_trn, y_trn)
            x_tst = lda.transform(x_tst)
    
            title += 'LDA + '
    
            print('Number of Significant Components (LDA): {0}'.format(len(lda.coef_[np.where(lda.coef_ > 1e-6)])))
        else:
            title = 'Scaled ' if ((classifier == 'logReg') and (scaled == True)) else title 
    
        if (classifier == 'logReg'):
            title += 'Logistic Regression Accuracy Score: '
    
        elif (classifier == 'lda'):
            title += 'LDA Accuracy Score: '
    
        elif (classifier == 'qda'):
            title += 'QDA Accuracy Score: '
    
        elif (classifier == 'knn'):
            title += 'KNN Accuracy Score: '
    
        elif (classifier == 'svmLin'):
            title += 'SVC (Linear) Accuracy Score: '
        
        elif(classifier == 'svmRad'):
            title += 'SVC (Radial) Accuracy Score: '

        clf.fit(x_trn, y_trn)
        y_pred = clf.predict(x_tst)
    
        accuracy = clf.score(x_tst, y_tst)
        title += '{0:.5g}'.format(accuracy)
    
        conf_mtx = metrics.confusion_matrix(y_tst, y_pred)
        tn, fp, fn, tp = conf_mtx.ravel()
        scores = pd.DataFrame(data = {'Accuracy': (tn + tp)/(tn + fp + fn + tp), 'Sensitivity': tp/(tp + fn), 'Specificity': tn/(fp + tn), 'FPR': 1- tn/(fp + tn)},  index=[0])
        #self.confusion_mtx(conf_mtx, title)
        conf_mtx = pd.DataFrame(data = {'Actual\Predicted': [0, 1], '0': conf_mtx[:,0], '1': conf_mtx[:,1]})
        
        return clf, scores, conf_mtx[['Actual\Predicted', '0', '1']]
    
