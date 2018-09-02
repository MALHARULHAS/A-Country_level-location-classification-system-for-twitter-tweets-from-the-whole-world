# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:55:20 2018

@author: MALHAR ULHAS AGALE
"""
#THANKS TO NPTEL:https://www.youtube.com/watch?v=w781X47Esj8
import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from IPython import get_ipython
import seaborn as sn
from pandas_ml import ConfusionMatrix
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('ggplot')
import os
#===============================================================================================================================================    


# Load CSV using Pandas
cwd_path = os.getcwd()
flme_path = cwd_path + "\DATASETS\TRAINING_SETS"
filename = flme_path + '\TRAINING_SET_TOTAL_TWEETS_Final_mod(WITH_FRIENDS_LOCATION).csv'
print(filename)
names = ['place', 'coordinates', 'Timezone', 'LOCFLD_Loc', 'LOCFLD_Loc', 'USERNAME', 'Textcont', 'Descripcont', 'Intrfc_Lang','Twt_Lang','User_Friends_location','country_class']
df = pd.read_csv(filename, names=names)
print(df.shape)

X = np.array(df.drop(['country_class'],1))
y = np.array(df['country_class'])

print(X)
print(y)
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y, test_size = 0.10)


#================================================================================================================
#TO USE THE AVERAGES JUST REPLACE "weighted" with "micro", "macro","none" and get the results.


'''
#=================================================================================================================

#USING SVM
##kernels = ('linear', 'poly', 'rbf')
kernels = ('poly')
for index,kernel in enumerate(kernels):    
    print('USING THE',kernel,' KERNEL')
    model = svm.SVC(kernel = kernel)
    model.fit(X_train, y_train)
    y_pred_svm = model.predict(X_test)
    
    
    #print('The mean accuracy on the given test data and labels =>',model.score(X_test, y_test))
    print('ACCURACY -> ',accuracy_score(y_test, y_pred_svm))    
    print('PRECISION -> ',precision_score(y_test, y_pred_svm, average= 'micro'))
    print('RECALL -> ',recall_score(y_test, y_pred_svm, average= 'micro')) 
    print('F-1 SCORE -> ',f1_score(y_test, y_pred_svm, average= 'micro'))

    print (classification_report(y_test, y_pred_svm))
    #cm1 = confusion_matrix(y_test, y_pred_svm)
    cm_svm = ConfusionMatrix(y_test, y_pred_svm)
    print(cm_svm)
    cm_svm.print_stats()
    cm_svm.plot()
    #cm1.plot()
    #sn.heatmap(cm1, annot=True)
    
#================================================================================================================  
'''


#=================================================================================================================

#USING SVM
##kernels = ('linear', 'poly', 'rbf')

print('USING THE','polynomial')



model = svm.SVC(kernel = 'poly')
#model = svm.SVC(kernel = 'linear')
#model = svm.SVC(kernel = 'rbf')
model.fit(X_train, y_train)
y_pred_svm = model.predict(X_test)
    
    
#print('The mean accuracy on the given test data and labels =>',model.score(X_test, y_test))
print('ACCURACY -> ',accuracy_score(y_test, y_pred_svm))    
print('PRECISION -> ',precision_score(y_test, y_pred_svm, average= 'weighted'))
#print('PRECISION -> ',precision_score(y_test, y_pred_svm, average= 'micro'))
#print('PRECISION -> ',precision_score(y_test, y_pred_svm, average= 'macro'))
print('RECALL -> ',recall_score(y_test, y_pred_svm, average= 'weighted')) 
#print('RECALL -> ',recall_score(y_test, y_pred_svm, average= 'micro')) 
#print('RECALL -> ',recall_score(y_test, y_pred_svm, average= 'macro')) 
print('F-1 SCORE -> ',f1_score(y_test, y_pred_svm, average= 'weighted'))
#print('F-1 SCORE -> ',f1_score(y_test, y_pred_svm, average= 'micro'))
#print('F-1 SCORE -> ',f1_score(y_test, y_pred_svm, average= 'macro'))

print (classification_report(y_test, y_pred_svm))
#cm1 = confusion_matrix(y_test, y_pred_svm)
print('ACCURACY -> ',accuracy_score(y_test, y_pred_svm))    

cm_svm = ConfusionMatrix(y_test, y_pred_svm)
print(cm_svm)
cm_svm.print_stats()
cm_svm.plot()

#cm1.plot()
#sn.heatmap(cm1, annot=True)
#================================================================================================================   

'''
#================================================================================================================    
#USING LOGISTIC REGRESSION
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)    

y_pred_maxEnt = LogReg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred_maxEnt)
print('ACCURACY -> ',accuracy_score(y_test, y_pred_maxEnt))    
print('PRECISION -> ',precision_score(y_test, y_pred_maxEnt, average= 'weighted'))
print('RECALL -> ',recall_score(y_test, y_pred_maxEnt, average= 'weighted')) 
print('F-1 SCORE -> ',f1_score(y_test, y_pred_maxEnt, average= 'weighted'))
print(classification_report(y_test, y_pred_maxEnt))
cm_LR = ConfusionMatrix(y_test, y_pred_maxEnt)
print(cm_LR)
cm_LR.print_stats()
cm_LR.plot()

#================================================================================================================    
'''

'''
#================================================================================================================    
#USING KNeighborsClassifier
KReg = KNeighborsClassifier()
KReg.fit(X_train, y_train)    

y_pred_Knei = KReg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred_Knei)
print('ACCURACY -> ',accuracy_score(y_test, y_pred_Knei))    
print('PRECISION -> ',precision_score(y_test, y_pred_Knei, average= 'weighted'))
print('RECALL -> ',recall_score(y_test, y_pred_Knei, average= 'weighted')) 
print('F-1 SCORE -> ',f1_score(y_test, y_pred_Knei, average= 'weighted'))
print(classification_report(y_test, y_pred_Knei))

cm_Knc = ConfusionMatrix(y_test, y_pred_Knei)
print(cm_Knc)
cm_Knc.print_stats()
cm_Knc.plot()

#================================================================================================================  
'''

'''
#================================================================================================================    
#USING RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)    

y_pred_RF = RF.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred_RF)
print('ACCURACY -> ',accuracy_score(y_test, y_pred_RF))    
print('PRECISION -> ',precision_score(y_test, y_pred_RF, average= 'weighted'))
print('RECALL -> ',recall_score(y_test, y_pred_RF, average= 'weighted')) 
print('F-1 SCORE -> ',f1_score(y_test, y_pred_RF, average= 'weighted'))
print(classification_report(y_test, y_pred_RF))

cm_RF = ConfusionMatrix(y_test, y_pred_RF)
print(cm_RF)
cm_RF.print_stats()
cm_RF.plot()


#===============================================================================================================  
'''

'''
#================================================================================================================    
#USING Gaussian Naive Bayes
NB = GaussianNB()
NB.fit(X_train, y_train)    

y_pred_NB = NB.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred_NB)
print('ACCURACY -> ',accuracy_score(y_test, y_pred_NB))    
print('PRECISION -> ',precision_score(y_test, y_pred_NB, average= 'weighted'))
print('RECALL -> ',recall_score(y_test, y_pred_NB, average= 'weighted')) 
print('F-1 SCORE -> ',f1_score(y_test, y_pred_NB, average= 'weighted'))
print(classification_report(y_test, y_pred_NB))
cm_NB = ConfusionMatrix(y_test, y_pred_NB)
print(cm_NB)
cm_NB.print_stats()
cm_NB.plot()

#================================================================================================================  
'''


'''
#================================================================================================================    
#USING MultinomialNB
MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)    

y_pred_MultiNB = MultiNB.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred_MultiNB)
print('ACCURACY -> ',accuracy_score(y_test, y_pred_MultiNB))    
print('PRECISION -> ',precision_score(y_test, y_pred_MultiNB, average= 'weighted'))
print('RECALL -> ',recall_score(y_test, y_pred_MultiNB, average= 'weighted')) 
print('F-1 SCORE -> ',f1_score(y_test, y_pred_MultiNB, average= 'weighted'))
print(classification_report(y_test, y_pred_MultiNB))
cm_MnNB = ConfusionMatrix(y_test, y_pred_MultiNB)
print(cm_MnNB)
cm_MnNB.print_stats()
cm_MnNB.plot()

#================================================================================================================ 
'''

'''
#================================================================================================================    
#USING DecisionTreeClassifier
Dtree = DecisionTreeClassifier()
Dtree.fit(X_train, y_train)    

y_pred_Dtree = Dtree.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred_Dtree)
print('ACCURACY -> ',accuracy_score(y_test, y_pred_Dtree))    
print('PRECISION -> ',precision_score(y_test, y_pred_Dtree, average= 'weighted'))
print('RECALL -> ',recall_score(y_test, y_pred_Dtree, average= 'weighted')) 
print('F-1 SCORE -> ',f1_score(y_test, y_pred_Dtree, average= 'weighted'))
print(classification_report(y_test, y_pred_Dtree))

cm_DeTree = ConfusionMatrix(y_test, y_pred_Dtree)
print(cm_DeTree)
cm_DeTree.print_stats()
cm_DeTree.plot()

#================================================================================================================ 
'''

