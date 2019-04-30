import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#import data
data = pd.read_csv('../data/breast-cancer-wisconsin.csv',header=None)

#set column names
data.columns = ['Sample Code Number','Clump Thickness','Uniformity of Cell Size',
'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

#view top 10 rows
#print(data.head(10))

#Clean the data and rename the class values as 0/1 for model building (where 1 represents a malignant case). 
#Also, let's observe the distribution of the class.

data = data.drop(['Sample Code Number'], axis=1) #Drop the 1st coloumn
data = data[data['Bare Nuclei'] != '?'] #Remove rows with missing data
data['Class'] = np.where(data['Class']==2, 0, 1)  #Change the Class representation; 0 for Benign and 1 for malignant
print('Class distribution:')
print(data['Class'].value_counts()) #Class distribution


#Split data into attributes and class
X = data.drop(['Class'],axis=1)
y = data['Class']

#perform training and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


##Grid Search
clf = LogisticRegression()
grid_values = {'penalty':['l1','l2'], 'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring='recall')
grid_clf_acc.fit(X_train,y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)


# New Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))

#Logistic Regression (Grid Search) Confusion matrix
confusion_matrix(y_test,y_pred_acc)