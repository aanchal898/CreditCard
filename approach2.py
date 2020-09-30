import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
dataset=pd.read_csv('..\creditcard.csv')


##############################################################################

## Get the Fraud and the normal dataset 
fraud = dataset[dataset['Class']==1]
normal = dataset[dataset['Class']==0]

Fraud = dataset[dataset['Class']==1]
Valid = dataset[dataset['Class']==0]
from sklearn.utils import resample
df_valid = (Valid).sample(n=4000, random_state=42) 

df_list = [df_valid, Fraud]
dataset = pd.concat(df_list)


print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))

#Create independent and Dependent Features
columns = dataset.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = dataset[columns]
y = dataset[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
##############################################################################

count_classes=pd.value_counts(dataset['Class'],sort=True)
count_classes.plot(kind='bar',rot=0)
plt.title('Transaction class distribution')
plt.xlabel('Class')
plt.ylabel('frequency')
##############################################################################

import seaborn as sns
#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")



# Select upper triangle of correlation matrix
upper = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.85
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
# Drop features 
dataset=dataset.drop(dataset[to_drop], axis=1)

import sklearn
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#ann classifier
import keras
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
#initializing the ANN
classifier_ann= tf.keras.models.Sequential()
#adding input layer and first hidden layer
classifier_ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#second layer
classifier_ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#output layer
classifier_ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#compiling the ann
classifier_ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to training set
classifier_ann.fit(X_train,y_train,batch_size=32, epochs=5, class_weight={0:1, 1:10})


#predicting new test results
y_pred_ann=classifier_ann.predict(X_test)
y_pred_ann=(y_pred_ann>0.5)
y_pred_proba_ann = classifier_ann.predict_proba(X_test)
score_ann= accuracy_score(y_test, y_pred_ann)

from sklearn.metrics import confusion_matrix
cm_ann=confusion_matrix(y_test,y_pred_ann, normalize='true')
 

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
f1_ann = f1_score(y_test, y_pred_ann)
prec_ann = precision_score(y_test, y_pred_ann)
recall_ann = recall_score(y_test, y_pred_ann) 

cm_ann = pd.DataFrame(cm_ann, range(2), range(2))
sns.set(font_scale=1.4) #label size
sns.heatmap(cm_ann, annot=True, annot_kws={"size": 16}) #font size
plt.show()

from sklearn.metrics import precision_recall_curve

prec_ann_sequence, recall_ann_sequence , _ = precision_recall_curve(y_test, y_pred_proba_ann[:, 0])

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#logistic regression
#fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(class_weight={0:1, 1:10})
classifier_lr.fit(X_train,y_train)

#predicting results
y_pred_lr=classifier_lr.predict(X_test)
y_pred_problr = classifier_lr.decision_function(X_test)
y_pred_proba_lr=classifier_lr.predict_proba(X_test)
score_lr= accuracy_score(y_test, y_pred_lr)
#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm_lr=confusion_matrix(y_test, y_pred_lr, normalize='true')


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
f1_lr = f1_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr) 

prec_lr_sequence, recall_lr_sequence , _ = precision_recall_curve(y_test, y_pred_proba_lr[:, 1])

from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', probability=True, random_state = 0, class_weight={0: 1, 1:10})
#svm_classifier = SVC(kernel='linear', random_state = 0, probability=True)
svm_classifier.fit(X_train,y_train)

#predicting results
y_pred_svm=svm_classifier.predict(X_test)
y_pred_probsvm = svm_classifier.decision_function(X_test)
y_pred_proba_svm=svm_classifier.predict_proba(X_test)
score_svm= accuracy_score(y_test, y_pred_svm)


#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm_svm=confusion_matrix(y_test, y_pred_svm, normalize='true')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
f1_svm = f1_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm) 
prec_svm_sequence, recall_svm_sequence , _ = precision_recall_curve(y_test, y_pred_proba_svm[:, 1])

##############################
#randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
randomforest_classifier=RandomForestClassifier(n_estimators=20, class_weight={0:1, 1:10})
randomforest_classifier.fit(X_train,y_train)

y_pred_rf=randomforest_classifier.predict(X_test)
y_pred_proba_rf = randomforest_classifier.predict_proba(X_test)
score_rf= accuracy_score(y_test, y_pred_rf)
from sklearn.metrics import confusion_matrix
cm_rf=confusion_matrix(y_test,y_pred_rf, normalize='true')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
f1_rf = f1_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf) 

prec_rf_sequence, recall_rf_sequence , _ = precision_recall_curve(y_test, y_pred_proba_rf[:, 1])


#decision tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(min_samples_leaf = 20, class_weight = {0:1, 1:10})
dt_classifier = dt_classifier.fit(X_train, y_train)


y_pred_dt = dt_classifier.predict(X_test)
y_pred_proba_dt = dt_classifier.predict_proba(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

score_dt= accuracy_score(y_test, y_pred_dt)
from sklearn.metrics import confusion_matrix
cm_dt=confusion_matrix(y_test,y_pred_dt, normalize='true')


f1_dt = f1_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt) 

prec_dt_sequence, recall_dt_sequence , _ = precision_recall_curve(y_test, y_pred_proba_dt[:, 1])

df_cm_lr = pd.DataFrame(cm_lr, range(2), range(2))
sns.set(font_scale=1.4) #label size
sns.heatmap(df_cm_lr, annot=True, annot_kws={"size": 16}) #font size


plt.show()

df_cm_dt = pd.DataFrame(cm_dt, range(2), range(2))
sns.set(font_scale=1.4) #label size
sns.heatmap(df_cm_dt, annot=True, annot_kws={"size": 16}) #font size


plt.show()

df_cm_svm = pd.DataFrame(cm_svm, range(2), range(2))
sns.set(font_scale=1.4) #label size
sns.heatmap(df_cm_svm, annot=True, annot_kws={"size": 16}) #font size


plt.show()

df_cm_rf = pd.DataFrame(cm_rf, range(2), range(2))
sns.set(font_scale=1.4) #label size
sns.heatmap(df_cm_rf, annot=True, annot_kws={"size": 16}) #font size

plt.show()

cm_ann = pd.DataFrame(cm_ann, range(2), range(2))
sns.set(font_scale=1.4) #label size
sns.heatmap(cm_ann, annot=True, annot_kws={"size": 16}) #font size
plt.show()

fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax1.plot(recall_ann_sequence,prec_ann_sequence,c='b',label='ANN')
ax1.plot(recall_lr_sequence, prec_lr_sequence, c='g', label='LR')
ax1.plot(recall_rf_sequence, prec_rf_sequence, c='r', label='RF')
ax1.plot(recall_dt_sequence, prec_dt_sequence, c='y', label='DT')
ax1.plot(recall_svm_sequence, prec_svm_sequence, c='c', label='SVM')

ax1.legend(loc='lower left')    


plt.show()
