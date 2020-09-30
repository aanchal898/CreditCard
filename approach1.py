import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
dataset=pd.read_csv('../creditcard.csv')

##############################################################################

count_classes=pd.value_counts(dataset['Class'],sort=True)
count_classes.plot(kind='bar',rot=0)
plt.title('Transaction class distribution')
plt.xlabel('Class')
plt.ylabel('frequency')
##############################################################################

## Get the Fraud and the normal dataset 
fraud = dataset[dataset['Class']==1]
normal = dataset[dataset['Class']==0]

Fraud = dataset[dataset['Class']==1]
Valid = dataset[dataset['Class']==0]
from sklearn.utils import resample
df_valid = (Valid).sample(n=4000, random_state=42)
df_fraud = resample(Fraud, replace=True, n_samples = 2000, random_state = 24)

df_list = [df_valid, df_fraud]
dataset = pd.concat(df_list)

count_classes=pd.value_counts(dataset['Class'],sort=True)
count_classes.plot(kind='bar',rot=0)
plt.title('Transaction class distribution')
plt.xlabel('Class')
plt.ylabel('frequency')

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

##################################################################################

## Correlation
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

################################################################################

import sklearn
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#################################################################################

#ann classifier

import keras
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
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
classifier_ann.fit(X_train,y_train,batch_size=32, epochs=5)


#predicting new test results
y_pred_ann=classifier_ann.predict(X_test)
y_pred_ann=(y_pred_ann>0.5)
y_pred_proba_ann = classifier_ann.predict_proba(X_test)
score_ann= accuracy_score(y_test, y_pred_ann)

from sklearn.metrics import confusion_matrix
cm_ann=confusion_matrix(y_test,y_pred_ann, normalize='true')
 

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
f1_ann = f1_score(y_test, y_pred_ann)
prec_ann = precision_score(y_test, y_pred_ann)
recall_ann = recall_score(y_test, y_pred_ann) 
prec_ann_sequence, recall_ann_sequence , _ = precision_recall_curve(y_test, y_pred_proba_ann[:, 0])


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc




########################################################################################

#logistic regression
#fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression()
classifier_lr.fit(X_train,y_train)

#predicting results
y_pred_lr=classifier_lr.predict(X_test)
y_pred_problr = classifier_lr.decision_function(X_test)
y_pred_proba_lr=classifier_lr.predict_proba(X_test)

score_lr= accuracy_score(y_test, y_pred_lr)
#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm_lr=confusion_matrix(y_test, y_pred_lr, normalize = 'true')


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
f1_lr = f1_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
prec_lr_sequence, recall_lr_sequence , _ = precision_recall_curve(y_test, y_pred_proba_lr[:, 1])

###################################################################################


########################################################################################

#fitting svm
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear',random_state=0,probability=True)
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
from sklearn.metrics import precision_recall_curve
f1_svm = f1_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm) 
prec_svm_sequence, recall_svm_sequence , _ = precision_recall_curve(y_test, y_pred_proba_svm[:, 1])

#####################################################################################

#randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
randomforest_classifier=RandomForestClassifier(n_estimators=20)
randomforest_classifier.fit(X_train,y_train)



y_pred_rf=randomforest_classifier.predict(X_test)
y_pred_proba_rf = randomforest_classifier.predict_proba(X_test)
  
score_rf= accuracy_score(y_test, y_pred_rf)
from sklearn.metrics import confusion_matrix
cm_rf=confusion_matrix(y_test,y_pred_rf, normalize='true')



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
f1_rf = f1_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
prec_rf_sequence, recall_rf_sequence , _ = precision_recall_curve(y_test, y_pred_proba_rf[:, 1])


################################################################################

#decision tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(min_samples_leaf = 20)
dt_classifier = dt_classifier.fit(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)
y_pred_proba_dt = dt_classifier.predict_proba(X_test)
#y_pred_probdt=dt_classifier.decision_function(X_test)

score_dt= accuracy_score(y_test, y_pred_dt)

from sklearn.metrics import confusion_matrix
cm_dt=confusion_matrix(y_test,y_pred_dt, normalize ='true')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
f1_dt = f1_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt) 
prec_dt_sequence, recall_dt_sequence , _ = precision_recall_curve(y_test, y_pred_proba_dt[:, 1])

####################################################################################

###################################################################################

for i in range(91):
    y_pred_proba_lr[:,1][i] = round(y_pred_proba_lr[:, 1][i], 1)
    
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
lr_fpr,lr_tpr,threshold_lr=roc_curve(y_test, y_pred_proba_lr[:,1])
auc_lr=auc(lr_fpr, lr_tpr)

svm_fpr,svm_tpr,threshold_svm=roc_curve(y_test, y_pred_proba_svm[:,1])
auc_svm=auc(svm_fpr, svm_tpr)

#print(roc_auc_score(y_test,y_pred_probrf[:,1]))
rf_fpr,rf_tpr,threshold_rf=roc_curve(y_test, y_pred_proba_rf[:,1])
auc_rf=auc(rf_fpr, rf_tpr)


dt_fpr,dt_tpr,threshold_dt=roc_curve(y_test, y_pred_proba_dt[:,1])
auc_dt=auc(dt_fpr, dt_tpr)

# ann_fpr,ann_tpr,threshold_ann=roc_curve(y_test, y_pred_proba_ann[:,1])
# auc_ann=auc(ann_fpr, ann_tpr)
fpr_ann, tpr_ann, _ = roc_curve(y_test, y_pred_ann)
auc_ann = auc(fpr_ann, tpr_ann)


#################

##################



plt.figure(figsize=(5,5),dpi=100)

plt.plot(lr_fpr,lr_tpr,linestyle='-.',label='LogRegression(auc=%0.3f)'%auc_lr)
plt.plot(svm_fpr,svm_tpr,linestyle='-',label='SVM(auc=%0.3f)'%auc_svm)

plt.plot(rf_fpr,rf_tpr,linestyle='dotted',label='RandomForest(auc=%0.3f)'%auc_rf)
plt.plot(dt_fpr,dt_tpr,linestyle='-.',label='DecisionTree(auc=%0.3f)'%auc_dt)
plt.plot(fpr_ann,tpr_ann,linestyle='-',label='ANN(auc=%0.3f)'%auc_ann)

plt.xlabel('False Positive Rate --->')
plt.ylabel('True Positive Rate --->')

plt.legend()
plt.show()

#################################################################################
#confusion matrix


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

############################################################################
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred_lr)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp1 = plot_precision_recall_curve(classifier_lr, X_test, y_test)
disp1.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

disp2 = plot_precision_recall_curve(randomforest_classifier, X_test, y_test)
disp2.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
disp3 = plot_precision_recall_curve(svm_classifier, X_test, y_test)
disp3.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

disp3 = plot_precision_recall_curve(dt_classifier, X_test, y_test)
disp3.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))



plt.show()


fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

# ax2 = fig.add_subplot(1,2,2)
# ax2.set_xlim([-0.05,1.05])
# ax2.set_ylim([-0.05,1.05])
# ax2.set_xlabel('False Positive Rate')
# ax2.set_ylabel('True Positive Rate')
# ax2.set_title('ROC Curve')



ax1.plot(recall_ann_sequence,prec_ann_sequence,c='b',label='ANN')
ax1.plot(recall_lr_sequence, prec_lr_sequence, c='g', label='LR')
ax1.plot(recall_rf_sequence, prec_rf_sequence, c='r', label='RF')
ax1.plot(recall_dt_sequence, prec_dt_sequence, c='y', label='DT')
ax1.plot(recall_svm_sequence, prec_svm_sequence, c='c', label='SVM')
    # ax2.plot(tpr,fpr,c=k,label=w)
ax1.legend(loc='lower left')    
# ax2.legend(loc='lower left')

plt.show()
