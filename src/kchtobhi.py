import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
data = pd.read_csv("voice.csv")
sns.set(style="white", color_codes=True)
# Let's check what our dataset looks like
print data.head()
# Let's check dimension of our dataset
print data.shape
# There are 3168 rows and 21 columns so 20 features and a label
# Check if there are null values in any column of our dataset
# Let's check 5 point summary of data
print data.describe()
print data.isnull().sum()
# There are no null values in our dataset
# Let's check if our data is biased
print data["label"].value_counts()
plt.style.use("ggplot")
k = plt.figure(1)
data.label.value_counts().plot(kind ="bar",stacked = True, title = "Label Distribution")
k.show()
# Distribution of Male and Female
plt.style.use("fivethirtyeight")
sns.FacetGrid(data, hue="label", size=6) \
   .map(sns.kdeplot, "meanfun") \
   .add_legend()
#Finding Correlation
cm = data.corr()
g= plt.figure(3)
sns.heatmap(cm, square = True)
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
g.show()
# Let's find correlation between features to remove highly correlated features
sns.pairplot(data[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']], 
                 hue='label', size=2)
plt.show()
# meanfrequncy and centroid are highly correlated. Let's investigate further
x = data.drop(["label"],axis=1)
# Let's check variance  and remove all features that don't meet threshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(x)
print sel.variances_
#All the features have good variance and none were removed
# Now Let's try SelectK for feature selection which uses best k features
#data['label'] = data['label'].replace(0, 'Female',inplace=True)
#data['label'] = data['label'].replace(1, 'Male',inplace=True)
#data=data.iloc[:500]
data.label.replace(['male', 'female'], [1, 0], inplace=True)

print(data.shape)

X, y = x, data['label']
X_new = SelectKBest(chi2, k=16).fit_transform(X, y)
print X_new.shape
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=1)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC 
from sklearn import datasets, neighbors, linear_model
#knn = neighbors.KNeighborsClassifier()
#logistic = linear_model.LogisticRegression()
#clf = SVC(kernel='poly').fit(X_train, y_train)

#print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
#a = knn.predict(X_test)
#print(y_train)
#print('LogisticRegression score: %f'
   # % logistic.fit(X_train, y_train).score(X_test, y_test))
#print(clf.score(X_test, y_test))


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10, 100, 1000]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters, cv=5)
clf.fit(X_train, y_train)
print(clf.cv_results_)
# Accuracy on test set
clf.score(X_test,y_test)
Prediction = clf.predict(X_test)

"""
m = SVC(kernel="rbf")
#m = SVC(kernel="linear")
k_fold = KFold(n_splits=3)
sv = m.fit(X_new, y)
cross_val_score(sv, X_new, y, cv=k_fold)
"""
plt.clf()
# ROC Curve
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, Prediction)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#Reporting Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, Prediction,labels=[1,0])
