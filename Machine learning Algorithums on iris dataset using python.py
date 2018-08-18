
# coding: utf-8

# In[175]:


import sys
print('Python: {}'.format(sys.version))

import scipy
print('scipy: {}'.format(scipy.__version__))

import numpy
print('numpy:{}'.format(numpy.__version__))

import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

import pandas
print('pandas: {}'.format(pandas.__version__))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[176]:


#Load libaries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


# In[177]:


url ='C:/Users/Administrator/Desktop/Data sets 3 sem/iris.csv'

#names = ['Sepel-length','sepel-width','petel-length','petal-width','class']

dataset = pandas.read_csv(url)
print(dataset)


# In[178]:


print(dataset.shape)


# In[179]:


print(dataset.head(30))


# In[180]:


print(dataset.describe())


# In[181]:


print(dataset.groupby('Species').size())


# In[182]:



dataset.plot(kind='box', subplots = True, layout=(2,4), sharex = False, sharey = False)
plt.show()


# In[183]:


dataset.hist()
plt.show()


# In[184]:


scatter_matrix(dataset)
plt.show()


# In[185]:


#validationdataset or training datase
array=dataset.values
x = array[:,0:4]
y = array[:,4]

validation_size =0.40
seed = 6
x_train,x_test, y_train,y_test = model_selection.train_test_split(x,y, test_size= validation_size, random_state = seed)


# In[186]:


seed = 6

scoring = 'accuracy'


# In[187]:


#spot check algorithum
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Linear DiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('KNeighbors Classifier', KNeighborsClassifier()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Gaussian NB', GaussianNB()))
models.append(('Support Vector Machine', SVC()))

# evaluate each modelin turn

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    print(msg)
    

