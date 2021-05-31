#!/usr/bin/env python
# coding: utf-8

# # Bank Loan Analysis

# # Notebook Outline

# 1. [**Importing Libraries**](#Importing-Libraries)   
# 2. [**Data Load**](#Data-Load)  
# 3. [**Data Visualizations**](#Data-Visualizations)  
# 4. [**Classification**](#Classification)  
#     4-1 [**Prepare Data**](#Prepare-Data)  
#     4-2 [**Random Forest Classifier**](#Random-Forest-Classifier)  

# # Importing Libraries

# In[1]:


get_ipython().system('pip install bubbly')


# In[2]:


# for basic operations
import numpy as np
import pandas as pd

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for advanced visualizations 
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from bubbly.bubbly import bubbleplot


# for model explanation
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# for classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# # Data Load

# In[3]:


# reading the data
data = pd.read_csv('UniversalBank.csv')

# getting the shape
data.shape


# In[4]:


# reading the head of the data

data.head()


# In[5]:


# describing the data

data.describe()


# 'ID' and 'ZIP Code' can not be treated as features, so delete them.

# In[6]:


# drop 'ID' and 'ZIP Code'
data = data.drop(["ID","ZIP Code"],axis=1)


# In[7]:


# check missing data
data.isnull().sum()


# No missing value.

# # Data Visualizations

# In[8]:


figure = bubbleplot(dataset = data, x_column = 'Experience', y_column = 'Income', 
    bubble_column = 'Personal Loan', time_column = 'Age', size_column = 'Mortgage', color_column = 'Personal Loan', 
    x_title = "Experience", y_title = "Income", title = 'Experience vs Income. vs Age vs Mortgage vs Personal Loan',
    x_logscale = False, scale_bubble = 3, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# The figure shows the relationship between income and years of experience for each age.    
# The size of the sphere represents the size of the mortgage.  
# If the sphere is red, it has a personal loan.  

# In[9]:


# making a heat map
plt.rcParams['figure.figsize'] = (20, 15)
plt.style.use('ggplot')

sns.heatmap(data.corr(), annot = True, cmap = 'Wistia')
plt.title('Heatmap for the Dataset', fontsize = 20)
plt.show()


# Let's check the overall correlation in the heat map.  
# 'Age' and 'Exprerience' has so high correlation.  
# So I'll delete one of them.    

# In[10]:


# checking the distribution of age

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 5)
sns.distplot(data['Age'], color = 'cyan')
plt.title('Distribution of Age', fontsize = 20)
plt.show()


# The above chart shows distribution of age.  
# Age is well-balanced.

# In[11]:


# plotting a donut chart for visualizing 'Personal Loan','Securities Account','CD Account','Online','CreditCard'

fig, ax = plt.subplots(1,5,figsize=(20,20))
columns = ['Personal Loan','Securities Account','CD Account','Online','CreditCard']

for i,column in enumerate(columns):
    plt.subplot(1,5,i+1)
    size = data[column].value_counts()
    colors = ['lightblue', 'lightgreen']
    labels = "No", "Yes"
    explode = [0, 0.01]

    my_circle = plt.Circle((0, 0), 0.7, color = 'white')

    plt.rcParams['figure.figsize'] = (20, 20)
    plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
    plt.title('Distribution of {}'.format(column), fontsize = 15)
    p = plt.gcf()
    p.gca().add_artist(my_circle)
plt.legend()
plt.show()


# The above pie chart shows distribution of personal loan, securities account, CD account, online, and credit card.  
# personal loan will be target of classification.  
# So it's unbalance dataset.  

# In[12]:


# show relation of family with personal loan
  
plt.rcParams['figure.figsize'] = (12, 9)
dat = pd.crosstab(data['Personal Loan'], data['Family']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar', 
                                                 stacked = False, 
                                                 color = plt.cm.rainbow(np.linspace(0, 1, 4)))
plt.title('Relation of Family with Personal Loan', fontsize = 20, fontweight = 30)
plt.show()


# The above chart shows relation of family with personal loan.  
# It seems that the more families there are, the more likely they are to make a loan.  

# In[13]:


# show relation of education with personal loan
  
plt.rcParams['figure.figsize'] = (12, 9)
dat = pd.crosstab(data['Personal Loan'], data['Education']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar', 
                                                 stacked = False, 
                                                 color = plt.cm.rainbow(np.linspace(0, 1, 4)))
plt.title('Relation of Education with Personal Loan', fontsize = 20, fontweight = 30)
plt.show()


# The above chart shows relation of Education with personal loan.  
# It seems that people who have a high education tend to have a loan.  

# In[14]:


# show relation of income with personal loan

plt.rcParams['figure.figsize'] = (12, 9)
sns.boxplot(data['Personal Loan'], data['Income'], palette = 'viridis')
plt.title('Relation of Income with Personal Loan', fontsize = 20)
plt.show()


# There is a clear difference in the relationship between income and personal loans.

# In[15]:


# show relation of CCAvg with personal loan

plt.rcParams['figure.figsize'] = (12, 9)
sns.violinplot(data['Personal Loan'], data['CCAvg'], palette = 'colorblind')
plt.title('Relation of CCAvg with Target', fontsize = 20, fontweight = 30)
plt.show()


# Although not as much as income, CCAvg is also likely to be related to the availability of personal loans.

# In[16]:


# show relation of mortgage with personal loan

plt.rcParams['figure.figsize'] = (12, 9)
sns.violinplot(data['Personal Loan'], data['Mortgage'], palette = 'colorblind')
plt.title('Relation of Mortgage with Target', fontsize = 20, fontweight = 30)
plt.show()


# Mortgages seem to be unrelated to personal loans.

# # Classification 

# ## Prepare Data

# In[17]:


# Give meaning to category data 

data['Securities Account'][data['Securities Account'] == 0] = 'No'
data['Securities Account'][data['Securities Account'] == 1] = 'Yes'

data['CD Account'][data['CD Account'] == 0] = 'No'
data['CD Account'][data['CD Account'] == 1] = 'Yes'

data['Online'][data['Online'] == 0] = 'No'
data['Online'][data['Online'] == 1] = 'Yes'

data['CreditCard'][data['CreditCard'] == 0] = 'No'
data['CreditCard'][data['CreditCard'] == 1] = 'Yes'


# In[18]:


data['Securities Account'] = data['Securities Account'].astype('object')
data['CD Account'] = data['CD Account'].astype('object')
data['Online'] = data['Online'].astype('object')
data['CreditCard'] = data['CreditCard'].astype('object')

# drop age (Because the correlation with experience is high.)
data = data.drop(["Age"],axis=1)


# In[19]:


# taking the labels out from the data

y = data['Personal Loan']
data = data.drop('Personal Loan', axis = 1)

print("Shape of y:", y.shape)


# In[20]:


# One hot encoding
data = pd.get_dummies(data, drop_first=True)


# In[21]:


# check data
data.head()


# In[22]:


# Split the data
x = data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# getting the shapes
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)


# ## Random Forest Classifier

# In[23]:


# MODELLING
# Random Forest Classifier

model = RandomForestClassifier(n_estimators = 50, max_depth = 5)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
y_pred_quant = model.predict_proba(x_test)[:, 1]
y_pred = model.predict(x_test)

# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

# cofusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')

# classification report
cr = classification_report(y_test, y_pred)
print(cr)


# In[24]:


import pickle
# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(model, open('model.pkl','wb'))


# In[ ]:




