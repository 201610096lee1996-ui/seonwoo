#!/usr/bin/env python
# coding: utf-8

# # 1. 로지스틱 회귀

# # 1) 합격여부 데이터 읽기

# In[2]:


import numpy as np

import pandas as pd
data = pd.read_csv('admit.txt', names=['ex1','ex2','Admitted'])
print data

x = np.c_[data['ex1'], data['ex2']]
y = data['Admitted']
m = len(data)


# In[3]:


print x.shape, y.shape


# # 2) 데이터 그리기

# In[4]:


pos = []
neg = []

for(i, val)in enumerate(y):
    if val==1:
        pos.append(i)
    else:
        neg.append(i)
print pos
print neg


# In[6]:


import matplotlib.pyplot as plt
plt.plot(x[pos,0].reshape(-1), x[pos,1].reshape(-1), 'b+',label='Admitted')
plt.plot(x[neg,0].reshape(-1), x[neg,1].reshape(-1), 'ro', label='Not admitted')
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc='upper right')
plt.show()


# # 3) 학습

# In[8]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='liblinear',C=10)
log_reg.fit(x,y)


# In[9]:


log_reg.predict([[30,70],
                [50,90]])


# # 4) Decision boundary

# In[10]:


x_min, x_max = x[:,0].min(), x[:,0].max()
y_min, y_max = x[:,1].min(), x[:,1].max()
h = .2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = log_reg.predict(np.c_[xx.ravel(),yy.ravel()])

z = z.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Paired)

plt.plot(x[pos,0].reshape(-1), x[pos,1].reshape(-1), 'b+', label='Passed')
plt.plot(x[neg,0].reshape(-1), x[neg,1].reshape(-1), 'ro', label='Not passed')
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend(loc='upper right')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()


# # 2. 로지스틱 회귀 + 정규화

# # 1) 불량여부 데이터 읽기

# In[35]:


import numpy as np

import pandas as pd
data = pd.read_csv('qa.txt', names=['t1','t2','Passed'])
print data

x = np.c_[data['t1'], data['t2']]
y = data['Passed']
m = len(data)


# In[12]:


print x.shape, y.shape


# # 2) 그래프 그리기

# In[13]:


pos = []
neg = []

for(i, val) in enumerate(y):
    if val==1:
        pos.append(i)
    else:
        neg.append(i)
print pos
print neg


# In[14]:


import matplotlib.pyplot as plt
plt.plot(x[pos,0].reshape(-1), x[pos,1].reshape(-1), 'b+', label='Passed')
plt.plot(x[neg,0].reshape(-1), x[neg,1].reshape(-1), 'ro', label='Failed')
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend(loc='upper right')
plt.show()


# # 3) 학습

# In[16]:


from sklearn.preprocessing import PolynomialFeatures
degree = 2
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
x_poly = poly_features.fit_transform(x)

print x[0]
print x_poly[0]
print x_poly[0].shape


# In[36]:


from sklearn.preprocessing import PolynomialFeatures
degree = 6
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
x_poly = poly_features.fit_transform(x)

print x[0]
print x_poly[0].shape


# In[39]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(penalty='l2', solver='liblinear', C=1e-1)
log_reg.fit(x_poly, y)


# # 4) Decision boundary

# In[40]:


u = np.linspace(-1, 1.5, 300)
v = np.linspace(-1, 1.5, 300)
z = np.zeros((len(u), len(v)))

for i in range(len(u)):
    a=[]
    for j in range(len(v)):
        a.append(np.array([u[i], v[j]]))
    
    my_data = poly_features.fit_transform( a )
    z[i] = log_reg.predict(my_data)

plt.contour(u,v,z,0)

plt.plot(x[pos,0].reshape(-1), x[pos,1].reshape(-1), 'b+', label='Passed')
plt.plot(x[neg,0].reshape(-1), x[neg,1].reshape(-1), 'ro', label='Failed')
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend(loc='upper right')
plt.show()


# # 5) 로지스틱 회귀의 성능 측정법

# In[41]:


y_pred=log_reg.predict(x_poly)
print y_pred

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y, y_pred)
print conf_mat
plt.matshow(conf_mat, cmap=plt.cm.gray)
plt.show()

from sklearn.metrics import precision_score, recall_score
print "precision_score: ", precision_score(y, y_pred)
print "recall_score: ",recall_score(y,y_pred)

from sklearn.metrics import f1_score
print "F1_score: ", f1_score(y, y_pred)


# In[42]:


y_scores = log_reg.decision_function(x_poly)
print y_scores


# In[43]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
print "roc_auc_score: ", roc_auc_score(y, y_scores)


# In[ ]:




