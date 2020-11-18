#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from collections import Counter as c
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[2]:


data=pd.read_csv(r"C:\Users\susmitha\Documents\credit_train.csv")







# In[5]:object_train_df=data.select_dtypes(include=['object']) 
object_train_df=data.select_dtypes(include=['object']) 
num_train_df=data.select_dtypes(include=['int','float']) 
data.dropna(subset=['Loan Status'], inplace = True)
le = preprocessing.LabelEncoder()
data['Loan Status'] = le.fit_transform(data['Loan Status'])

#loan status is the target column, assigned to be zero here,it gives the count of charged off people
coffvalue = data[data['Loan Status'] == 0]['Loan Status'].count()
#loan status is the target column, assigned to be one here,it gives the count of fully paid people
fpaidvalue = data[data['Loan Status'] == 1]['Loan Status'].count()
data1 = {"Counts":[coffvalue, fpaidvalue] }
statusDF = pd.DataFrame(data1, index=["Charged Off", "Fully Paid"])
# statusDF.head()
statusDF.plot(kind='bar', title="Status of the Loan")
data['Term'].replace(("Short Term","Long Term"),(0,1), inplace=True)


scount = data[data['Term'] == 0]['Term'].count()
lcount = data[data['Term'] ==1]['Term'].count()

data1 = {"Counts":[scount, lcount]}
#gives the count of short and long term
termDF = pd.DataFrame(data1, index=["Short Term", "Long Term"])
termDF.head()

data['Credit Score'] = data['Credit Score'].apply(lambda val: (val /10) if val>850 else val)

do_nothing = lambda: None
cscoredf = data[data['Term']==0]
stermAVG = cscoredf['Credit Score'].mean()
lscoredf = data[data['Term']==1]
ltermAVG = lscoredf['Credit Score'].mean()
data.loc[(data.Term ==0) & (data['Credit Score'].isnull()),'Credit Score'] = stermAVG
data.loc[(data.Term ==1) & (data['Credit Score'].isnull()),'Credit Score'] = ltermAVG


data['Credit Score'] = data['Credit Score'].apply(lambda val: "Poor" if np.isreal(val)
                                                  and val < 580 else val)
data['Credit Score'] = data['Credit Score'].apply(lambda val: "Average" if np.isreal(val)
                                                  and (val >= 580 and val < 670) else val)
data['Credit Score'] = data['Credit Score'].apply(lambda val: "Good" if np.isreal(val) 
                                                  and (val >= 670 and val < 740) else val)
data['Credit Score'] = data['Credit Score'].apply(lambda val: "Very Good" if np.isreal(val) 
                                                  and (val >= 740 and val < 800) else val)
data['Credit Score'] = data['Credit Score'].apply(lambda val: "Exceptional" if np.isreal(val) 
                                                  and (val >= 800 and val <= 850) else val)

data['Annual Income'].fillna(data['Annual Income'].mean(), inplace=True)
print(c(data['Credit Score']))  
data['Credit Score'] = le.fit_transform(data['Credit Score'])  #applying label encoder
c(data['Credit Score'])

print(c(data['Home Ownership']))
data['Home Ownership'] = le.fit_transform(data['Home Ownership'])
print(c(data['Home Ownership']))

data['Years in current job']=data['Years in current job'].str.extract(r"(\d+)")
data['Years in current job'] = data['Years in current job'].astype(float)
expmean = data['Years in current job'].mean()
data['Years in current job'].fillna(expmean, inplace=True)
data['Years in current job'].fillna(expmean, inplace=True)

data = data.drop(['Loan ID','Customer ID','Purpose'], axis=1)


data['Credit Problems'] = data['Number of Credit Problems'].apply(lambda x: "No Credit Problem" if x==0 
                        else ("Some Credit promblem" if x>0 and x<5 else "Major Credit Problems"))
print(c(data['Credit Problems']))
data['Credit Problems'] = le.fit_transform(data['Credit Problems'])
print(c(data['Credit Problems']))


data['Credit Age'] = data['Years of Credit History'].apply(lambda x: "Short Credit Age" if x<5 
                                else ("Good Credit Age" if x>5 and x<17 else "Exceptional Credit Age"))
print(c(data['Credit Age']))
data['Credit Age'] = le.fit_transform(data['Credit Age'])
print(c(data['Credit Age']))
data = data.drop(['Months since last delinquent','Number of Open Accounts',
                  'Maximum Open Credit','Current Credit Balance','Monthly Debt'],axis=1)


data['Tax Liens'] = data['Tax Liens'].apply(lambda x: "No Tax Lien" if x==0
                                else ("Some Tax Liens" if x>0 and x<3 else "Many Tax Liens"))
print(c(data['Tax Liens']))
data['Tax Liens'] = le.fit_transform(data['Tax Liens'])
print(c(data['Tax Liens']))

data['Bankruptcies'] = data['Bankruptcies'].apply(lambda x: "No bankruptcies" if x==0 
                            else ("Some Bankruptcies" if x>0 and x<3 else "Many Bankruptcies"))
print(c(data['Bankruptcies']))
data['Bankruptcies'] = le.fit_transform(data['Bankruptcies'])
print(c(data['Bankruptcies']))


meanxoutlier = data[data['Annual Income'] < 99999999.00 ]['Annual Income'].mean()
stddevxoutlier = data[data['Annual Income'] < 99999999.00 ]['Annual Income'].std()
poorline = meanxoutlier -  stddevxoutlier
richline = meanxoutlier + stddevxoutlier
data['Annual Income'] = data['Annual Income'].apply(lambda x: "Low Income" if x<=poorline 
                            else ("Average Income" if x>poorline and x<richline else "High Income"))
print(c(data['Annual Income']))
data['Annual Income'] = le.fit_transform(data['Annual Income'])
print(c(data['Annual Income']))


y = data['Loan Status']
X = data.drop(['Loan Status'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt =dt.predict(X_test)  #prediction
c(y_pred_dt)

import pickle    #importing the pickle file

pickle.dump(dt,open('loan.pkl','wb')) 
