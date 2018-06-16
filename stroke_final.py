import numpy as np
from sklearn import preprocessing, cross_validation
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv')
df.drop(['id'],1,inplace=True)
df.drop(['work_type'],1,inplace=True)
df['gender'] = df['gender'].map({'Female':1,'Male':0,'Other':-99999})
df['ever_married'] = df['ever_married'].map({'No':0,'Yes':1}).astype(float)
df['Residence_type'] = df['Residence_type'].map({'Rural':1,'Urban':0}).astype(float)
df['smoking_status'] = df['smoking_status'].map({'formerly smoked':2,'never smoked':0,'smokes':1})
df['smoking_status'].fillna(-99999,inplace=True)
df['bmi'].fillna(-99999,inplace=True)
df.astype(float).values.tolist()

X = np.array(df.drop(['stroke'],1))
y = np.array(df['stroke'])

X_train,X_test, y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = RandomForestClassifier(max_depth=8, random_state=2)

td = pd.read_csv('test.csv')
td.drop(['id'],1,inplace=True)
td.drop(['work_type'],1,inplace=True)
td['gender'] = td['gender'].map({'Female':1,'Male':0,'Other':-99999})
td['ever_married'] = td['ever_married'].map({'No':0,'Yes':1}).astype(float)
td['Residence_type'] = td['Residence_type'].map({'Rural':1,'Urban':0}).astype(float)
td['smoking_status'] = td['smoking_status'].map({'formerly smoked':2,'never smoked':0,'smokes':1})
td['smoking_status'].fillna(-99999,inplace=True)
td['bmi'].fillna(-99999,inplace=True)
test_data = td.astype(float).values.tolist()
z = clf.fit(X_train,y_train)
confidence = clf.score(X_test,y_test)

print(confidence)

p = clf.predict(test_data)
pro = clf.predict_proba(test_data)

np.savetxt("C:/Users/AK/Desktop/result_pro.csv", pro, delimiter=",")
