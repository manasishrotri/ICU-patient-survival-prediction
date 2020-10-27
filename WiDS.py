# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:25:25 2020

@author: manas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:01:07 2020

@author: manas
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier ,AdaBoostClassifier
from sklearn.model_selection import train_test_split
import lightgbm 
# roc curve and auc score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

train = pd.read_csv("D:/MS DS WPI/Sem II/WiDS/training_v2.csv")#.drop(drop_cols, axis=1)
test = pd.read_csv("D:/MS DS WPI/Sem II/WiDS/unlabeled.csv")#.drop(drop_cols_test, axis=1)

target = 'hospital_death'
#-------------Convert categorical columns--------------------
train.hospital_death.nunique()
label_count = train.groupby("hospital_death")['patient_id'].nunique()
cat_columns = train.select_dtypes(['object']).columns
#Converting object type columns to category type
train[cat_columns] = train[cat_columns].apply(lambda x: x.astype('category'))
#Converting categorical columns to numeric
train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)
train = train.apply(lambda x: x.fillna(x.mean()),axis=0)
train.head(10)

test = test.drop(['hospital_death'],axis=1)
test[cat_columns] = test[cat_columns].apply(lambda x: x.astype('category'))
test[cat_columns] = test[cat_columns].apply(lambda x: x.cat.codes)
test = test.apply(lambda x: x.fillna(x.mean()),axis=0)
test.head(10)
# creating independent features X and dependant feature Y

y = train['hospital_death']
X = train
X = train.drop('hospital_death',axis = 1)
#test = test.drop('hospital_death',axis = 1)

    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

#----------------Model 1--------------------------------
#model1 = AdaBoostClassifier(random_state=1)
#model1.fit(X_train,y_train)
#train_pred1=(model1.predict_proba(X_val))[:,1]
#test_pred1=(model1.predict_proba(X_test))[:,1]
#print("Ada: ",roc_auc_score(y_true=y_test, y_score=test_pred1))

#----------------Model 2--------------------------------

model2= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model2.fit(X_train,y_train)
train_pred2=(model2.predict_proba(X_val))[:,1]
test_pred2=(model2.predict_proba(X_test))[:,1]
print("GBM: ",roc_auc_score(y_true=y_test, y_score=test_pred2))

#----------------Model 3--------------------------------

model3 = RandomForestClassifier(n_estimators=100,bootstrap = True,max_features = 'sqrt')
model3.fit(X_train,y_train)
train_pred3=(model3.predict_proba(X_val))[:,1]
test_pred3=(model3.predict_proba(X_test))[:,1]
print("RF: ",roc_auc_score(y_true=y_test, y_score=test_pred3))


#----------------Model 4--------------------------------

dtrain = lightgbm.Dataset(X_train, label=y_train)
dvalid = lightgbm.Dataset(X_val, label=y_val)
params = {"objective": "binary", 
          "boosting": "gbdt",
          "metric": "auc",
          "n_jobs":-1,
          "verbose":-1}
lgb = lightgbm.train(params=params, train_set=dtrain, num_boost_round=2000, 
                         valid_sets=[dtrain, dvalid], verbose_eval=250, early_stopping_rounds=500)
train_pred4= (lgb.predict(X_val,num_iteration=lgb.best_iteration))
test_pred4 = (lgb.predict(X_test,num_iteration=lgb.best_iteration))
print("RF: ",roc_auc_score(y_true=y_test, y_score=test_pred4))

#----------------Model 5--------------------------------

#model5= xgb.XGBRegressor(max_depth=3,n_estimators=2200,objective='multi:softprob',
#                              seed=0, silent=True, nthread=-1, learning_rate=0.05)
#model5.fit(X_train,y_train,eval_set=[(X_val, y_val)],eval_metric="merror")
#train_pred5=(model5.predict_proba(X_val))[:,1]
#test_pred5=(model5.predict_proba(X_test))[:,1]
#print("RF: ",roc_auc_score(y_true=y_test, y_score=test_pred5))



train_val1=pd.DataFrame(train_pred1)
train_val2=pd.DataFrame(train_pred2)
train_val3=pd.DataFrame(train_pred3)
train_val4=pd.DataFrame(train_pred4)
test_pred1=pd.DataFrame(test_pred1)
test_pred2=pd.DataFrame(test_pred2)
test_pred3=pd.DataFrame(test_pred3)
test_pred4=pd.DataFrame(test_pred4)
df_train= pd.concat([train_val1,train_val2,train_val3,train_val4],axis=1)
df_test = pd.concat([test_pred1, test_pred2,test_pred3,test_pred4], axis=1)
df_train= pd.concat([train_val2,train_val3,train_val4],axis=1)
df_test = pd.concat([test_pred2,test_pred3,test_pred4], axis=1)


test_1=pd.DataFrame((model1.predict_proba(test))[:,1])
test_2=pd.DataFrame((model2.predict_proba(test))[:,1])
test_3=pd.DataFrame((model3.predict_proba(test))[:,1])
test_4=pd.DataFrame((lgb.predict(test,num_iteration=lgb.best_iteration)))
df_finaltest = pd.concat([test_1, test_2,test_3,test_4], axis=1)

df_finaltest = pd.concat([test_2,test_3,test_4], axis=1)


grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression(random_state=0)
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(df_train,y_val)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
logreg2=LogisticRegression(C=0.1,penalty="l2")
logreg2.fit(df_test,y_test)
print("score",logreg2.score(df_test,y_test))

probs=logreg2.predict_proba(df_finaltest)
probs = probs[:, 1]

solution_template.hospital_death = probs
solution_template.to_csv("D:/MS DS WPI/Sem II/WiDS/submission.csv", index=0)
