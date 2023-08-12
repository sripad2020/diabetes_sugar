import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from keras.models import  Sequential
from keras.layers import Dense
import keras.activations,keras.optimizers,keras.losses,keras.metrics
data=pd.read_csv('SugarPrediction.csv')
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())
for i in data.select_dtypes(include='number').columns.values:
    sn.boxplot(data[i])
    plt.show()

for i in data.select_dtypes(include='number').columns.values:
      for j in data.select_dtypes(include='number').columns.values:
        plt.plot(data[i],marker='o',label=f"{i}",color='red')
        plt.plot(data[j],marker='x',label=f"{j}",color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        plt.scatter(data[i],data[j],marker='o',color='red')
        plt.scatter(data[i],data[j],marker='x',color='blue')
        plt.xlabel(f'{i}')
        plt.ylabel(f'{j}')
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='red')
        sn.histplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.distplot(data[i], label=f"{i}", color='red')
        sn.distplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='object').columns.values:
    if (len(data[i].value_counts())) <=5 :
        sn.countplot(data[i])
        plt.show()

sn.pairplot(data)
plt.show()

lab=LabelEncoder()
data['gender']=lab.fit_transform(data['Gender'])
data['genetic']=lab.fit_transform(data['Genetic'])
data['life_style']=lab.fit_transform(data['Lifestyle'])
data['area']=lab.fit_transform(data['Area'])
data['stress']=lab.fit_transform(data['Stress'])
data['diabetic']=lab.fit_transform(data['Diabetic'])

plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

x=data[['Age','Eduration','BMI','FBS','PPBS','gender','genetic','life_style','stress']]
y=data['diabetic']

x_train,x_test,y_train,y_test=train_test_split(x,y)
print(y_train)

lr=LogisticRegression(max_iter=200)
lr.fit(x_train,y_train)
print('The logistic regression: ',lr.score(x_test,y_test))

xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print("the Xgb : ",xgb.score(x_test,y_test))

lgb=LGBMClassifier()
lgb.fit(x_train,y_train)
print('The LGB',lgb.score(x_test,y_test))

tree=DecisionTreeClassifier(criterion='entropy',max_depth=1)
tree.fit(x_train,y_train)
print('Dtree ',tree.score(x_test,y_test))

rforest=RandomForestClassifier(criterion='entropy')
rforest.fit(x_train,y_train)
print('The random forest: ',rforest.score(x_test,y_test))

adb=AdaBoostClassifier()
adb.fit(x_train,y_train)
print('the adb ',adb.score(x_test,y_test))

grb=GradientBoostingClassifier()
grb.fit(x_train,y_train)
print('Gradient boosting ',grb.score(x_test,y_test))

bag=BaggingClassifier()
bag.fit(x_train,y_train)
print('Bagging',bag.score(x_test,y_test))


X=data[['stress','Eduration','BMI','FBS','PPBS','gender','genetic','diabetic','area','Age']]
Y=pd.get_dummies(data['life_style'])
x_trin,x_tst,y_trin,y_tst=train_test_split(X,Y)

models=Sequential()
models.add(Dense(units=X.shape[1],input_dim=X.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=X.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=X.shape[1],activation=keras.activations.relu))
models.add(Dense(units=X.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=X.shape[1],activation=keras.activations.relu))
models.add(Dense(units=Y.shape[1],activation=keras.activations.softmax))
models.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics='accuracy')
models.fit(x_trin,y_trin,batch_size=20,epochs=300)