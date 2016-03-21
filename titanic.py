"""
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)
"""

import pandas as pd
import numpy as np
import re
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 读取数据并合并
train = pd.read_csv('train.csv', index_col='PassengerId')
test = pd.read_csv('test.csv', index_col='PassengerId')
y = train['Survived'].copy()
train.drop('Survived', axis=1, inplace=True)
combi = pd.concat([train, test])

# 数据预处理
# 将Ticket中数字序号的第一个数字作为新的特征并分解为dummy，但对最后的模型准确率作用不大，所以省去此尝试，
# 可考虑对Ticket的前两位字符作提取为新的特征，利用counts learning的方式进行特征提取(未尝试）
#ticket = combi['Ticket'].values
#combi['Ticket_'] = np.array(["".join(re.findall(r'\d{2,}', item)) or np.nan for item in ticket])
#combi['Ticket_'] = combi['Ticket_'].astype(np.object)
#combi['Ticket_'][combi['Ticket_']=='nan'] = np.nan
#combi['Ticket_'].fillna(method='bfill', inplace=True)
#combi['Ticket_'].fillna(method='ffill', inplace=True)
#combi['Ticket_'] = combi['Ticket_'].str[0]
# 去掉Ticket列，由于Cabin中包含大量缺失值，而且与类别的共信息不大，也将其去除
combi.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
# Embarked列中众数为'S', 将缺失值填为'S'
combi['Embarked'].fillna('S', inplace=True)
# 将Fare中的缺失值填为其平均值
combi['Fare'].fillna(combi['Fare'].mean(), inplace=True)
# 由于Age中存在大量缺失值，而且此特征比较重要，不能直接抛弃，利用其它特征进行回归填值，
# 过程中包括对Sex和Embarked列进行label encoding过程
combi_age_not_null = combi[pd.notnull(combi['Age'])]
svr = SVR() #使用回归预测age中大量的缺失值
lbe = LabelEncoder()
combi['Sex'] = lbe.fit_transform(combi['Sex'])
combi['Embarked'] = lbe.fit_transform(combi['Embarked'])
combi_age_not_null = combi[pd.notnull(combi['Age'])]
svr.fit(combi_age_not_null.drop(['Name', 'Age', 'Ticket_'], axis=1).values, combi_age_not_null['Age'])
combi_age_null = combi[pd.isnull(combi['Age'])]
age_isnull_pred = svr.predict(combi_age_null.drop(['Name', 'Age', 'Ticket_'], axis=1))
# 添加Child列，将小于18岁的乘客标为1， 其它为0
combi['Age'][pd.isnull(combi['Age'])] = age_isnull_pred.copy()
combi['Child'] = 0
combi['Child'][combi['Age'] <= 18] = 1
# 添加familysize列， 计算乘客在船上的亲人总数
combi['FamilySize'] = combi['SibSp'] + combi['Parch']
combi['FamilySize'][combi['FamilySize'] == 0] = 0
combi['FamilySize'][combi['FamilySize'].isin([1,2,3])] = 1
combi['FamilySize'][combi['FamilySize'] > 3] = 0
# 对乘客名字中的称谓进行提取，并对相似称谓进行统一
names = combi['Name'].str.split(r'[.,]')
combi['Title'] = [name[1] for name in names]
combi['Title'][combi['Title'].isin(['Mme', 'Mlle'])] = 'Mlle'
combi['Title'][combi['Title'].isin(['Capt', 'Don', 'Major', 'Sir'])] = 'Sir'
combi['Title'][combi['Title'].isin(['Dona', 'Lady', 'the Countess', 'Jonkheer'])] = 'Lady'
# 对票价进行分类，大于30的标为1， 其余标为0
combi['Fare'][combi['Fare']<=30] = 0
combi['Fare'][combi['Fare']>30] = 1
#combi['Ticket_'] = combi['Ticket_'].astype(np.int)	
#combi['Ticket_cat'] = 0
#combi['Ticket_cat'][combi['Ticket_'].isin([1,2])] = 1
# 对阶层进行标记
combi['Pclass'][combi['Pclass'].isin([1,2])] = 1
combi['Pclass'][combi['Pclass']==3] = 0
# 去除不需要的特征
combi.drop(['Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Ticket_', 'Ticket_cat'], axis=1, inplace=True)


# 还原处理好的训练数据与测试数据
train = combi.values[:891, :]
test = combi.values[ 891:, :]
#rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5)
#X_train, X_test, y_train, y_test = train_test_split(train, y)
#combi2_ohe = OneHotEncoder(categorical_features=[0,-1])
#data = combi2_ohe.fit_transform(combi2.values).toarray()

# 使用XGBOOST训练模型
X_train, X_test, y_train, y_test = train_test_split(train, y)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {'max_depth':6, 'silent':1, 'eta':0.1, 'objective':'binary:logistic', 'sub_sample':0.9}
watchlist = [(dtrain, 'train'), (dtest, 'eval')]
bst = xgb.train(params, dtrain, 70, watchlist)

# 对测试数据进行预测并导出到csv文件
train_xgb = xgb.DMatrix(train, label=y)
test_xgb = xgb.DMatrix(test)
bst = xgb.train(params, train_xgb, 70, watchlist)
predicts = bst.predict(test_xgb)
results = np.array(predicts > 0.5, dtype=np.int)
results = np.asarray(results[:, np.newaxis])
df_results = pd.DataFrame(results, index=range(892, 1310))
df_results.to_csv('submission_xgb.csv')

