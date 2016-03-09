import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# 从Dates字串中提供日期相关特征
def extract_time(dates):
	items = datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
	year = items.year
	month = items.month
	day = items.day
	hour = items.hour

	return  month, day, hour

# 数据处理函数
def process_data(trainDF, testDF):
	trainDF.drop(['Descript', 'Resolution'], axis=1, inplace=True)
	testDF.drop(['Id'], axis=1, inplace=True)
	labels = trainDF['Category'].copy()
	y = trainDF['Category'].copy()	
	combi = pd.concat([trainDF.drop(['Category'], axis=1),  testDF])
	
	combi['Month'], combi['Day'], combi['Hour'] = zip(*combi['Dates'].apply(extract_time))
	combi.drop(['Dates'], axis=1, inplace=True)
	combi['intesect'] = combi['Address'].apply(lambda x: 1 if '/' in x else 0)
	combi['Wake'] = combi['Hour'].apply(lambda x: 1 if (int(x)>=8 and int(x)<=23) else 0)
	addresses = sorted(combi['Address'].unique())
	categories = sorted(trainDF['Category'].unique())
	addr_counts = combi.groupby('Address').size()
	cat_counts = trainDF.groupby('Category').size()
	addr_cat_counts = trainDF.groupby(['Address', 'Category']).size()
	# 使用counts learning方法对地址信息和分类结果进行特征提取， 可参考https://msdn.microsoft.com/en-us/library/azure/dn913056.aspx
	logoddsPA = {}
	logodds = {}
	PA = cat_counts/float(len(trainDF))
	default_logodds = np.log(PA/(1-PA))
	for addr in addresses:
		PA = addr_counts[addr]/float(len(combi))
		logoddsPA[addr] = np.log(PA/(1.0-PA))
		logodds[addr] = deepcopy(default_logodds)
		if addr in addr_cat_counts.keys():
			for cat in addr_cat_counts[addr].keys():
				if addr_cat_counts[addr][cat] >= 2 and addr_cat_counts[addr][cat] < addr_counts[addr]:
					PA = addr_cat_counts[addr][cat] / float(addr_counts[addr])
					logodds[addr][categories.index(cat)] = np.log(PA/(1.0-PA))
		logodds[addr] = pd.Series(logodds[addr])
		logodds[addr].index = range(len(categories))
	combi['LogoddsPA'] = combi['Address'].apply(lambda x: logoddsPA[x])
	logodds_features = combi['Address'].apply(lambda x: logodds[x])
	logodds_features.colums = ["logodds"+str(x) for x in range(len(categories))]
	combi_full = pd.concat([combi, logodds_features], axis=1)
	xy_scaler = StandardScaler()
	combi_full[['X', 'Y']] = xy_scaler.fit_transform(combi_full[['X', 'Y']])
	lbe = LabelEncoder()
	combi_full['DayOfWeek'] = lbe.fit_transform(combi_full['DayOfWeek'])
	combi_full['PdDistrict'] = lbe.fit_transform(combi_full['PdDistrict'])
	combi_full['intesect'] = combi_full['Address'].apply(lambda x: 1 if '/' in x else 0)
	combi_full['Wake'] = combi_full['Hour'].apply(lambda x: 1 if (int(x)>=8 and int(x)<=23) else 0)
	combi_full["IsDup"]=pd.Series(combi_full.duplicated()|combi_full.duplicated(take_last=True)).apply(int)
	combi_full.drop(['Address'], axis=1, inplace=True)
	y = lbe.fit_transform(y)
	ohe = OneHotEncoder(categorical_features=[0, 1,4,5,6])
	data = ohe.fit_transform(combi_full.values)
	train = combi_full.values[:878049, :]
	test = combi_full.values[878049:, :]

	return train, test, y, lbe.classes_


# 数据读取与预处理
trainDF = pd.read_csv('train.csv')
testDF = pd.read_csv('test.csv')
train, test, y, classes= process_data(trainDF, testDF)
X_train, X_test, y_train, y_test = train_test_split(train, y)

# 使用XGBOOST进行模型训练
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {'max_depth':5, 'silent':1, 'eta':0.4, 'objective':'multi:softprob', 'sub_sample':0.9, 'num_class':39, 'eval_metric':'mlogloss'}
watchlist = [(dtrain, 'train'), (dtest, 'val')]
bst = xgb.train(params, dtrain, 50, watchlist)
xgb_train = xgb.DMatrix(train, label=y)
xgb_test = xgb.DMatrix(test)
bst = xgb.train(params, xgb_train, 50)

# 对测试数据进行预测
predicts = bst.predict(xgb_test)
df_results = pd.DataFrame(predicts, index=range(len(test)), columns=classes)
df_results.to_csv('submission_xgb.csv')


