import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# 读入训练数据
df = pd.read_csv('train.csv')

# 数据预处理
X, y = df.values[:, 1:]/255.0, df.values[:, :1]
X_train, X_test, y_train, y_test = train_test_split(X, np.squzze(y))

# 模型参数选择
pipeline = Pipeline([('clf', SVC(C=1, gamma=0.01, cache_size=10000, verbose=True))])
parameters = {'clf__C': (1, 3, 5, 7, 9), 'clf__gamma': (0.01, 0.03, 0.05, 0.07, 0.09)}

# 训练模型
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 在训练数据集上查看模型精度
predicts = grid_search.predict(X_test)
print(classification_report(y_test, predicts))

# 读入测试数据，并输出预测结果至csv文件中
test = pd.read_csv('test.csv')
predicts = grid_search.predict(test.values/255.0)
predicts = np.asarray(predicts[:, np.newaxis])
df_predict = pd.DataFrame(data=predicts, index=range(1,28001))
df_predict.to_csv('submission_svm3.csv')
