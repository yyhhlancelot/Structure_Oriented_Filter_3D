from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
from data_analysis import trainData, testData
import os

# 只取星期几和街区作为分类器输入特征
features = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

hour_features = [x for x in range(0, 24)]

print(hour_features)

features = features + hour_features

print(features)

# 从训练集.6中分割出验证集.4
training, validation = train_test_split(trainData, train_size = .60)

# 朴素贝叶斯建模,计算log_loss
model_B = BernoulliNB()
nbStart = time.time()
model_B.fit(training[features], training['crime'])
nbCostTime = time.time() - nbStart
predicted_B = np.array(model_B.predict_proba(validation[features])) # 算出每个特征相对应的概率

print(predicted_B)
print(validation['crime'])
print('朴素贝叶斯建模时间：%f s' % nbCostTime)
print('朴素贝叶斯log损失 : %f ' % log_loss(validation['crime'], predicted_B)) # log损失
os.system("pause")

# 逻辑回归建模, 计算log_loss
# model_L = LogisticRegression()
# lrStart = time.time()
# model_L.fit(training[features], training['crime'])
# lrCostTime = time.time() - lrStart
# predicted_L = np.array(model_L.predict_proba(validation[features]))
# print('逻辑回归建模时间：%f s' % lrCostTime)
# print('逻辑回归log损失 : %f ' % log_loss(validation['crime'], predicted_L))
