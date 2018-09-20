import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import os


train = pd.read_csv('J:/Code/kaggle/SanFranciscoCrimeClassification/train.csv', parse_dates = ['Dates']) # parse_dates将指定列解析为日期格式
test = pd.read_csv('J:/Code/kaggle/SanFranciscoCrimeClassification/test.csv', parse_dates = ['Dates'])

# LabelEncoder对不同的犯罪类型编号
lebelCrime = preprocessing.LabelEncoder()
crime = lebelCrime.fit_transform(train.Category)
# print(np.max(crime))
# print(np.min(crime))
# os.system("pause")

# get_dummies因子化星期几，街区等特征
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = pd.get_dummies(train.Dates.dt.hour)

# 组合特征
trainData = pd.concat([hour, days, district], axis = 1)
trainData['crime'] = crime
print(trainData.iloc[4])

# testData is the same as before
days_test = pd.get_dummies(test.DayOfWeek)
district_test = pd.get_dummies(test.PdDistrict)
hour_test = pd.get_dummies(test.Dates.dt.hour)

testData = pd.concat([hour_test, days_test, district_test], axis = 1)
print(testData.iloc[4])
