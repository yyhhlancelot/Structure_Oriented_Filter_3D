import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import os
from baseline import model_B, features
from data_analysis import testData, test

# from sklearn.naive_bayes import BernoulliNB
templet = pd.read_csv('J:/Code/kaggle/SanFranciscoCrimeClassification/sampleSubmission.csv')

templet_column = templet.columns.values.tolist()

prediction_test = model_B.predict_proba(testData[features])

prediction_test_df = pd.DataFrame(prediction_test, columns = templet_column[1:])

result = pd.DataFrame({'Id' : test['Id'].as_matrix()})

result = pd.concat([result, prediction_test_df], axis = 1)

print(result)
result.to_csv("J:/Code/kaggle/SanFranciscoCrimeClassification/BernoulliNB_predictions.csv", index = False)