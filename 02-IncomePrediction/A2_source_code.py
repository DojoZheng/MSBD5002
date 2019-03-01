'''
==============================
Name:   ZHENG Dongjia
SID:    20546139
Date:   2018/10/25
==============================
'''


'''
1. Import the training data & test data
'''
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# import Training Data & Testing Data
train_features = pd.read_csv("trainFeatures.csv")
train_labels = pd.read_csv("trainLabels.csv", header=None, names=["label"])
test_features = pd.read_csv("testFeatures.csv")

'''
2. Data Preprocessing
'''
# 2.1 Combine the training datasets and testing datasets
combined_features = pd.concat([train_features, test_features], axis=0)

# 2.2 Apply Logarithmic Transformation on the training features & testing features
skewed = ['capital-gain', 'capital-loss']
features_log = pd.DataFrame(data = combined_features)
features_log[skewed] = combined_features[skewed].apply(lambda x: np.log(x + 1))

# 2.3 Normalizing Numerical Features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt']

features_log_minmax = pd.DataFrame(data = features_log)
features_log_minmax[numerical] = scaler.fit_transform(features_log[numerical])

# 2.4 One-Hot Encoding
features_encoded = pd.get_dummies(features_log_minmax)

# 2.5 Shuffle & Split Data
train_features_pre = features_encoded[:train_features.shape[0]]
test_features_pre = features_encoded[train_features.shape[0]:]


'''
3. Model Training & Prediction
'''
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=0.8, gamma=0.3, learning_rate=0.1,
                    max_delta_step=0, max_depth=2, min_child_weight=2,
                    missing=None, n_estimators=1000, n_jobs=1, nthread=4,
                    objective='binary:logistic', random_state=0, reg_alpha=0,
                    reg_lambda=1, scale_pos_weight=1, seed=27, silent=True,
                    subsample=0.8)

# fit the training data
xgb = xgb.fit(train_features_pre, train_labels.values.ravel())

# predict the testing data
test_pred = xgb.predict(test_features_pre)
labels_pred_df = pd.DataFrame(data=test_pred)

# write to "A2_dzhengah_20546139_prediction.csv"
labels_pred_df.to_csv("A2_dzhengah_20546139_prediction.csv", index=None, header=None)