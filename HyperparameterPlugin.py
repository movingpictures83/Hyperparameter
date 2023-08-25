#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[328]:


################################################ Preprocessing #########################################################
class HyperparameterPlugin:
 def input(self, inputfile):
  self.data_path = inputfile#"MASH_combined.csv"

 def run(self):
     pass

 def output(self, outputfile):
  data_df = pd.read_csv(self.data_path)

  # # Tramsform categorical data to categorical format:
  # for category in categorical_cols:
  #     data_df[category] = data_df[category].astype('category')
  #

  # Clean numbers:
  #"Cocain_Use": {"yes":1, "no":0},
  cleanup_nums = { "Cocain_Use": {"yes":1, "no":0},
                 "race": {"White":1, "Black":0, "BlackIsraelite":0, "Latina":1},
  }

  data_df.replace(cleanup_nums, inplace=True)

  # Drop id column:
  data_df = data_df.drop(["pilotpid"], axis=1)

  # remove NaN:
  data_df = data_df.fillna(0)

  # Standartize variables
  from sklearn import preprocessing
  names = data_df.columns
  scaler = preprocessing.StandardScaler()
  data_df_scaled = scaler.fit_transform(data_df)
  data_df_scaled = pd.DataFrame(data_df_scaled, columns=names)


  # In[303]:


  ################################################ Users vs Non-Users #########################################################

  # Random Forest
  # Benchmark

  y_col = "Cocain_Use"
  test_size = 0.3
  validate = True

  y = data_df[y_col]

  X = data_df_scaled.drop([y_col], axis=1)

  # Create random variable for benchmarking
  X["random"] = np.random.random(size= len(X))

  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = test_size, random_state = 2)

  rf = RandomForestClassifier(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)

  rf.fit(X_train, y_train)

  print('Training accuracy: {:.2f} \nOOB Score: {:.2f} \nTest Accuracy: {:.2f}'.format(rf.score(X_train, y_train),
                                                                                             rf.oob_score_,
                                                                                             rf.score(X_valid, y_valid)))
  # scores = cross_val_score(rf, X, y, cv=5)
  # print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
  # a = rf.predict(X_valid)

  # importances_df = drop_col_feat_imp(rf, X, y, X_valid, y_valid)
  # importances_df.to_csv("/Users/stebliankin/Desktop/SabrinaProject/FeatureSelection/importance_df.csv")

  scores = cross_val_score(rf, X, y, cv=5)
  print("CV Accuracy: %0.2f " % (scores.mean()))


  # In[122]:





  # In[305]:


  # Lasso Feature selection

  from sklearn.linear_model import LassoCV
  from sklearn.feature_selection import SelectFromModel
  from sklearn.linear_model import LogisticRegression

  # clf = LassoCV(cv=5)


  sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l2', random_state=41)) # Lasso L1 penalty
  sel_.fit(X_train, y_train)

  selected_feat = list(X_train.columns[(sel_.get_support())])
  print(selected_feat)

  X_lasso = X[selected_feat]


  # In[233]:


  # Retrain the model with random forest with selected features
  X_train_lasso, X_valid_lasso, y_train_lasso, y_valid_lasso = train_test_split(X_lasso, y, test_size = test_size, random_state = 42)

  rf = RandomForestClassifier(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)

  rf.fit(X_train_lasso, y_train_lasso)

  print('Training accuracy: {:.2f} \nOOB Score: {:.2f} \nTest Accuracy: {:.2f}'.format(rf.score(X_train_lasso, y_train_lasso),
                                                                                             rf.oob_score_,
                                                                                             rf.score(X_valid_lasso, y_valid_lasso)))
  # scores = cross_val_score(rf, X, y, cv=5)
  # print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
  # a = rf.predict(X_valid)

  # importances_df = drop_col_feat_imp(rf, X, y, X_valid, y_valid)
  # importances_df.to_csv("/Users/stebliankin/Desktop/SabrinaProject/FeatureSelection/importance_df.csv")
  # if validate:
  #     scores = cross_val_score(rf, X, y, cv=25)
  #     print("CV Accuracy: %0.2f " % (scores.mean()))


  # In[234]:


  # Try Hyperparameter tuning to reduce overfitting
  validate=True

  from sklearn.model_selection import RandomizedSearchCV

  # Step 1 - define hyperparameters

  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  # Function to measure quality of the split:
  criterion = ["gini", "entropy"]

  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

  print(random_grid)


  # Step 2 - Random Search traning

  # Use the random grid to search for best hyperparameters
  # First create the base model to tune
  rf = RandomForestClassifier()
  # Random search of parameters, using 3 fold cross validation, 
  # search across 100 different combinations, and use all available cores
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
  rf_random.fit(X_train_lasso, y_train_lasso)

  # print('Training accuracy: {:.2f} \nOOB Score: {:.2f} \nTest Accuracy: {:.2f}'.format(rf_random.score(X_train, y_train),
  #                                                                                              rf.oob_score_,
  #                                                                                              rf.score(X_valid, y_valid)))

  # if validate:
  #     scores = cross_val_score(rf_random, X, y, cv=5)
  #     print("CV Accuracy: %0.2f " % (scores.mean()))


  # In[70]:


  rf_random.best_params_


  # In[237]:


  # Retrain based on optimal hyperparameters

  # Benchmark from the regular model
  validate=True
  base_model = RandomForestClassifier(n_estimators = 100,
                               n_jobs = -1,
                               bootstrap = True,
                               random_state = 42)
  base_model.fit(X_train_lasso, y_train_lasso)
  print("Accuracy for the base model")
  print('Training accuracy: {:.2f}  \nTest Accuracy: {:.2f}'.format(base_model.score(X_train_lasso, y_train_lasso),
                                                                                              base_model.score(X_valid_lasso, y_valid_lasso)))
  # if validate:
  #     scores = cross_val_score(base_model, X, y, cv=5)
  #     print("CV Accuracy: %0.2f " % (scores.mean()))
  print()
  print("Accuracy for the hyperparameter tunned")


  # base_model = RandomForestClassifier(n_estimators = 100,
  #                                 max_depth=1,
  #                                n_jobs = -1,
  #                                bootstrap = True,
  #                                random_state = 42)
  best_random = rf_random.best_estimator_
  best_random.fit(X_train_lasso, y_train_lasso)


  print('Training accuracy: {:.2f} \nTest Accuracy: {:.2f}'.format(best_random.score(X_train_lasso, y_train_lasso),
                                                                                             best_random.score(X_valid_lasso, y_valid_lasso)))

  if validate:
    scores = cross_val_score(best_random, X, y, cv=5)
    print("CV Accuracy: %0.2f " % (scores.mean()))


  # In[405]:


