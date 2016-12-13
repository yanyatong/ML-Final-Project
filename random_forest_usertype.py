from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

class BikePredictionModel():
  def __init__(self):
    vectorizer = None
    self._imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    self._binarizer = preprocessing.Binarizer()
    self._scaler = preprocessing.MinMaxScaler()
    self._preprocs = [self._imputer, \
                      self._binarizer, \
                      self._scaler
                      ]

  def _fit_transform(self, dataset):
    for p in self._preprocs:
      dataset = self._proc_fit_transform(p, dataset)
    return dataset

  def _transform(self, dataset):
    for p in self._preprocs:
      dataset = p.transform(dataset)

    return dataset

  def _proc_fit_transform(self, p, dataset):
    p.fit(dataset)
    dataset = p.transform(dataset)
    return dataset

  def create_features(self, dataset, training=False):
    
   
    data = dataset[ [
                  'positionX',
                  'positionY'
                  ]
                ]
   
    if training:
      data = self._fit_transform(data)
    else:
      data = self._transform(data)
    return data


def main():
  data = pd.read_csv('prepared_dataset.csv', nrows = 3000)
  featurizer = BikePredictionModel()

  print ("Transforming dataset into features...")
  X = featurizer.create_features(data, training=True)
  y = data.age

  clf = RandomForestClassifier(n_estimators=8,criterion="gini")
  # clf = LogisticRegression(C=3)
  clf.fit(X, y)

  # predict ages 
  print( "model used : Random Forest" )
  print( clf.score(X,y) )
  print( "station id => 146, 347 ,402, 499 , 237" )
  print( "customer age :" )
  print(  clf.predict([[-11.475,-3.35],[10.26,-15.41],[-6.621,-10.55],[7.89,-11.14],[18,-13]]) )
  # predict gender
  y = data.gender
  clf.fit(X,y)
  print(  "customer gender : ( male => 1, female => 2)" )
  print(  clf.predict([[-11.475,-3.35],[10.26,-15.41],[-6.621,-10.55],[7.89,-11.14],[18,-13]]) )


if __name__ == '__main__':
  main()



