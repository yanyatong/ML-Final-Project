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
                    'hour', #starttime
                    'weekend',
                    'holiday',
                  ]
                ]
   
    if training:
      data = self._fit_transform(data)
    else:
      data = self._transform(data)
    return data


def main():
  data = pd.read_csv('prepared_dataset_1211.csv')
  df_2 = pd.read_csv('station_status.csv', header = 0)
  featurizer = BikePredictionModel()
  stations = df_2['station_id']
  
  X = featurizer.create_features(data, training=True)
  clf = LogisticRegression(C=10)

  for station in stations:
    y = data[str(station)]
    clf.fit(X, y)
    print( "station id = " )
    print( station )
    print ("Cross validation, score =  ")
    print ( clf.score(X,y) )

if __name__ == '__main__':
  main()

