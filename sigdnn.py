import tensorflow as tf
import tensorflow.contrib.learn.python.learn as learn
from tensorflow.python.platform import gfile
import numpy as np
import collections
import sys
import csv

Dataset = collections.namedtuple('Dataset', ['data', 'target'])

def writeresults(preds,regressor,fname):
  results = regressor.predict(preds)
  f = open(fname,'w+')
  for x in results:
    f.write(str(x)+'\n')
  f.close()

def load_csv(filename,target_dtype,features_dtype,target_column=-1):
  """Load dataset from CSV file without a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    for row in data_file:
      target.append(row.pop(target_column))
      data.append(np.asarray(row, dtype=features_dtype))
    target = np.array(target, dtype=target_dtype)
    data = np.array(data)
  return Dataset(data=data, target=target)

def splitset(tuple_set):
  pass

#get numpy ready datasets
DATASET = sys.argv[1]
np_dataset = np.genfromtxt(DATASET, delimiter=',')

# Take our the first column (the measured result) and concatinate i
# into one array. This makes an array ready to feed the
# predictor when trained.
np_predictors = np_dataset[:,1:]

# Load datasets.
test_set = load_csv(
                   filename=DATASET,
                   features_dtype=np.float32,
                   target_column=0,
                   target_dtype=np.float32)

feature_columns = learn.infer_real_valued_columns_from_input(test_set.data)

regressor = learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[512,256,128,64])

lowest_loss = sys.float_info.max
for x in range(100000):
  regressor.fit(x=test_set.data,y=test_set.target,steps=10000)
  loss = regressor.evaluate(x=test_set.data,y=test_set.target)['loss']
  if lowest_loss > loss:
    lowest_loss = loss
    lowest_dnn_weights = regressor.dnn_weights_
    lowest_dnn_bias = regressor.dnn_bias_
    np.save(open('lowest_dnn_weights.np','wb'),lowest_dnn_weights)
    np.save(open('lowest_dnn_bias.np','wb'),lowest_dnn_bias)

  print('Loss: {0:f}'.format(loss))
