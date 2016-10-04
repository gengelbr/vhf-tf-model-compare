import tensorflow as tf
import tensorflow.contrib.learn.python.learn as learn
import numpy as np
import sys

def writeresults(preds,regressor,fname):
  results = regressor.predict(preds)
  f = open(fname,'w+')
  for x in results:
    f.write(str(x)+'\n')
  f.close()

# Data sets.
TRAINING = "s1_out.csv"
TEST = "s2_out.csv"

#get numpy ready datasets
s1_dataset = np.genfromtxt('s1_out.csv', delimiter=',')
s2_dataset = np.genfromtxt('s2_out.csv', delimiter=',')

# Take our the first column (the measured result) and concatinate i
# into one array. This makes an array ready to feed the
# predictor when trained.
dnn_predictors = np.vstack((s1_dataset[:,1:],s2_dataset[:,1:]))

# Load datasets.
training_set = learn.datasets.base.load_csv_without_header(
                                                        filename=TRAINING,
                                                        features_dtype=np.float32,
                                                        target_column=0,
                                                        target_dtype=np.float32)

test_set = learn.datasets.base.load_csv_without_header(
                                                    filename=TEST,
                                                    features_dtype=np.float32,
                                                    target_column=0,
                                                    target_dtype=np.float32)

feature_columns = learn.infer_real_valued_columns_from_input(training_set.data)

regressor = learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[512,256,128,64])

lowest_loss = sys.float_info.max
for x in range(100000):
  regressor.fit(x=training_set.data,y=training_set.target,steps=100)
  loss = regressor.evaluate(x=test_set.data,y=test_set.target)['loss']
  if lowest_loss > loss:
    lowest_loss = loss
    lowest_dnn_weights = regressor.dnn_weights_
    lowest_dnn_bias = regressor.dnn_bias_
    np.save(open('lowest_dnn_weights.np','wb'),lowest_dnn_weights)
    np.save(open('lowest_dnn_bias.np','wb'),lowest_dnn_bias)

  print('Loss: {0:f}'.format(loss))
