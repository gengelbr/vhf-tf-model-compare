import tensorflow as tf
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
training_set = tf.contrib.learn.datasets.base.load_csv( has_header=False,
                                                        filename=TRAINING,
                                                        target_column=0,
                                                        target_dtype=np.float64)

test_set = tf.contrib.learn.datasets.base.load_csv( has_header=False,
                                                    filename=TEST,
                                                    target_column=0,
                                                    target_dtype=np.float64)

regressor = tf.contrib.learn.DNNRegressor([512,256,128,64])

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
