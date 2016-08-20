import tensorflow as tf
import numpy as np
import sys

# Data sets.
TRAINING = "d1.csv"
TEST = "t1.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv( has_header=False,
                                                        filename=TRAINING,
                                                        target_column=8,
                                                        target_dtype=np.float64)
test_set = tf.contrib.learn.datasets.base.load_csv( has_header=False,
                                                    filename=TEST,
                                                    target_column=8,
                                                    target_dtype=np.float64)

regressor = tf.contrib.learn.DNNRegressor([512,256,128,64])

lowest_loss = sys.float_info.max

for x in range(100000):
  regressor.fit(x=training_set.data,y=training_set.target,steps=1000)
  loss = regressor.evaluate(x=test_set.data,y=test_set.target)['loss']
  if lowest_loss > loss:
    lowest_loss = loss
    lowest_dnn_w = regressor.dnn_weights_
    lowest_dnn_b = regressor.dnn_bias_
  #bias = regressor.linear_bias_
  #weights = regressor.linear_weights_
  print('Loss: {0:f}'.format(loss))
  #print('Bias: ',bias)
  #print('Weights: ',weights)

# Classify two new flower samples.
# new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# y = regressor.predict(new_samples)
# print('Predictions: {}'.format(str(y)))
