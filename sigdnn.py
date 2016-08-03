import tensorflow as tf
import numpy as np

# Data sets.
TRAINING = "d1.csv"
TEST = "t1.csv"

# Overloaded print monitor.
class PrintLossEveryN(tf.contrib.learn.monitors.EveryN):
  def every_n_step_end(self,step,outputs):
    self._estimator.evaluate(x=test_set.data,y=test_set.target)

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv( has_header=False,
                                                        filename=TRAINING,
                                                        target_column=8,
                                                        target_dtype=np.float64)
test_set = tf.contrib.learn.datasets.base.load_csv( has_header=False,
                                                    filename=TEST,
                                                    target_column=8,
                                                    target_dtype=np.float64)

regressor = tf.contrib.learn.LinearRegressor()

for x in range(100):
  regressor.fit(x=training_set.data,y=training_set.target,steps=10000)
  loss = regressor.evaluate(x=test_set.data,y=test_set.target)['loss']
  bias = regressor.linear_bias_
  weights = regressor.linear_weights_
  print('Loss: {0:f}'.format(loss))
  print('Bias: ',bias)
  print('Weights: ',weights)

# Classify two new flower samples.
# new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# y = regressor.predict(new_samples)
# print('Predictions: {}'.format(str(y)))
