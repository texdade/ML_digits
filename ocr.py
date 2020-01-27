try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import argmax
from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax, Flatten, MaxPool2D, Dropout
from datetime import datetime

tf.random.set_seed(0)

#
# CLASSES DEFINITION
#

class CharConvolutional(Model):
  def __init__(self, in_channels, out_channels, size):
    super().__init__() # setup the moedl basic functionalities (mandatory)
    initial = tf.random.truncated_normal([size, size, in_channels, out_channels], stddev=0.1)
    self.filters = tf.Variable(initial) # create weights for the filters

  def call(self, x):
    res = tf.nn.conv2d(x, self.filters, 1, padding="SAME")
    return res

class CharFullyConnected(Model):
  def __init__(self, input_shape, output_shape):
    super().__init__() # initialize the model
    self.W = tf.Variable(tf.random.truncated_normal([input_shape, output_shape], stddev=0.1)) # declare weights 
    self.b = tf.Variable(tf.constant(0.1, shape=[1, output_shape]))  # declare biases
    
  def call(self, x):
    res = tf.matmul(x, self.W) + self.b 
    return res

class CharDeepModel(Model):
  def __init__(self):
    super().__init__()                        
    self.conv1 = CharConvolutional(1, 16, 4) 
    self.pool1 = MaxPool2D([2,2])                
    self.conv2 = CharConvolutional(16, 32, 4) 
    self.pool2 = MaxPool2D([2,2])                
    self.conv3 = CharConvolutional(32, 64, 2)
    self.pool3 = MaxPool2D([2,2])                
    self.flatten = Flatten()                     
    self.fc1 = CharFullyConnected(2*1*64, 256)
    self.dropout = Dropout(0.5)
    self.fc2 = CharFullyConnected(256, 26)
    self.softmax = Softmax()

  def call(self, x, training=False):
    x = tf.nn.relu(self.conv1(x))
    x = self.pool1(x)
    x = tf.nn.relu(self.conv2(x))
    x = self.pool2(x)
    x = tf.nn.relu(self.conv3(x))
    x = self.pool3(x)

    x = self.flatten(x)
    x = tf.nn.relu(self.fc1(x))

    x = self.dropout(x, training=training) # behavior of dropout changes between train and test
    
    x = self.fc2(x)
    prob = self.softmax(x)
    
    return prob

#
# MODEL CREATION, TRAINING AND TEST
#

to_num = { 
           'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 
           'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v':21, 'w': 22, 'x': 23, 'y': 24,
           'z': 25
          }

#dictionary to map numbers back into chars
to_char = {}
for key in to_num:
  to_char[to_num[key]] = key

#map an array of chars into an array of numbers wrt to the dictionary above
def char_to_int(data):
  ints = []
  for el in data:
    ints.append( to_num[el] )
  
  return ints

def int_to_char(data):
  chars = []
  for el in data:
    chars.append( to_char[el] )
  
  return chars


def train_step(images, labels, model, loss_fn, optimizer):
  with tf.GradientTape() as tape: # all the operations within this scope will be recorded in tape
    predictions = model(images, training=True)
    loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss_metric(loss)
  train_accuracy_metric(labels, predictions)

def train_loop(epochs, train_ds, model, loss_fn, optimizer, validation):
  for epoch in range(epochs):
      # reset the metrics for the next epoch
    train_loss_metric.reset_states()
    train_accuracy_metric.reset_states()

    start = datetime.now() # save start time 
    for images, labels in train_ds:
      train_step(images, labels, model, loss_fn, optimizer)

    template = 'Epoch {}, Time {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch+1,
                          datetime.now() - start, 
                          train_loss_metric.result(),
                          train_accuracy_metric.result()*100))
    
    #if performance are really poor cut the validation with these settings
    if(train_accuracy_metric.result()<0.2 and validation):
      print("Poor performances - Validation aborted")
      return -1
      

def test_step(images, labels, model, loss_fn, validation):
  predictions = model(images, training=False)
  if(not validation): #if not in validation, print out predictions
    final_pred=np.array(int_to_char(tf.argmax(predictions, axis=1).numpy()))
    final_pred=final_pred.reshape(len(final_pred),1) #put them in a column
    final_pred=pd.DataFrame(final_pred)
    final_pred.to_csv("test-pred.txt", index=False, header=False)

  t_loss = loss_fn(labels, predictions)

  test_loss_metric(t_loss)
  test_accuracy_metric(labels, predictions)

def test_loop(test_ds, model, loss_fn, validation):
  # reset the metrics for the next epoch
  test_loss_metric.reset_states()
  test_accuracy_metric.reset_states()
 
  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels, model, loss_fn, validation)

  template = 'Test Loss: {}, Test Accuracy: {}'
  print(template.format(test_loss_metric.result(),
                        test_accuracy_metric.result()*100)) 
  return test_accuracy_metric.result()*100

url_train_data = "https://raw.githubusercontent.com/texdade/ML_digits/master/train-data.csv?token=ADSHSB7VA6724V6HFO52M6K6GKRLI"
url_train_target = "https://raw.githubusercontent.com/texdade/ML_digits/master/train-target.csv?token=ADSHSB57C3AH2GGPKDZ6QT26GKVTA"
url_test_data = "https://raw.githubusercontent.com/texdade/ML_digits/master/test-data.csv?token=ADSHSB2XCXVAXBYLXF2HJNS6GKVVQ"
url_test_target = "https://raw.githubusercontent.com/texdade/ML_digits/master/test-target.csv?token=ADSHSB2EJB4YSRYD55MLVL26GKVWS"

data_train = pd.read_csv(url_train_data, header=None).to_numpy()
target_train = pd.read_csv(url_train_target, header=None).to_numpy()
data_test = pd.read_csv(url_test_data, header=None).to_numpy()
target_test = pd.read_csv(url_test_target, header=None).to_numpy()

data_train = data_train.reshape(data_train.shape[0], 16, 8)
data_test = data_test.reshape(data_test.shape[0], 16, 8)

x_train = tf.cast(data_train, tf.float32) / 255.0
x_test = tf.cast(data_test, tf.float32) / 255.0

x_train = tf.convert_to_tensor(x_train[..., tf.newaxis])
x_test = tf.convert_to_tensor(x_test[..., tf.newaxis])

y_train = tf.one_hot(char_to_int(target_train[:,0]), 26)
y_test = tf.one_hot(char_to_int(target_test[:,0]), 26)

train_loss_metric = tf.keras.metrics.Mean()
train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
test_loss_metric = tf.keras.metrics.Mean()
test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(y_test.shape[0])

#MODEL SELECTION + VALIDATION
EPOCHS=16
learning_rates=[1e-1, 1e-2, 1e-3, 1e-4]
optimizers = [0,1,2]
best_learning_rate=1e-1
best_optimizer = 0
best_acc=0.0

train_ds_validation = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)
test_ds_validation = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)

for rate in learning_rates:
  for optimizer in optimizers:
    network = CharDeepModel()

    network_loss = tf.keras.losses.CategoricalCrossentropy()
    if(optimizer==0):
      print(("\nModel validating with optimizer Adam at learning rate = {}").format(rate))
      network_optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
    elif(optimizer==1):
      print(("\nModel validating with optimizer Adamax at learning rate = {}").format(rate))
      network_optimizer = tf.keras.optimizers.Adamax(learning_rate=rate)
    elif(optimizer==2):
      print(("\nModel validating with optimizer Nadam at learning rate = {}").format(rate))
      network_optimizer = tf.keras.optimizers.Nadam(learning_rate=rate)
    
    completed = train_loop(EPOCHS, train_ds_validation, network, network_loss, network_optimizer, True)
    if(completed != -1):
      acc=test_loop(test_ds_validation, network, network_loss, True)
    if(acc>best_acc):
      best_acc=acc
      best_learning_rate=rate
      best_optimizer=optimizer

# Create an instance of the model
network = CharDeepModel()

network_loss = tf.keras.losses.CategoricalCrossentropy()
if(best_optimizer==0):
  print(("\nModel training with optimizer Adam at learning rate = {}").format(best_learning_rate))
  network_optimizer = tf.keras.optimizers.Adam(learning_rate=best_learning_rate)
elif(best_optimizer==1):
  print(("\nModel training with optimizer Adamax at learning rate = {}").format(best_learning_rate))
  network_optimizer = tf.keras.optimizers.Adamax(learning_rate=best_learning_rate)
elif(best_optimizer==2):
  print(("\nModel training with optimizer Nadam at learning rate = {}").format(best_learning_rate))
  network_optimizer = tf.keras.optimizers.Nadam(learning_rate=best_learning_rate)

#FULL TRAINING
EPOCHS = 32
train_loop(EPOCHS, train_ds, network, network_loss, network_optimizer, False)

#TESTING OF THE FINAL NETWORK
test_loop(test_ds, network, network_loss, False)



