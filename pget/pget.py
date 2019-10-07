import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

from ml_utils.keras import get_states, set_states #for RNN compatibility
from ml_utils.model_builders import dense_stack

def create_traces(model):
  traces = []

  for var in model.trainable_variables:
    trace = tf.Variable(np.zeros(var.shape, dtype="float32"), trainable=False)
    traces.append(trace)

  return traces

#here, y_true should be some action that we took instead of y_pred (ie, from exploration)
def update_traces(model, states, traces, x, y_true, loss, lambda_=0.9):
  """states=the original states at the beginning of the step"""
  #save current states
  saved_states = get_states(model)
  #set to pre-step state
  set_states(model, states)
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss_value = loss(y_true, y_pred)
    grads = tape.gradient(loss_value, model.trainable_variables)
    for grad, trace in zip(grads, traces):
      trace.assign(trace * lambda_ + grad * (1 - lambda_))# + tf.clip_by_norm(grad, clip_norm = clip_norm))
  #now that we calculated our traces, restore state to the old post-step state
  #  (this probably doesnt need to be done unless the model is stochastic or something...)
  set_states(model, saved_states)

def step_weights(model, traces, lr, reward, optimizer=None):
  if optimizer == None:
    #step direction
    alpha = lr * reward
    for weight, trace in zip(model.trainable_variables, traces):
      weight.assign_add(-trace * alpha) #gradient descent modulated by reward
  else:
    traces2 = [t * reward for t in traces] #modulate by reward
    #then apply normally
    optimizer.apply_gradients(zip(traces2, model.trainable_variables))

## UTILS ##
def categorical_crossentropy(y_true, y_pred):
  #yes, this is (probably) bad practice
  _epsilon = 1e-7 #TODO: hardcoded epsilon
  #use keras trick to recover logits from softmax:
  #https://github.com/keras-team/keras/issues/11801
  y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
  y_pred = tf.log(y_pred)
  return tf.losses.softmax_cross_entropy(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
  #yes, this is (probably) bad practice
  _epsilon = 1e-7 #TODO: hardcoded epsilon
  #use keras trick to recover logits from softmax:
  #https://github.com/keras-team/keras/issues/11801
  y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
  y_pred = tf.log(y_pred)
  return tf.losses.sigmoid_cross_entropy(y_true, y_pred)

def explore_continuous(x, noise_stdev=0.1):
  return x + np.random.normal(0, noise_stdev, x.shape).astype("float32") #y u output float64

def explore_discrete(x, epsilon=0.01):
  #choose randomly
  action = np.random.choice(len(x), p=x)
  #add eps.greedy on top
  if np.random.random() < epsilon:
    action = np.random.choice(len(x))

  return to_categorical(action, len(x))

def explore_multibinary(xs, epsilon=0.01):
  action = [np.random.choice([0, 1], p=[x, 1-x]) for x in xs]
  action = [np.random.choice([0, 1]) if np.random.random() < epsilon else x for x in action]

  return np.array(action)

#TODO: tuple and multidiscrete
