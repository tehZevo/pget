import tensorflow as tf
import numpy as np

#for RNN compatibility
from ml_utils.rnn import get_states, set_states

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

def step_weights(model, traces, lr, reward):
  if optimizer == None:
    #step direction
    alpha = lr * reward
    for weight, trace in zip(model.trainable_variables, traces):
      weight.assign_add(-trace * alpha) #gradient descent modulated by reward
  else:
    traces2 = [t * reward for t in traces] #modulate by reward
    #then apply normally
    optimizer.apply_gradients(zip(traces2, model.trainable_variables))
