import tensorflow as tf
import numpy as np

from ml_utils.keras import get_states, set_states, apply_regularization
from ml_utils.model_builders import dense_stack

from .pget import create_traces, update_traces, step_weights
from .pget import explore_continuous, explore_discrete, explore_multibinary

#TODO: saving/loading?
#TODO: args/kwargs for get_action/train, maybe accept "done" in train

class Agent():
  """Note: requires TF eager"""
  def __init__(self, model, action_type="continuous", alt_trace_method=False,
    epsilon=1e-7, advantage_clip=1, gamma=0.99, lr=1e-4, lambda_=0.9,
    regularization_scale=1e-4, optimizer="adam", noise=0.1, initial_deviation=10):
    self.model = model

    #TODO: is this needed?
    self.input_shape = tuple(self.model.input_shape[1:])
    self.output_shape = tuple(self.model.output_shape[1:])

    #hyperparameters
    self.eps = epsilon
    self.advantage_clip = advantage_clip
    self.gamma = gamma
    self.lr = lr
    self.lambda_ = lambda_
    self.alt_trace_method = alt_trace_method
    self.regularization = regularization_scale * self.lr
    self.noise = noise
    self.last_advantage = 0

    #TODO: support more optimizers by name... or by object
    self.optimizer = None if optimizer is None else tf.train.AdamOptimizer(self.lr)

    #resolve exploration method/loss function
    self.action_type = action_type.lower()

    if self.action_type == "discrete":
      self.loss = tf.keras.losses.categorical_crossentropy
      explore_func = explore_discrete
    elif self.action_type == "multibinary":
      self.loss = tf.keras.losses.binary_crossentropy
      explore_func = explore_multibinary
    elif self.action_type == "continuous":
      #TODO: try huber loss again?
      self.loss = tf.losses.mean_squared_error
      explore_func = explore_continuous
    else:
      raise ValueError("Unknown action type '{}'".format(action_type))

    self.explore = lambda x: explore_func(x, self.noise)

    #initialization
    self.traces = create_traces(self.model)
    self.reward_mean = 0
    self.reward_deviation = initial_deviation

  def get_action(self, state):
    #housekeeping
    state = state.astype("float32")
    #save pre-step hidden state
    pre_step_state = get_states(self.model)
    #calc action from state
    action = self.model.predict(np.expand_dims(state, 0))[0]

    #apply noise to action
    action = self.explore(action)

    #TODO: early bail?

    #calc gradient for modified action & update traces based on gradient
    update_traces(self.model, pre_step_state, self.traces,
      np.expand_dims(state, 0), np.expand_dims(action, 0), self.loss, lambda_=self.lambda_)

    return action

  def train(self, reward):
    #scale/clip reward to calculate advantage
    delta_reward = reward - self.reward_mean
    advantage = delta_reward / (self.reward_deviation + self.eps)
    if self.advantage_clip is not None:
      advantage = np.clip(advantage, -self.advantage_clip, self.advantage_clip)

    #update reward mean/deviation
    self.reward_mean += delta_reward * (1 - self.gamma)
    self.reward_deviation += (np.abs(delta_reward) - self.reward_deviation) * (1 - self.gamma)
    self.last_advantage = advantage

    #step network in direction of trace gradient * lr * reward
    apply_regularization(self.model, self.regularization)
    step_weights(self.model, self.traces, self.lr, advantage, self.optimizer)
