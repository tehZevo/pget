import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from pget import Agent

tf.enable_eager_execution()

model = Sequential([
  Dense(32, input_shape=[8], activation="relu"),
  Dense(32, activation="relu"),
  Dense(4, activation="softmax"),
])

agent = Agent(model, action_type="discrete")
agent.model.summary()

s = np.random.random(size=[8])
a = agent.get_action(s)
agent.train(1)
